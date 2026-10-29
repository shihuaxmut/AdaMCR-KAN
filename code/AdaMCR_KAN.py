# -*- coding: utf-8 -*-


import math
import os
import warnings
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def _load_vocab_size():
    candidates = [
        os.path.join(os.getcwd(), "vocab_dict.npy"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab_dict.npy"),
    ]

    for path in candidates:
        if os.path.exists(path):
            vocab = np.load(path, allow_pickle=True).item()
            return len(vocab)

    raise FileNotFoundError(
        "Cannot find vocab_dict.npy. Put it in the working directory "
        "or in the same directory as this module."
    )


vocab_size = _load_vocab_size()


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()
    pad_attn_mask = seq_k.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class SequenceFFTSpectralGate(nn.Module):


    def __init__(
        self,
        dim=64,
        num_heads=1,
        dropout=0.1,
        max_seq_len=64,
        fft_norm="ortho",
    ):
        super(SequenceFFTSpectralGate, self).__init__()
        del num_heads

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.max_freq_len = max_seq_len // 2 + 1
        self.fft_norm = fft_norm

        # Near-identity initialization in the frequency branch.
        self.gain_real = nn.Parameter(
            torch.full((1, dim, self.max_freq_len), 0.1)
        )
        self.gain_imag = nn.Parameter(
            torch.zeros(1, dim, self.max_freq_len)
        )

        self.channel_mixer = nn.Linear(dim, dim, bias=False)
        self.activation = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout)
        self.spectral_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, valid_mask=None):
        if valid_mask is not None:
            x = x * valid_mask.to(
                dtype=x.dtype,
                device=x.device,
            ).unsqueeze(-1)

        _, seq_len, channels = x.shape
        if channels != self.dim:
            raise ValueError(
                f"Expected feature dimension {self.dim}, got {channels}."
            )
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len "
                f"{self.max_seq_len}."
            )

        spectrum = torch.fft.rfft(
            x.transpose(1, 2).contiguous(),
            dim=-1,
            norm=self.fft_norm,
        )
        freq_len = spectrum.size(-1)

        complex_gain = torch.complex(
            self.gain_real[:, :, :freq_len],
            self.gain_imag[:, :, :freq_len],
        ).to(dtype=spectrum.dtype)

        filtered = spectrum * complex_gain
        out = torch.fft.irfft(
            filtered,
            n=seq_len,
            dim=-1,
            norm=self.fft_norm,
        ).transpose(1, 2).contiguous()

        out = self.channel_mixer(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.spectral_scale * out

        if valid_mask is not None:
            out = out * valid_mask.to(
                dtype=out.dtype,
                device=out.device,
            ).unsqueeze(-1)

        return out


class LocalSequenceCNN(nn.Module):
    def __init__(self, d_model=64, kernel_size=3, dropout=0.1):
        super(LocalSequenceCNN, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        self.dwconv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        self.pwconv = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, valid_mask=None):
        residual = x
        out = x.transpose(1, 2).contiguous()
        out = self.dwconv(out)
        out = self.act(out)
        out = self.pwconv(out)
        out = out.transpose(1, 2).contiguous()
        out = self.dropout(out)
        out = self.layer_norm(residual + out)

        if valid_mask is not None:
            out = out * valid_mask.to(dtype=out.dtype).unsqueeze(-1)
        return out


class ChebyKANLinear(nn.Module):
    """Chebyshev-polynomial channel mixer."""

    def __init__(self, input_dim, output_dim, degree=2):
        super(ChebyKANLinear, self).__init__()
        if degree < 1:
            raise ValueError("degree must be >= 1.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.epsilon = 1e-6

        self.cheby_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )
        nn.init.normal_(
            self.cheby_coeffs,
            mean=0.0,
            std=1.0 / (input_dim * (degree + 1)),
        )
        self.register_buffer(
            "arange",
            torch.arange(0, degree + 1, dtype=torch.float32),
        )

    def forward(self, x):
        original_shape = x.shape[:-1]
        x = x.reshape(-1, self.input_dim)

    
        x = torch.tanh(x)
        x = torch.clamp(x, -1.0 + self.epsilon, 1.0 - self.epsilon)
        theta = torch.acos(x).unsqueeze(-1)
        basis = torch.cos(theta * self.arange.to(dtype=x.dtype))

        out = torch.einsum("bid,iod->bo", basis, self.cheby_coeffs)
        return out.reshape(*original_shape, self.output_dim)


class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features, order=2):
        super(ChebyKANLayer, self).__init__()
        self.fc1 = ChebyKANLinear(in_features, out_features, order)

    def forward(self, x):
        return self.fc1(x)


class BasicConv(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size=3,
        degree=2,
        stride=1,
        dilation=1,
        groups=1,
        act=False,
        bn=False,
        bias=False,
        dropout=0.0,
    ):
        super(BasicConv, self).__init__()
        del degree
        self.bn = nn.BatchNorm1d(c_in) if bn else None
        self.conv = nn.Conv1d(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.act = nn.ReLU(inplace=False) if act else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x.transpose(1, 2).contiguous()
        if self.bn is not None:
            out = self.bn(out)
        out = self.conv(out)
        if self.act is not None:
            out = self.act(out)
        out = self.dropout(out)
        return out.transpose(1, 2).contiguous()


class M_KAN(nn.Module):


    def __init__(self, d_model=64, seq_len=None, order=2, dropout=0.1):
        super(M_KAN, self).__init__()
        del seq_len

        self.channel_mixer = ChebyKANLayer(d_model, d_model, order)
        self.conv = BasicConv(
            d_model,
            d_model,
            kernel_size=3,
            degree=order,
            groups=d_model,
            dropout=0.0,
        )
        self.act = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        channel_out = self.channel_mixer(x)
        local_out = self.conv(x)
        return self.dropout(self.act(channel_out + local_out))


class MKANResidualBlock(nn.Module):
    def __init__(self, d_model=64, order=2, dropout=0.1):
        super(MKANResidualBlock, self).__init__()
        self.mkan = M_KAN(
            d_model=d_model,
            order=order,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, valid_mask=None):
        out = self.layer_norm(x + self.mkan(x))
        if valid_mask is not None:
            out = out * valid_mask.to(dtype=out.dtype).unsqueeze(-1)
        return out


class PoswiseFeedForwardNet(nn.Module):
    

    def __init__(self, d_model=64, d_ff=512, dropout=0.1, use_cuda=use_cuda):
        super(PoswiseFeedForwardNet, self).__init__()
        del use_cuda
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs, valid_mask=None):
        output = self.layer_norm(inputs + self.fc(inputs))
        if valid_mask is not None:
            output = output * valid_mask.to(dtype=output.dtype).unsqueeze(-1)
        return output


class MKANPoswiseFeedForwardNet(nn.Module):
    

    def __init__(
        self,
        d_model=64,
        d_ff=512,
        dropout=0.1,
        use_cuda=use_cuda,
        mkan_order=2,
    ):
        super(MKANPoswiseFeedForwardNet, self).__init__()
        del d_ff, use_cuda
        self.mkan = M_KAN(
            d_model=d_model,
            order=mkan_order,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs, valid_mask=None):
        output = self.layer_norm(inputs + self.mkan(inputs))
        if valid_mask is not None:
            output = output * valid_mask.to(dtype=output.dtype).unsqueeze(-1)
        return output


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=64,
        d_ff=512,
        n_heads=1,
        dropout=0.1,
        cnn_kernel_size=3,
        mkan_order=2,
    ):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = SequenceFFTSpectralGate(
            dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
        )
        self.fft_norm = nn.LayerNorm(d_model)

        self.feature_enhancer = LocalSequenceCNN(
            d_model=d_model,
            kernel_size=cnn_kernel_size,
            dropout=dropout,
        )

        self.pos_ffn = MKANPoswiseFeedForwardNet(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            mkan_order=mkan_order,
        )

    def forward(self, enc_inputs, enc_self_attn_mask=None, valid_mask=None):
        del enc_self_attn_mask

        spectral_output = self.enc_self_attn(
            enc_inputs,
            valid_mask=valid_mask,
        )
        enc_outputs = self.fft_norm(enc_inputs + spectral_output)

        if valid_mask is not None:
            enc_outputs = enc_outputs * valid_mask.to(
                dtype=enc_outputs.dtype,
                device=enc_outputs.device,
            ).unsqueeze(-1)

        enc_outputs = self.feature_enhancer(
            enc_outputs,
            valid_mask=valid_mask,
        )
        enc_outputs = self.pos_ffn(
            enc_outputs,
            valid_mask=valid_mask,
        )
        return enc_outputs, None


class HRIEncoder(nn.Module):
    def __init__(
        self,
        d_model=64,
        n_layers=1,
        d_ff=512,
        d_k=64,
        d_v=64,
        n_heads=1,
        dropout=0.1,
        use_cuda=use_cuda,
        cnn_kernel_size=3,
        mkan_order=2,
    ):
        super(HRIEncoder, self).__init__()
        del d_k, d_v, use_cuda

        self.src_emb = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=0,
        )

        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                dropout=dropout,
                cnn_kernel_size=cnn_kernel_size,
                mkan_order=mkan_order,
            )
            for _ in range(n_layers)
        ])

    def forward(self, enc_inputs):
        valid_token_mask = enc_inputs.ne(0)
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = enc_outputs * valid_token_mask.to(
            dtype=enc_outputs.dtype,
            device=enc_outputs.device,
        ).unsqueeze(-1)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(
                enc_outputs,
                valid_mask=valid_token_mask,
            )
            enc_outputs = enc_outputs * valid_token_mask.to(
                dtype=enc_outputs.dtype,
                device=enc_outputs.device,
            ).unsqueeze(-1)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k=64):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.bool(), -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v), attn



class MultiHeadAttention(nn.Module):

    def __init__(self, d_model=64, d_k=64, d_v=64, n_heads=1, dropout=0.1, use_cuda=False):
        super(MultiHeadAttention, self).__init__()
        del d_k, d_v, use_cuda
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.head_gate = nn.Linear(d_model, n_heads, bias=True)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        nn.init.zeros_(self.head_gate.weight)
        nn.init.zeros_(self.head_gate.bias)

    def forward(self, input_Q, input_K, input_V, attn_mask=None, query_mask=None):
        residual = input_Q
        batch_size, q_len, _ = input_Q.shape
        k_len = input_K.size(1)
        q = self.W_Q(input_Q).view(batch_size, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(input_K).view(batch_size, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(input_V).view(batch_size, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attn_mask is not None:
            expanded_mask = attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            scores = scores.masked_fill(expanded_mask.bool(), -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        context = torch.matmul(attn, v)
        head_weights = torch.softmax(self.head_gate(input_Q), dim=-1)
        head_weights = (head_weights * self.n_heads).transpose(1, 2).unsqueeze(-1)
        context = context * head_weights
        context = context.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        output = self.fc(context)
        output = self.output_dropout(output)
        output = self.layer_norm(residual + output)
        if query_mask is not None:
            output = output * (~query_mask).to(dtype=output.dtype, device=output.device).unsqueeze(-1)
        return output, attn

class CrossInteractionGate(nn.Module):

    def __init__(self, d_model=64, dropout=0.1):
        super(CrossInteractionGate, self).__init__()
        self.gate_proj = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -1.0)

    def forward(self, original, cross, valid_mask=None):
        gate_input = torch.cat(
            [original, cross, original * cross, torch.abs(original - cross)],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_proj(gate_input))
        delta = self.dropout(cross - original)
        output = self.layer_norm(original + gate * delta)

        if valid_mask is not None:
            output = output * valid_mask.to(
                dtype=output.dtype,
                device=output.device,
            ).unsqueeze(-1)
        return output


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=64,
        d_ff=512,
        d_k=64,
        d_v=64,
        n_heads=1,
        dropout=0.1,
        mkan_order=2,
    ):
        super(DecoderLayer, self).__init__()

        self.pep_to_hla_attn = MultiHeadAttention(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.hla_to_pep_attn = MultiHeadAttention(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.pep_cross_gate = CrossInteractionGate(
            d_model=d_model,
            dropout=dropout,
        )
        self.hla_cross_gate = CrossInteractionGate(
            d_model=d_model,
            dropout=dropout,
        )

        self.pos_ffn = MKANPoswiseFeedForwardNet(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            mkan_order=mkan_order,
        )

    def forward(self, pep_inputs, hla_inputs):
        batch_size = pep_inputs.size(0)
        pep_len = pep_inputs.size(1)
        hla_len = hla_inputs.size(1)

        pep_key_mask = pep_inputs.abs().sum(dim=-1).eq(0)
        hla_key_mask = hla_inputs.abs().sum(dim=-1).eq(0)

        pep_to_hla_mask = hla_key_mask.unsqueeze(1).expand(
            batch_size, pep_len, hla_len
        )
        hla_to_pep_mask = pep_key_mask.unsqueeze(1).expand(
            batch_size, hla_len, pep_len
        )

        pep_outputs, pep_to_hla_attn = self.pep_to_hla_attn(
            pep_inputs,
            hla_inputs,
            hla_inputs,
            pep_to_hla_mask,
            query_mask=pep_key_mask,
        )
        hla_outputs, hla_to_pep_attn = self.hla_to_pep_attn(
            hla_inputs,
            pep_inputs,
            pep_inputs,
            hla_to_pep_mask,
            query_mask=hla_key_mask,
        )

        pep_outputs = self.pep_cross_gate(
            pep_inputs,
            pep_outputs,
            valid_mask=~pep_key_mask,
        )
        hla_outputs = self.hla_cross_gate(
            hla_inputs,
            hla_outputs,
            valid_mask=~hla_key_mask,
        )

        dec_outputs = torch.cat([pep_outputs, hla_outputs], dim=1)
        valid_mask = torch.cat(
            [~pep_key_mask, ~hla_key_mask],
            dim=1,
        )
        dec_outputs = self.pos_ffn(
            dec_outputs,
            valid_mask=valid_mask,
        )
        return dec_outputs, (pep_to_hla_attn, hla_to_pep_attn)


class ABGC_Decoder(nn.Module):
    def __init__(
        self,
        tgt_len=59,
        d_model=64,
        n_layers=1,
        use_cuda=True,
        d_ff=512,
        d_k=64,
        d_v=64,
        n_heads=1,
        dropout=0.1,
        pep_len=25,
        hla_len=34,
        mkan_order=2,
    ):
        super(ABGC_Decoder, self).__init__()
        del use_cuda

        self.tgt_len = tgt_len
        self.pep_len = pep_len
        self.hla_len = hla_len

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                d_ff=d_ff,
                d_k=d_k,
                d_v=d_v,
                n_heads=n_heads,
                dropout=dropout,
                mkan_order=mkan_order,
            )
            for _ in range(n_layers)
        ])

    def _get_split_len(self, seq_len):
        if self.pep_len is not None and 0 < self.pep_len < seq_len:
            return self.pep_len
        return seq_len // 2

    def forward(self, dec_inputs):
        dec_outputs = dec_inputs
        split_len = self._get_split_len(dec_outputs.size(1))
        dec_self_attns = []

        for layer in self.layers:
            pep_outputs = dec_outputs[:, :split_len]
            hla_outputs = dec_outputs[:, split_len:]
            dec_outputs, dec_attn = layer(pep_outputs, hla_outputs)
            dec_self_attns.append(dec_attn)

        return dec_outputs, dec_self_attns