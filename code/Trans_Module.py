import math
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn



vocab = np.load('vocab_dict.npy', allow_pickle = True).item()
vocab_size = len(vocab)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



def get_attn_pad_mask(seq_q, seq_k, batch_size=1024):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]



class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k=64):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        #scores.masked_fill_(attn_mask, -1e9)  # Apply the mask to the scores
        scores.masked_fill_(attn_mask.bool(), -1e9)  # 将 attn_mask 转换为布尔类型
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, d_k=64, d_v=64, n_heads=1, use_cuda=False):
        super(MultiHeadAttention, self).__init__()
        self.use_cuda = use_cuda
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False).to(self.device)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False).to(self.device)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False).to(self.device)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False).to(self.device)
        self.attention = ScaledDotProductAttention(d_k).to(self.device)
        self.layer_norm = nn.LayerNorm(d_model).to(self.device)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual = input_Q
        batch_size = input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = self.attention(Q, K, V, attn_mask)

        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)

        return self.layer_norm(output + residual), attn



class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model=64, d_ff=512, use_cuda=use_cuda):
        super(PoswiseFeedForwardNet, self).__init__()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        ).to(self.device)
        self.d_model = d_model

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        if not hasattr(self, 'layer_norm'):
            self.layer_norm = nn.LayerNorm(self.d_model).to(self.device)
        return self.layer_norm(output + residual)



class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet(d_model=64, d_ff=512, use_cuda=use_cuda)

    def forward(self, enc_inputs, enc_self_attn_mask):

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, d_model = 64, n_layers = 1):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):

        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


### Decoder


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet(d_model=64, d_ff=512, use_cuda=use_cuda)

    def forward(self, dec_inputs, dec_self_attn_mask):  # dec_inputs = enc_outputs
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn


class Decoder(nn.Module):
    def __init__(self, tgt_len=86,d_model = 64,n_layers=1,use_cuda=True):
        super(Decoder, self).__init__()
        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.tgt_len = tgt_len

    def forward(self, dec_inputs):  # dec_inputs = enc_outputs (batch_size, peptide_hla_maxlen_sum, d_model)
        dec_outputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1).to(device)
        dec_self_attn_pad_mask = torch.LongTensor(np.zeros((dec_inputs.shape[0], self.tgt_len, self.tgt_len))).byte().to(device)

        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_pad_mask)
            dec_self_attns.append(dec_self_attn)

        return dec_outputs, dec_self_attns



