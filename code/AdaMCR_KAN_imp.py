# -*- coding: utf-8 -*-

import inspect
import os
import random

import torch
import torch.nn as nn

random.seed(2026)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

from AdaMCR_KAN import HRIEncoder, ABGC_Decoder


def build_module(module_cls, **kwargs):
    signature = inspect.signature(module_cls.__init__)
    valid_keys = set(signature.parameters.keys())
    valid_keys.discard("self")
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    return module_cls(**filtered_kwargs)


class AdaMCR_KAN_MHC(nn.Module):
    def __init__(
        self,
        use_cuda=True,
        tgt_len=59,
        d_model=64,
        d_ff=512,
        d_k=64,
        d_v=64,
        n_layers=1,
        n_heads=1,
        dropout=0.1,
        pep_len=25,
        hla_len=34,
        cnn_kernel_size=3,
        mkan_order=3,
    ):
        super(AdaMCR_KAN_MHC, self).__init__()

        if tgt_len != pep_len + hla_len:
            raise ValueError(
                f"tgt_len ({tgt_len}) must equal pep_len + hla_len "
                f"({pep_len} + {hla_len} = {pep_len + hla_len})."
            )

        self.use_cuda = use_cuda
        self.tgt_len = tgt_len
        self.d_model = d_model
        self.pep_len = pep_len
        self.hla_len = hla_len

        encoder_kwargs = {
            "d_model": d_model,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "d_k": d_k,
            "d_v": d_v,
            "n_heads": n_heads,
            "dropout": dropout,
            "use_cuda": use_cuda,
            "cnn_kernel_size": cnn_kernel_size,
            "mkan_order": mkan_order,
        }

        self.pep_encoder = build_module(HRIEncoder, **encoder_kwargs)
        self.hla_encoder = build_module(HRIEncoder, **encoder_kwargs)

        decoder_kwargs = {
            "tgt_len": tgt_len,
            "d_model": d_model,
            "n_layers": n_layers,
            "use_cuda": use_cuda,
            "d_ff": d_ff,
            "d_k": d_k,
            "d_v": d_v,
            "n_heads": n_heads,
            "dropout": dropout,
            "pep_len": pep_len,
            "hla_len": hla_len,
            "mkan_order": mkan_order,
        }
        self.decoder = build_module(ABGC_Decoder, **decoder_kwargs)

        self.projection = nn.Sequential(
            nn.Linear(tgt_len * d_model, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, pep_inputs, hla_inputs):
        pep_enc_outputs, pep_enc_self_attns = self.pep_encoder(pep_inputs)
        hla_enc_outputs, hla_enc_self_attns = self.hla_encoder(hla_inputs)

        enc_outputs = torch.cat(
            [pep_enc_outputs, hla_enc_outputs],
            dim=1,
        )
        dec_outputs, dec_self_attns = self.decoder(enc_outputs)

        dec_outputs = dec_outputs.contiguous().view(dec_outputs.size(0), -1)
        dec_logits = self.projection(dec_outputs)

        return (
            dec_logits.view(-1, dec_logits.size(-1)),
            pep_enc_self_attns,
            hla_enc_self_attns,
            dec_self_attns,
        )