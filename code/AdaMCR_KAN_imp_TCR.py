import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import inspect
import random
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

random.seed(2026)

import AdaMCR_KAN_imp
from AdaMCR_KAN_imp import AdaMCR_KAN_MHC

from AdaMCR_KAN import HRIEncoder, ABGC_Decoder


def build_module(module_cls, **kwargs):
    """Pass only arguments supported by the target module constructor."""
    signature = inspect.signature(module_cls.__init__)
    valid_keys = set(signature.parameters.keys())
    valid_keys.discard("self")

    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in valid_keys
    }
    return module_cls(**filtered_kwargs)


def _unwrap_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    """Extract and normalize a state_dict from common checkpoint formats."""
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The pretrained AdaMCR_KAN_MHC checkpoint must be a state_dict or a "
            "dictionary containing a state_dict."
        )

    for candidate_key in (
        "state_dict",
        "model_state_dict",
        "model",
        "net",
    ):
        candidate = checkpoint.get(candidate_key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break

    cleaned_state_dict = {}

    for key, value in checkpoint.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned_state_dict[new_key] = value

    if not cleaned_state_dict:
        raise ValueError("The pretrained AdaMCR_KAN_MHC state_dict is empty.")

    return cleaned_state_dict


def _validate_state_shapes(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    checkpoint_name: str,
) -> None:
    """Raise a readable error before load_state_dict on shape mismatch."""
    model_state = model.state_dict()
    mismatches = []

    for key in sorted(set(model_state) & set(state_dict)):
        if model_state[key].shape != state_dict[key].shape:
            mismatches.append(
                (
                    key,
                    tuple(state_dict[key].shape),
                    tuple(model_state[key].shape),
                )
            )

    if mismatches:
        lines = [
            f"{checkpoint_name} is incompatible with the current model "
            "configuration.",
            "Shape mismatches:",
        ]

        for key, checkpoint_shape, model_shape in mismatches[:40]:
            lines.append(
                f"  {key}: checkpoint={checkpoint_shape}, "
                f"current_model={model_shape}"
            )

        lines.extend(
            [
                "",
                "Make sure d_model, n_layers, n_heads, mkan_order, "
                "cnn_kernel_size, peptide length and HLA length are exactly "
                "the same as those used to train the pretrained AdaMCR_KAN_MHC.",
            ]
        )

        raise RuntimeError("\n".join(lines))


class AdaMCR_KAN_TCR(nn.Module):
    """
    TCR-pMHC-I binding model built from the AdaMCR_KAN_MHC architecture.

    Parameters are deliberately aligned with AdaMCR_KAN_MHC. The same values
    from YAML are used for the pretrained pMHC branch, the TCR Encoder and the
    final pMHC-TCR cross Decoder.
    """

    def __init__(
        self,
        use_cuda: bool = True,
        tgt_len: int = 86,
        d_model: int = 64,
        d_ff: int = 512,
        d_k: int = 64,
        d_v: int = 64,
        n_layers: int = 1,
        n_heads: int = 1,
        dropout: float = 0.1,
        pep_len: int = 25,
        hla_len: int = 34,
        tcr_len: int = 27,
        cnn_kernel_size: int = 3,
        mkan_order: int = 2,
        pmhc_model_path: str | None = None,
        freeze_pmhc: bool = True,
        strict_pmhc_load: bool = True,
    ):
        super(AdaMCR_KAN_TCR, self).__init__()

        pmhc_len = pep_len + hla_len
        expected_tgt_len = pmhc_len + tcr_len

        if tgt_len != expected_tgt_len:
            raise ValueError(
                f"tgt_len ({tgt_len}) must equal pep_len + hla_len + tcr_len "
                f"({pep_len} + {hla_len} + {tcr_len} = {expected_tgt_len})."
            )

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads "
                f"({n_heads})."
            )

        if not pmhc_model_path:
            raise ValueError(
                "pmhc_model_path is required. Put the pretrained "
                "AdaMCR_KAN_MHC Best_model.pkl path in the YAML model section."
            )

        self.use_cuda = use_cuda
        self.tgt_len = tgt_len
        self.pmhc_len = pmhc_len
        self.pep_len = pep_len
        self.hla_len = hla_len
        self.tcr_len = tcr_len
        self.d_model = d_model
        self.freeze_pmhc = bool(freeze_pmhc)
        self.pmhc_model_path = str(pmhc_model_path)

        # -------------------------------------------------------------
        # Pretrained pMHC feature extractor.
        # It is created and loaded once, not inside forward().
        # -------------------------------------------------------------
        self.pmhc_model = AdaMCR_KAN_MHC(
            use_cuda=use_cuda,
            tgt_len=pmhc_len,
            d_model=d_model,
            d_ff=d_ff,
            d_k=d_k,
            d_v=d_v,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            pep_len=pep_len,
            hla_len=hla_len,
            cnn_kernel_size=cnn_kernel_size,
            mkan_order=mkan_order,
        )

        self._load_pretrained_pmhc(
            model_path=self.pmhc_model_path,
            strict=strict_pmhc_load,
        )

        if self.freeze_pmhc:
            for parameter in self.pmhc_model.parameters():
                parameter.requires_grad = False
            self.pmhc_model.eval()

        # -------------------------------------------------------------
        # TCR feature extraction: same Encoder and same YAML parameters.
        # -------------------------------------------------------------
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

        self.tcr_encoder = build_module(
            HRIEncoder,
            **encoder_kwargs,
        )

        # -------------------------------------------------------------
        # pMHC <-> TCR interaction decoder.
        # The generic decoder's two segments are configured as:
        #   first segment  = pMHC representation, length pep_len + hla_len
        #   second segment = TCR representation, length tcr_len
        # -------------------------------------------------------------
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
            "pep_len": pmhc_len,
            "hla_len": tcr_len,
            "mkan_order": mkan_order,
        }

        self.decoder = build_module(
            ABGC_Decoder,
            **decoder_kwargs,
        )

        self.projection = nn.Sequential(
            nn.Linear(tgt_len * d_model, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def _load_pretrained_pmhc(
        self,
        model_path: str,
        strict: bool = True,
    ) -> None:
        path = Path(model_path).expanduser()

        if not path.exists():
            raise FileNotFoundError(
                f"Cannot find pretrained AdaMCR_KAN_MHC checkpoint: {path}"
            )

        checkpoint = torch.load(
            str(path),
            map_location="cpu",
        )

        state_dict = _unwrap_state_dict(checkpoint)

        _validate_state_shapes(
            self.pmhc_model,
            state_dict,
            checkpoint_name="Pretrained AdaMCR_KAN_MHC checkpoint",
        )

        load_result = self.pmhc_model.load_state_dict(
            state_dict,
            strict=strict,
        )

        if not strict:
            if load_result.missing_keys:
                print(
                    "Warning: missing pretrained pMHC keys:",
                    load_result.missing_keys,
                )
            if load_result.unexpected_keys:
                print(
                    "Warning: unexpected pretrained pMHC keys:",
                    load_result.unexpected_keys,
                )

    def train(self, mode: bool = True):
        """
        Keep the frozen pretrained pMHC branch in evaluation mode even when
        the outer TransTCR model enters training mode.
        """
        super().train(mode)

        if self.freeze_pmhc:
            self.pmhc_model.eval()

        return self

    def encode_pmhc(
        self,
        pep_inputs: torch.Tensor,
        hla_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, object]:
        """Return the pretrained pMHC decoder representation [B, 59, C]."""

        def _forward_pmhc():
            pep_outputs, pep_attns = self.pmhc_model.pep_encoder(
                pep_inputs
            )
            hla_outputs, hla_attns = self.pmhc_model.hla_encoder(
                hla_inputs
            )

            pmhc_encoder_outputs = torch.cat(
                [pep_outputs, hla_outputs],
                dim=1,
            )

            pmhc_outputs, pmhc_decoder_attns = self.pmhc_model.decoder(
                pmhc_encoder_outputs
            )

            pmhc_attns = {
                "pep_encoder": pep_attns,
                "hla_encoder": hla_attns,
                "pmhc_decoder": pmhc_decoder_attns,
            }
            return pmhc_outputs, pmhc_attns

        if self.freeze_pmhc:
            with torch.no_grad():
                return _forward_pmhc()

        return _forward_pmhc()

    def forward_from_pmhc(
        self,
        tcr_inputs: torch.Tensor,
        pmhc_outputs: torch.Tensor,
        pmhc_attns=None,
    ):
        """
        Classify a TCR against an already-computed pMHC representation.

        This method lets the training loop compute the frozen pMHC branch only
        once for a positive/negative TCR pair, reducing duplicated computation.
        """
        tcr_outputs, tcr_attns = self.tcr_encoder(
            tcr_inputs
        )

        if pmhc_outputs.size(1) != self.pmhc_len:
            raise RuntimeError(
                f"Unexpected pMHC representation length: "
                f"{pmhc_outputs.size(1)}; expected {self.pmhc_len}."
            )

        if tcr_outputs.size(1) != self.tcr_len:
            raise RuntimeError(
                f"Unexpected TCR representation length: "
                f"{tcr_outputs.size(1)}; expected {self.tcr_len}."
            )

        interaction_inputs = torch.cat(
            [pmhc_outputs, tcr_outputs],
            dim=1,
        )

        decoder_outputs, decoder_attns = self.decoder(
            interaction_inputs
        )

        decoder_outputs = decoder_outputs.contiguous().view(
            decoder_outputs.size(0),
            -1,
        )

        logits = self.projection(
            decoder_outputs
        )

        return (
            logits.view(-1, logits.size(-1)),
            tcr_attns,
            pmhc_attns,
            decoder_attns,
        )

    def forward(
        self,
        tcr_inputs: torch.Tensor,
        pep_inputs: torch.Tensor,
        hla_inputs: torch.Tensor,
    ):
        """
        Inputs
        ------
        tcr_inputs: [batch_size, tcr_len]
        pep_inputs: [batch_size, pep_len]
        hla_inputs: [batch_size, hla_len]
        """
        pmhc_outputs, pmhc_attns = self.encode_pmhc(
            pep_inputs,
            hla_inputs,
        )

        return self.forward_from_pmhc(
            tcr_inputs=tcr_inputs,
            pmhc_outputs=pmhc_outputs,
            pmhc_attns=pmhc_attns,
        )