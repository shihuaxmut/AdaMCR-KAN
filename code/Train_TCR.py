# -*- coding: utf-8 -*-
import os

import argparse
import copy
import io
import json
import logging
import os
import random
import shutil
import time
import warnings
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "缺少 PyYAML，请先执行：python -m pip install pyyaml"
    ) from exc

import DataProcessing
from DataProcessing import (
    pMHCI_TCR_test_data_with_loader,
    pMHCI_TCR_test_dataset,
    pMHCI_TCR_train_data_with_loader,
    pMHCI_TCR_train_dataset,
)
import Performance
from Performance import performances
import AdaMCR_KAN_imp_TCR
from AdaMCR_KAN_imp_TCR import AdaMCR_KAN_TCR


warnings.filterwarnings("ignore")

LOCAL_TZ = timezone(timedelta(hours=8))
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = (
    "/home/zqliudachen/AdaMCR-KAN/"
)

DEFAULT_CONFIG_PATH = (
    "/home/zqliudachen/AdaMCR-KAN/"
    "config/"
    "TCR_train.yaml"
)
def now_dt():
    return datetime.now(LOCAL_TZ)


def now_file_time():
    return now_dt().strftime("%Y-%m-%d_%H-%M-%S")


def now_log_time():
    return now_dt().strftime("%Y-%m-%d %H:%M:%S.%f")


# =========================================================
# Reproducibility
# =========================================================
def seed_everything(seed=2026):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# =========================================================
# YAML and path helpers
# =========================================================
def load_yaml_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return {} if config is None else config


def cfg_get(config, keys, default=None):
    for key in keys:
        current = config
        found = True

        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break

        if found:
            return current

    return default


def resolve_path(path):
    if path is None:
        return None

    path = os.path.expanduser(str(path))

    if os.path.isabs(path):
        return path

    candidates = [
        os.path.abspath(os.path.join(CODE_DIR, path)),
        os.path.abspath(os.path.join(PROJECT_ROOT, path)),
        os.path.abspath(path),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return candidates[0]


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AdaMCR_KAN_TCR using a YAML configuration."
    )

    parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG_PATH,
        type=str,
        help="Path to AdaMCR_KAN_TCR best-score training YAML",
    )

    parser.add_argument(
        "-d",
        "--data",
        dest="data",
        default=None,
        type=str,
        help="Override YAML TCR-pMHC test CSV path.",
    )

    parser.add_argument(
        "-pos",
        "--pos_tcr_pmhc",
        dest="pos_tcr_pmhc",
        default=None,
        type=str,
        help="Override YAML positive TCR-pMHC training CSV path.",
    )

    parser.add_argument(
        "-neg",
        "--neg_tcr_pmhc",
        dest="neg_tcr_pmhc",
        default=None,
        type=str,
        help="Override YAML negative TCR pool CSV path.",
    )

    parser.add_argument(
        "-pmhc",
        "--pmhc_model",
        dest="pmhc_model",
        default=None,
        type=str,
        help="Override YAML pretrained TranspMHC Best_model.pkl path.",
    )

    return parser.parse_args()


# =========================================================
# Logging
# =========================================================
class BeijingFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        current = datetime.fromtimestamp(record.created, LOCAL_TZ)
        return (
            current.strftime("%Y-%m-%d %H:%M:%S")
            + f",{int(record.msecs):03d}"
        )


def setup_logger(log_file):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    formatter = BeijingFormatter(
        fmt="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )

    file_handler = logging.FileHandler(
        log_file,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return root


# =========================================================
# General helpers
# =========================================================
def transfer(y_prob, threshold=0.5):
    y_prob = np.asarray(y_prob, dtype=np.float64)
    return (y_prob > float(threshold)).astype(np.int64)


def tensor_loss_mean(loss_list):
    if not loss_list:
        return 0.0

    values = []

    for item in loss_list:
        if torch.is_tensor(item):
            values.append(float(item.detach().cpu().item()))
        else:
            values.append(float(item))

    return sum(values) / len(values)


def count_trainable_params(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_total_params(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def metrics_to_dict(metrics, score=None, threshold=None):
    result = {
        "AUROC": float(metrics[0]),
        "AUPRC": float(metrics[1]),
        "ACC": float(metrics[2]),
        "F1": float(metrics[3]),
        "MCC": float(metrics[4]),
        "sensitivity": float(metrics[5]),
        "specificity": float(metrics[6]),
        "precision": float(metrics[7]),
        "recall": float(metrics[8]),
    }

    if score is not None:
        result["score"] = float(score)

    if threshold is not None:
        result["threshold"] = float(threshold)

    return result


def fold_result_dict(
    metrics,
    fold,
    score,
    best_epoch,
    threshold,
    model_path,
):
    bacc = (
        float(metrics[5])
        + float(metrics[6])
    ) / 2.0

    return {
        "AUC": float(metrics[0]),
        "AUPR": float(metrics[1]),
        "ACC": float(metrics[2]),
        "BACC": float(bacc),
        "F1": float(metrics[3]),
        "MCC": float(metrics[4]),
        "fold": int(fold),
        "best_epoch": int(best_epoch),
        "score": float(score),
        "threshold": float(threshold),
        "model_path": model_path,
    }


def ensure_tensor(value, dtype=torch.long):
    if torch.is_tensor(value):
        return value.detach().cpu().to(dtype=dtype)

    return torch.as_tensor(
        value,
        dtype=dtype,
    )


def repeat_rows(tensor, repeat_count):
    if repeat_count <= 0:
        raise ValueError("repeat_count must be positive.")

    return tensor.repeat_interleave(
        repeat_count,
        dim=0,
    )


def sample_pool_rows(
    pool,
    sample_size,
    generator,
    allow_replacement=True,
):
    """Sample rows from a tensor pool with deterministic torch RNG."""
    pool_size = int(pool.size(0))

    if pool_size == 0:
        raise ValueError("The negative TCR pool is empty.")

    if sample_size <= pool_size:
        indices = torch.randperm(
            pool_size,
            generator=generator,
        )[:sample_size]
        used_replacement = False
    else:
        if not allow_replacement:
            raise ValueError(
                f"Need {sample_size} negative TCR samples but the pool only "
                f"contains {pool_size}. Set train.allow_negative_replacement "
                "to true or provide a larger negative pool."
            )

        indices = torch.randint(
            low=0,
            high=pool_size,
            size=(sample_size,),
            generator=generator,
        )
        used_replacement = True

    return pool[indices], used_replacement


def make_loader_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def select_best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_prob,
    )

    if len(thresholds) == 0:
        return 0.5

    # precision and recall contain one more element than thresholds.
    objective = precision[:-1] * recall[:-1]
    best_index = int(np.argmax(objective))

    return float(thresholds[best_index])


def normalized_auprc(auprc, positive_prevalence):

    prevalence = float(positive_prevalence)

    if not 0.0 <= prevalence < 1.0:
        return float("nan")

    denominator = 1.0 - prevalence

    if denominator <= 0:
        return float("nan")

    return float(
        (float(auprc) - prevalence)
        / denominator
    )


def select_best_mcc_threshold(y_true, y_prob):

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_true.size == 0:
        return 0.5, 0.0

    positive_count = int((y_true == 1).sum())
    negative_count = int((y_true == 0).sum())

    if positive_count == 0 or negative_count == 0:
        return 0.5, 0.0

    order = np.argsort(-y_prob, kind="mergesort")
    sorted_prob = y_prob[order]
    sorted_true = y_true[order]

    cumulative_tp = np.cumsum(sorted_true == 1)
    cumulative_fp = np.cumsum(sorted_true == 0)

    # Only evaluate the end of each equal-probability group.
    group_end = np.r_[
        sorted_prob[:-1] != sorted_prob[1:],
        True,
    ]

    tp = cumulative_tp[group_end].astype(np.float64)
    fp = cumulative_fp[group_end].astype(np.float64)
    fn = positive_count - tp
    tn = negative_count - fp

    numerator = tp * tn - fp * fn
    denominator = np.sqrt(
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    mcc_values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )

    candidate_thresholds = sorted_prob[group_end]
    best_index = int(np.argmax(mcc_values))

    return (
        float(candidate_thresholds[best_index]),
        float(mcc_values[best_index]),
    )


# =========================================================
# Classification losses
# =========================================================
def normalize_loss_name(loss_name):
    name = str(loss_name).strip().lower()

    aliases = {
        "bce": "bce_logits",
        "bcewithlogits": "bce_logits",
        "bce_with_logits": "bce_logits",
        "ce": "cross_entropy",
        "crossentropy": "cross_entropy",
        "label_smooth": "label_smoothing",
        "labelsmoothing": "label_smoothing",
    }

    name = aliases.get(name, name)

    allowed = {
        "bce_logits",
        "cross_entropy",
        "label_smoothing",
        "focal",
    }

    if name not in allowed:
        raise ValueError(
            f"Unsupported loss.name={loss_name!r}. "
            f"Supported losses: {sorted(allowed)}"
        )

    return name


def classification_loss(
    logits,
    labels,
    loss_name,
    class0_weight=1.0,
    class1_weight=1.0,
    positive_weight=1.0,
    label_smoothing=0.0,
    focal_gamma=2.0,
):

    if logits.ndim != 2 or logits.size(1) != 2:
        raise ValueError(
            "The AdaMCR_KAN_TCR classifier must return logits with shape [B, 2], "
            f"but received {tuple(logits.shape)}."
        )

    labels = labels.long().view(-1)
    loss_name = normalize_loss_name(loss_name)

    if loss_name == "bce_logits":
        binary_logits = logits[:, 1] - logits[:, 0]

        pos_weight_tensor = torch.tensor(
            float(positive_weight),
            dtype=logits.dtype,
            device=logits.device,
        )

        return F.binary_cross_entropy_with_logits(
            binary_logits,
            labels.to(dtype=logits.dtype),
            pos_weight=pos_weight_tensor,
        )

    class_weights = torch.tensor(
        [float(class0_weight), float(class1_weight)],
        dtype=logits.dtype,
        device=logits.device,
    )

    if loss_name == "cross_entropy":
        return F.cross_entropy(
            logits,
            labels,
            weight=class_weights,
        )

    if loss_name == "label_smoothing":
        smoothing = float(label_smoothing)

        if not 0.0 <= smoothing < 1.0:
            raise ValueError(
                "loss.label_smoothing must satisfy 0 <= value < 1."
            )

        return F.cross_entropy(
            logits,
            labels,
            weight=class_weights,
            label_smoothing=smoothing,
        )

    # focal
    gamma = float(focal_gamma)

    if gamma < 0:
        raise ValueError("loss.focal_gamma must be non-negative.")

    ce_per_sample = F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        reduction="none",
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )
    pt = probabilities.gather(
        1,
        labels.unsqueeze(1),
    ).squeeze(1)

    return (
        (1.0 - pt).pow(gamma)
        * ce_per_sample
    ).mean()



def train_step(
    model,
    optimizer,
    train_loader,
    device,
    fold,
    epoch,
    epochs,
    loss_name,
    class0_weight,
    class1_weight,
    positive_weight,
    label_smoothing,
    focal_gamma,
    l2_weight,
    grad_clip,
    negative_pool,
    negative_generator,
    train_neg_ratio,
    dynamic_negative_sampling,
):
    model.train()

    loss_list = []
    time_train_epoch = 0.0

    negative_pool_size = int(negative_pool.size(0))

    if negative_pool_size == 0:
        raise ValueError("The negative TCR pool is empty.")

    for (
        positive_tcr,
        peptide,
        hla,
        fixed_negative_tcr,
        labels,
    ) in tqdm(
        train_loader,
        desc=f"Fold {fold} Epoch {epoch}/{epochs} Train",
        leave=False,
    ):
        positive_tcr = positive_tcr.to(
            device,
            non_blocking=True,
        )
        peptide = peptide.to(
            device,
            non_blocking=True,
        )
        hla = hla.to(
            device,
            non_blocking=True,
        )

        positive_batch_size = int(positive_tcr.size(0))
        negative_batch_size = (
            positive_batch_size * int(train_neg_ratio)
        )

        if dynamic_negative_sampling:
            negative_indices = torch.randint(
                low=0,
                high=negative_pool_size,
                size=(negative_batch_size,),
                generator=negative_generator,
            )
            negative_tcr = negative_pool[
                negative_indices
            ].to(
                device,
                non_blocking=True,
            )
        else:
            fixed_negative_tcr = fixed_negative_tcr.to(
                device,
                non_blocking=True,
            )

            if train_neg_ratio != 1:
                negative_tcr = fixed_negative_tcr.repeat(
                    train_neg_ratio,
                    1,
                )
            else:
                negative_tcr = fixed_negative_tcr

        combined_tcr = torch.cat(
            [positive_tcr, negative_tcr],
            dim=0,
        )
        combined_peptide = torch.cat(
            [peptide]
            + [peptide] * int(train_neg_ratio),
            dim=0,
        )
        combined_hla = torch.cat(
            [hla]
            + [hla] * int(train_neg_ratio),
            dim=0,
        )
        combined_labels = torch.cat(
            [
                torch.ones(
                    positive_batch_size,
                    dtype=torch.long,
                    device=device,
                ),
                torch.zeros(
                    negative_batch_size,
                    dtype=torch.long,
                    device=device,
                ),
            ],
            dim=0,
        )

        # Shuffle the mixed batch so no module can exploit class order.
        permutation = torch.randperm(
            combined_tcr.size(0),
            device=device,
        )
        combined_tcr = combined_tcr[permutation]
        combined_peptide = combined_peptide[permutation]
        combined_hla = combined_hla[permutation]
        combined_labels = combined_labels[permutation]

        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)

        combined_logits = model(
            combined_tcr,
            combined_peptide,
            combined_hla,
        )[0]

        task_loss = classification_loss(
            logits=combined_logits,
            labels=combined_labels,
            loss_name=loss_name,
            class0_weight=class0_weight,
            class1_weight=class1_weight,
            positive_weight=positive_weight,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
        )

        trainable_l2 = torch.zeros(
            (),
            device=device,
        )

        if l2_weight > 0:
            for parameter in model.parameters():
                if parameter.requires_grad:
                    trainable_l2 = (
                        trainable_l2
                        + parameter.pow(2).sum()
                    )

        total_loss = (
            task_loss
            + l2_weight * trainable_l2
        )

        total_loss.backward()

        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(
                (
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                max_norm=float(grad_clip),
            )

        optimizer.step()

        time_train_epoch += time.time() - start_time
        loss_list.append(total_loss.detach())

    return loss_list, time_train_epoch


def eval_step(
    model,
    loader,
    device,
    loss_name,
    class0_weight,
    class1_weight,
    positive_weight,
    label_smoothing,
    focal_gamma,
    name="val",
    fixed_threshold=None,
    optimize_mcc_threshold=False,
):
    model.eval()

    y_true_list = []
    y_prob_list = []
    batch_loss_list = []

    with torch.inference_mode():
        for (
            tcr_inputs,
            peptide_inputs,
            hla_inputs,
            labels,
        ) in tqdm(
            loader,
            desc=f"Eval {name}",
            leave=False,
        ):
            tcr_inputs = tcr_inputs.to(
                device,
                non_blocking=True,
            )
            peptide_inputs = peptide_inputs.to(
                device,
                non_blocking=True,
            )
            hla_inputs = hla_inputs.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(
                tcr_inputs,
                peptide_inputs,
                hla_inputs,
            )[0]

            loss = classification_loss(
                logits=logits,
                labels=labels,
                loss_name=loss_name,
                class0_weight=class0_weight,
                class1_weight=class1_weight,
                positive_weight=positive_weight,
                label_smoothing=label_smoothing,
                focal_gamma=focal_gamma,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            batch_loss_list.append(
                float(loss.detach().cpu().item())
            )
            y_true_list.extend(
                labels.cpu().numpy().tolist()
            )
            y_prob_list.extend(
                probabilities.cpu().numpy().tolist()
            )

    average_loss = (
        float(np.mean(batch_loss_list))
        if batch_loss_list
        else float("inf")
    )

    selected_mcc = None

    if fixed_threshold is not None:
        threshold = float(fixed_threshold)
    elif optimize_mcc_threshold:
        threshold, selected_mcc = select_best_mcc_threshold(
            y_true_list,
            y_prob_list,
        )
    else:
        threshold = 0.5

    y_pred_list = transfer(
        y_prob_list,
        threshold,
    )

    performance_stdout = io.StringIO()

    with redirect_stdout(performance_stdout):
        metrics = performances(
            y_true_list,
            y_pred_list,
            y_prob_list,
            print_=True,
        )

    performance_text = performance_stdout.getvalue().strip()

    return (
        (y_true_list, y_pred_list, y_prob_list),
        average_loss,
        metrics,
        threshold,
        performance_text,
        selected_mcc,
    )


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    seed = int(
        cfg_get(config, ["train.seed", "seed"], 2026)
    )
    seed_everything(seed)

    mixed_batch_size = int(
        cfg_get(config, ["train.batch_size", "batch_size"], 768)
    )

    if mixed_batch_size < 2:
        raise ValueError(
            "train.batch_size must be at least 2."
        )
    epochs = int(
        cfg_get(config, ["train.epochs", "epochs"], 15)
    )
    learning_rate = float(
        cfg_get(
            config,
            [
                "train.learning_rate",
                "train.lr",
                "learning_rate",
                "lr",
            ],
            5e-4,
        )
    )
    k_folds = int(
        cfg_get(config, ["train.k_folds", "k_folds"], 5)
    )
    pos_repeat = int(
        cfg_get(config, ["train.pos_repeat", "pos_repeat"], 3)
    )
    val_neg_ratio = int(
        cfg_get(
            config,
            ["train.val_neg_ratio", "val_neg_ratio"],
            3,
        )
    )
    num_workers = int(
        cfg_get(config, ["train.num_workers", "num_workers"], 0)
    )
    allow_negative_replacement = bool(
        cfg_get(
            config,
            [
                "train.allow_negative_replacement",
                "allow_negative_replacement",
            ],
            True,
        )
    )

    max_folds_raw = cfg_get(
        config,
        ["train.max_folds", "max_folds"],
        k_folds,
    )
    max_folds = int(max_folds_raw)

    if max_folds < 1 or max_folds > k_folds:
        raise ValueError(
            "train.max_folds must satisfy "
            f"1 <= max_folds <= k_folds ({k_folds})."
        )

    weight_decay = float(
        cfg_get(
            config,
            ["train.weight_decay", "weight_decay"],
            5e-6,
        )
    )
    grad_clip = float(
        cfg_get(
            config,
            ["train.grad_clip", "grad_clip"],
            1.0,
        )
    )

    scheduler_patience = int(
        cfg_get(
            config,
            ["scheduler.patience", "train.scheduler_patience"],
            4,
        )
    )
    scheduler_cooldown = int(
        cfg_get(
            config,
            ["scheduler.cooldown", "train.cooldown"],
            4,
        )
    )
    scheduler_min_lr = float(
        cfg_get(
            config,
            ["scheduler.min_lr", "train.min_lr"],
            1e-6,
        )
    )
    scheduler_factor = float(
        cfg_get(
            config,
            ["scheduler.factor", "train.scheduler_factor", "train.factor"],
            0.2,
        )
    )
    loss_delta = float(
        cfg_get(
            config,
            ["early_stopping.delta", "train.loss_delta"],
            1e-4,
        )
    )
    early_stopping_patience = int(
        cfg_get(
            config,
            ["early_stopping.patience", "train.callback_patience"],
            10,
        )
    )
    checkpoint_period = int(
        cfg_get(
            config,
            ["checkpoint.period", "train.period"],
            1,
        )
    )

    train_neg_ratio = int(
        cfg_get(
            config,
            ["train.train_neg_ratio", "train_neg_ratio"],
            1,
        )
    )
    dynamic_negative_sampling = bool(
        cfg_get(
            config,
            [
                "train.dynamic_negative_sampling",
                "dynamic_negative_sampling",
            ],
            True,
        )
    )

    if train_neg_ratio < 1:
        raise ValueError(
            "train.train_neg_ratio must be at least 1."
        )

    samples_per_anchor = 1 + train_neg_ratio

    if mixed_batch_size % samples_per_anchor != 0:
        raise ValueError(
            "train.batch_size must be divisible by "
            f"(1 + train_neg_ratio)={samples_per_anchor}. "
            f"Received batch_size={mixed_batch_size}."
        )

    train_anchor_batch_size = (
        mixed_batch_size // samples_per_anchor
    )

    if train_anchor_batch_size < 1:
        raise ValueError(
            "The derived positive-anchor batch size must be at least 1."
        )

    # -------------------------
    # Sequence lengths
    # -------------------------
    pep_max_len = int(
        cfg_get(config, ["data.pep_max_len", "pep_max_len"], 25)
    )
    hla_max_len = int(
        cfg_get(config, ["data.hla_max_len", "hla_max_len"], 34)
    )
    tcr_max_len = int(
        cfg_get(config, ["data.tcr_max_len", "tcr_max_len"], 27)
    )
    tgt_len = pep_max_len + hla_max_len + tcr_max_len

    d_model = int(
        cfg_get(config, ["model.d_model", "d_model"], 64)
    )
    d_ff = int(
        cfg_get(config, ["model.d_ff", "d_ff"], 512)
    )
    d_k = int(
        cfg_get(config, ["model.d_k", "d_k"], 64)
    )
    d_v = int(
        cfg_get(config, ["model.d_v", "d_v"], 64)
    )
    n_layers = int(
        cfg_get(config, ["model.n_layers", "n_layers"], 1)
    )
    n_heads = int(
        cfg_get(config, ["model.n_heads", "n_heads"], 1)
    )
    dropout = float(
        cfg_get(config, ["model.dropout", "dropout"], 0.1)
    )
    mkan_order = int(
        cfg_get(config, ["model.mkan_order", "mkan_order"], 2)
    )
    cnn_kernel_size = int(
        cfg_get(
            config,
            ["model.cnn_kernel_size", "cnn_kernel_size"],
            3,
        )
    )
    pmhc_model_path = cfg_get(
        config,
        [
            "model.pmhc_model_path",
            "model.pretrained_pmhc_path",
            "pmhc_model_path",
            "pretrained_pmhc_path",
        ],
        None,
    )

    if args.pmhc_model is not None:
        pmhc_model_path = args.pmhc_model

    if pmhc_model_path is None:
        raise ValueError(
            "The YAML must contain model.pmhc_model_path pointing to the "
            "pretrained TranspMHC Best_model.pkl."
        )

    # -------------------------
    # Loss parameters
    # -------------------------
    loss_name = normalize_loss_name(
        cfg_get(
            config,
            ["loss.name", "loss_name"],
            "bce_logits",
        )
    )
    class0_weight = float(
        cfg_get(
            config,
            ["loss.class0_weight", "class0_weight"],
            1.0,
        )
    )
    class1_weight = float(
        cfg_get(
            config,
            ["loss.class1_weight", "class1_weight"],
            1.0,
        )
    )
    positive_weight = float(
        cfg_get(
            config,
            ["loss.positive_weight", "positive_weight"],
            1.0,
        )
    )
    label_smoothing = float(
        cfg_get(
            config,
            ["loss.label_smoothing", "label_smoothing"],
            0.05,
        )
    )
    focal_gamma = float(
        cfg_get(
            config,
            ["loss.focal_gamma", "focal_gamma"],
            2.0,
        )
    )
    l2_weight = float(
        cfg_get(
            config,
            ["loss.l2_weight", "l2_weight"],
            0.0,
        )
    )

    threshold = float(
        cfg_get(config, ["eval.threshold", "threshold"], 0.5)
    )

    # -------------------------
    # Data paths
    # -------------------------
    positive_path = cfg_get(
        config,
        [
            "data.pos_tcr_pmhc",
            "data.pos",
            "pos_tcr_pmhc",
            "pos",
        ],
        "../data/pmhc_tcr_train_pos.csv",
    )
    negative_path = cfg_get(
        config,
        [
            "data.neg_tcr_pmhc",
            "data.neg",
            "neg_tcr_pmhc",
            "neg",
        ],
        "../data/pmhc_tcr_train_neg.csv",
    )
    test_path = cfg_get(
        config,
        [
            "data.tcr_test",
            "data.test",
            "test_data",
        ],
        "../data/pmhc_tcr_test.csv",
    )
    vocab_path = cfg_get(
        config,
        ["data.vocab", "vocab", "vocab_path"],
        "vocab_dict.npy",
    )

    if args.pos_tcr_pmhc is not None:
        positive_path = args.pos_tcr_pmhc
    if args.neg_tcr_pmhc is not None:
        negative_path = args.neg_tcr_pmhc
    if args.data is not None:
        test_path = args.data

    positive_path = resolve_path(positive_path)
    negative_path = resolve_path(negative_path)
    test_path = resolve_path(test_path)
    vocab_path = resolve_path(vocab_path)
    pmhc_model_path = resolve_path(pmhc_model_path)

    for path, description in (
        (positive_path, "Positive training data"),
        (negative_path, "Negative TCR pool"),
        (test_path, "Test data"),
        (vocab_path, "Vocabulary"),
        (pmhc_model_path, "Pretrained TranspMHC checkpoint"),
    ):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"{description} not found: {path}"
            )

    use_cuda = torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda else "cpu"
    )

    # -------------------------
    # Run paths
    # -------------------------
    current_time = now_file_time()
    run_name = f"AdaMCR_KAN_TCR_BEST_SCORE_RUN_{current_time}"

    log_dir = os.path.join(
        PROJECT_ROOT,
        "result",
        "log",
    )
    model_dir = os.path.join(
        PROJECT_ROOT,
        "model",
        "TCR",
        run_name,
    )
    prediction_dir = os.path.join(
        PROJECT_ROOT,
        "result",
        "TCR",
        run_name,
    )

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)

    log_path = os.path.join(
        log_dir,
        f"{run_name}_train_log.txt",
    )

    setup_logger(log_path)

    root_logger = logging.getLogger("root")
    experiment_logger = logging.getLogger("Experiment Name")
    description_logger = logging.getLogger("Description")
    tags_logger = logging.getLogger("Tags")
    fold_begin_logger = logging.getLogger("Begin training model")
    parameter_logger = logging.getLogger("Model parameters")
    training_logger = logging.getLogger("Training process")
    validation_logger = logging.getLogger("val metric")
    test_metric_logger = logging.getLogger("test metric")
    validation_raw_logger = logging.getLogger("val raw")
    test_raw_logger = logging.getLogger("test raw")
    best_logger = logging.getLogger("Best model")
    score_summary_logger = logging.getLogger("Epoch score summary")
    fold_result_logger = logging.getLogger("Fold result")
    final_result_logger = logging.getLogger("Fold final result")

    # -------------------------
    # Load data
    # -------------------------
    vocab = np.load(
        vocab_path,
        allow_pickle=True,
    ).item()

    positive_data = pd.read_csv(positive_path).drop_duplicates()
    negative_data = pd.read_csv(negative_path)
    test_data = pd.read_csv(test_path)

    (
        positive_tcr_inputs,
        peptide_inputs,
        hla_inputs,
        negative_tcr_inputs,
        train_labels,
        _,
    ) = pMHCI_TCR_train_data_with_loader(
        positive_data,
        negative_data,
        vocab,
        batch_size=mixed_batch_size,
        pep_max_len=pep_max_len,
        hla_max_len=hla_max_len,
        tcr_max_len=tcr_max_len,
        shuffle=False,
    )

    (
        test_tcr_inputs,
        test_peptide_inputs,
        test_hla_inputs,
        test_labels,
        test_loader,
    ) = pMHCI_TCR_test_data_with_loader(
        test_data,
        vocab,
        batch_size=mixed_batch_size,
        pep_max_len=pep_max_len,
        hla_max_len=hla_max_len,
        tcr_max_len=tcr_max_len,
        shuffle=False,
    )

    positive_tcr_inputs = ensure_tensor(positive_tcr_inputs)
    peptide_inputs = ensure_tensor(peptide_inputs)
    hla_inputs = ensure_tensor(hla_inputs)
    negative_tcr_inputs = ensure_tensor(negative_tcr_inputs)
    train_labels = ensure_tensor(train_labels)
    test_labels = ensure_tensor(test_labels)

    test_positive_count = int(
        (test_labels.view(-1) == 1).sum().item()
    )
    test_negative_count = int(
        (test_labels.view(-1) == 0).sum().item()
    )
    test_total_count = (
        test_positive_count + test_negative_count
    )
    test_positive_prevalence = (
        test_positive_count / test_total_count
        if test_total_count > 0
        else float("nan")
    )
    test_negative_to_positive_ratio = (
        test_negative_count / test_positive_count
        if test_positive_count > 0
        else float("inf")
    )

    # -------------------------
    # Basic log
    # -------------------------
    experiment_logger.info(
        cfg_get(
            config,
            ["experiment.name", "name"],
            "AdaMCR_KAN_TCR_train",
        )
    )
    description_logger.info(
        cfg_get(
            config,
            ["experiment.description", "description"],
            "AdaMCR_KAN_TCR using pretrained AdaMCR_KAN_MHC features",
        )
    )
    tags_logger.info(
        cfg_get(
            config,
            ["experiment.tags", "tags"],
            ["AdaMCR_KAN_TCR"],
        )
    )

    root_logger.info("Run name: %s", run_name)
    root_logger.info("Config path: %s", args.config)
    root_logger.info("Fixed project root: %s", PROJECT_ROOT)
    root_logger.info(
        "Fixed model base directory: %s",
        os.path.join(PROJECT_ROOT, "model", "TCR"),
    )
    root_logger.info(
        "Fixed log directory: %s",
        os.path.join(PROJECT_ROOT, "result", "log"),
    )
    root_logger.info("Start time: %s", now_log_time())
    root_logger.info("Log file: %s", log_path)
    root_logger.info("Model directory: %s", model_dir)
    root_logger.info("Prediction directory: %s", prediction_dir)
    root_logger.info("Device: %s", device)

    if use_cuda:
        root_logger.info(
            "GPU: %s",
            torch.cuda.get_device_name(0),
        )

    root_logger.info("Positive data: %s", positive_path)
    root_logger.info("Negative pool: %s", negative_path)
    root_logger.info("Test data: %s", test_path)
    root_logger.info("Vocabulary: %s", vocab_path)
    root_logger.info(
        "Pretrained TranspMHC: %s",
        pmhc_model_path,
    )
    root_logger.info(
        "Positive samples after duplicate removal: %d",
        len(positive_data),
    )
    root_logger.info(
        "Negative pool samples: %d",
        len(negative_data),
    )
    root_logger.info("Test samples: %d", len(test_data))
    root_logger.info(
        "Test label distribution: positive=%d, negative=%d, "
        "positive_prevalence=%.6f, negative_to_positive_ratio=%.6f, "
        "random_AUPRC_baseline=%.6f",
        test_positive_count,
        test_negative_count,
        test_positive_prevalence,
        test_negative_to_positive_ratio,
        test_positive_prevalence,
    )

    configured_validation_prevalence = (
        1.0 / (1.0 + val_neg_ratio)
    )

    root_logger.info(
        "Configured validation distribution: positive:negative=1:%d, "
        "positive_prevalence=%.6f, random_AUPRC_baseline=%.6f",
        val_neg_ratio,
        configured_validation_prevalence,
        configured_validation_prevalence,
    )

    if (
        np.isfinite(test_negative_to_positive_ratio)
        and abs(
            test_negative_to_positive_ratio
            - float(val_neg_ratio)
        ) > 0.5
    ):
        root_logger.warning(
            "Validation ratio 1:%d differs from the test "
            "negative-to-positive ratio %.3f. Set val_neg_ratio "
            "close to the test/deployment ratio for unbiased selection.",
            val_neg_ratio,
            test_negative_to_positive_ratio,
        )

    parameter_logger.info(
        "mixed_batch_size=%d, train_anchor_batch_size=%d, "
        "train_neg_ratio=%d, epochs=%d, learning_rate=%s, k_folds=%d, "
        "pos_repeat=%d, val_neg_ratio=%d, pep_len=%d, hla_len=%d, "
        "tcr_len=%d, tgt_len=%d, d_model=%d, d_ff=%d, d_k=%d, "
        "d_v=%d, n_layers=%d, n_heads=%d, dropout=%.4f, "
        "mkan_order=%d, cnn_kernel_size=%d",
        mixed_batch_size,
        train_anchor_batch_size,
        train_neg_ratio,
        epochs,
        learning_rate,
        k_folds,
        pos_repeat,
        val_neg_ratio,
        pep_max_len,
        hla_max_len,
        tcr_max_len,
        tgt_len,
        d_model,
        d_ff,
        d_k,
        d_v,
        n_layers,
        n_heads,
        dropout,
        mkan_order,
        cnn_kernel_size,
    )

    parameter_logger.info(
        "Loss implementation: cross_entropy uses the two class logits "
        "directly; bce_logits uses binary_logit=logits[:,1]-logits[:,0]"
    )

    parameter_logger.info(
        "Loss: name=%s, class0_weight=%s, class1_weight=%s, "
        "positive_weight=%s, label_smoothing=%s, focal_gamma=%s, "
        "l2_weight=%s",
        loss_name,
        class0_weight,
        class1_weight,
        positive_weight,
        label_smoothing,
        focal_gamma,
        l2_weight,
    )
    parameter_logger.info(
        "Monitor definition: best_score = "
        "(validation_AUROC + validation_AUPRC) / 2"
    )
    parameter_logger.info(
        "Scheduler, early stopping, fold checkpoint and "
        "global Best_model.pkl all monitor best_score."
    )

    parameter_logger.info(
        "Optimization: weight_decay=%s, grad_clip=%s, "
        "scheduler_patience=%d, cooldown=%d, min_lr=%s, factor=%s, "
        "early_stop_patience=%d, delta=%s, checkpoint_period=%d, "
        "max_folds=%s, train_neg_ratio=%d, "
        "dynamic_negative_sampling=%s",
        weight_decay,
        grad_clip,
        scheduler_patience,
        scheduler_cooldown,
        scheduler_min_lr,
        scheduler_factor,
        early_stopping_patience,
        loss_delta,
        checkpoint_period,
        max_folds,
        train_neg_ratio,
        dynamic_negative_sampling,
    )

    # -------------------------
    # K-fold
    # -------------------------
    kfold = KFold(
        n_splits=k_folds,
        shuffle=True,
        random_state=seed,
    )

    fold_summary_list = []

    for fold, (train_indices, validation_indices) in enumerate(
        kfold.split(positive_tcr_inputs)
    ):
        if max_folds is not None and fold >= max_folds:
            break

        fold_seed = seed + fold
        seed_everything(fold_seed)

        fold_generator = make_loader_generator(fold_seed)
        sample_generator = make_loader_generator(fold_seed + 10000)
        dynamic_negative_generator = make_loader_generator(
            fold_seed + 20000
        )

        fold_logger = logging.getLogger(f"Fold {fold}")
        fold_begin_logger.info("%d", fold)
        fold_logger.info(
            "Train positive samples=%d, validation positive samples=%d",
            len(train_indices),
            len(validation_indices),
        )

        train_indices_tensor = torch.as_tensor(
            train_indices,
            dtype=torch.long,
        )
        validation_indices_tensor = torch.as_tensor(
            validation_indices,
            dtype=torch.long,
        )

        fold_positive_tcr = repeat_rows(
            positive_tcr_inputs[train_indices_tensor],
            pos_repeat,
        )
        fold_peptide = repeat_rows(
            peptide_inputs[train_indices_tensor],
            pos_repeat,
        )
        fold_hla = repeat_rows(
            hla_inputs[train_indices_tensor],
            pos_repeat,
        )
        fold_labels = repeat_rows(
            train_labels[train_indices_tensor],
            pos_repeat,
        )

        fold_negative_tcr, train_used_replacement = sample_pool_rows(
            negative_tcr_inputs,
            sample_size=fold_positive_tcr.size(0),
            generator=sample_generator,
            allow_replacement=allow_negative_replacement,
        )

        fold_logger.info(
            "Training negative sampling replacement: %s",
            train_used_replacement,
        )

        train_dataset = pMHCI_TCR_train_dataset(
            fold_positive_tcr,
            fold_peptide,
            fold_hla,
            fold_negative_tcr,
            fold_labels,
        )

        # BatchNorm1d in the classifier cannot train on a final batch of size 1.
        # Drop only that exceptional one-sample remainder.
        drop_last_train = (
            len(train_dataset) > train_anchor_batch_size
            and len(train_dataset) % train_anchor_batch_size == 1
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_anchor_batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=fold_generator,
            pin_memory=use_cuda,
            drop_last=drop_last_train,
        )

        fold_logger.info(
            "Training DataLoader drop_last: %s",
            drop_last_train,
        )

        # Validation: each positive pair plus val_neg_ratio negative TCRs.
        validation_positive_tcr = positive_tcr_inputs[
            validation_indices_tensor
        ]
        validation_peptide = peptide_inputs[
            validation_indices_tensor
        ]
        validation_hla = hla_inputs[
            validation_indices_tensor
        ]

        validation_negative_count = (
            validation_positive_tcr.size(0)
            * val_neg_ratio
        )

        validation_negative_tcr, val_used_replacement = sample_pool_rows(
            negative_tcr_inputs,
            sample_size=validation_negative_count,
            generator=sample_generator,
            allow_replacement=allow_negative_replacement,
        )

        fold_logger.info(
            "Validation negative sampling replacement: %s",
            val_used_replacement,
        )

        validation_tcr = torch.cat(
            [
                validation_positive_tcr,
                validation_negative_tcr,
            ],
            dim=0,
        )

        validation_peptide_all = torch.cat(
            [validation_peptide] * (val_neg_ratio + 1),
            dim=0,
        )
        validation_hla_all = torch.cat(
            [validation_hla] * (val_neg_ratio + 1),
            dim=0,
        )

        validation_labels = torch.cat(
            [
                torch.ones(
                    validation_positive_tcr.size(0),
                    dtype=torch.long,
                ),
                torch.zeros(
                    validation_negative_tcr.size(0),
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )

        validation_dataset = pMHCI_TCR_test_dataset(
            validation_tcr,
            validation_peptide_all,
            validation_hla_all,
            validation_labels,
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=mixed_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )

        model = AdaMCR_KAN_TCR(
            use_cuda=use_cuda,
            tgt_len=tgt_len,
            d_model=d_model,
            d_ff=d_ff,
            d_k=d_k,
            d_v=d_v,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            pep_len=pep_max_len,
            hla_len=hla_max_len,
            tcr_len=tcr_max_len,
            cnn_kernel_size=cnn_kernel_size,
            mkan_order=mkan_order,
            pmhc_model_path=pmhc_model_path,
        ).to(device)

        optimizer = optim.Adam(
            (
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            threshold=loss_delta,
            threshold_mode="abs",
            patience=scheduler_patience,
            cooldown=scheduler_cooldown,
            min_lr=scheduler_min_lr,
            factor=scheduler_factor,
        )

        fold_logger.info(
            "Total parameters: %d",
            count_total_params(model),
        )
        fold_logger.info(
            "Trainable parameters: %d",
            count_trainable_params(model),
        )

        # Unified monitor:
        # best_score = (validation_AUROC + validation_AUPRC) / 2
        best_score = float("-inf")
        best_validation_auroc = float("-inf")
        best_validation_auprc = float("-inf")
        best_validation_loss = float("inf")
        best_validation_threshold = 0.5
        best_validation_mcc = float("-inf")
        best_epoch_round = 0
        best_model_state = None
        epochs_without_improvement = 0
        epoch_score_list = []

        last_model_path = os.path.join(
            model_dir,
            f"{run_name}_last_model_fold{fold}.pkl",
        )

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            loss_list, train_time = train_step(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                device=device,
                fold=fold,
                epoch=epoch,
                epochs=epochs,
                loss_name=loss_name,
                class0_weight=class0_weight,
                class1_weight=class1_weight,
                positive_weight=positive_weight,
                label_smoothing=label_smoothing,
                focal_gamma=focal_gamma,
                l2_weight=l2_weight,
                grad_clip=grad_clip,
                negative_pool=negative_tcr_inputs,
                negative_generator=dynamic_negative_generator,
                train_neg_ratio=train_neg_ratio,
                dynamic_negative_sampling=dynamic_negative_sampling,
            )

            train_loss = tensor_loss_mean(loss_list)

            (
                validation_outputs,
                validation_loss,
                validation_metrics,
                validation_threshold,
                validation_text,
                validation_threshold_mcc,
            ) = eval_step(
                model=model,
                loader=validation_loader,
                device=device,
                loss_name=loss_name,
                class0_weight=class0_weight,
                class1_weight=class1_weight,
                positive_weight=positive_weight,
                label_smoothing=label_smoothing,
                focal_gamma=focal_gamma,
                name="val",
                optimize_mcc_threshold=True,
            )

            validation_score = sum(
                validation_metrics[:2]
            ) / 2.0

            validation_positive_prevalence = (
                1.0 / (1.0 + val_neg_ratio)
            )
            validation_normalized_auprc = normalized_auprc(
                validation_metrics[1],
                validation_positive_prevalence,
            )

            # Monitor validation best_score rather than AUPRC alone.
            current_score = float(validation_score)
            scheduler.step(current_score)

            current_lr = float(
                optimizer.param_groups[0]["lr"]
            )

            epoch_duration = time.time() - epoch_start

            epoch_score_list.append(
                {
                    "fold": fold,
                    "epoch": epoch - 1,
                    "epoch_round": epoch,
                    "learning_rate": current_lr,
                    "train_loss": float(train_loss),
                    "val_loss": float(validation_loss),
                    "val_AUROC": float(validation_metrics[0]),
                    "val_AUPRC": float(validation_metrics[1]),
                    "val_AUPRC_random_baseline": float(
                        validation_positive_prevalence
                    ),
                    "val_normalized_AUPRC": float(
                        validation_normalized_auprc
                    ),
                    "val_score": float(validation_score),
                    "monitor_best_score": float(current_score),
                    "val_ACC": float(validation_metrics[2]),
                    "val_F1": float(validation_metrics[3]),
                    "val_MCC": float(validation_metrics[4]),
                    "val_threshold": float(validation_threshold),
                    "threshold_search_MCC": float(
                        validation_threshold_mcc
                        if validation_threshold_mcc is not None
                        else validation_metrics[4]
                    ),
                }
            )

            training_logger.info(
                "[Fold %d][Epoch %04d | round %02d/%02d] "
                "time=%d s, train_time=%d s, lr=%.8f, "
                "train_loss=%.6f, val_loss=%.6f, "
                "val_AUPRC=%.6f, normalized_AUPRC=%.6f, "
                "best_score=%.6f, val_threshold=%.6f",
                fold,
                epoch - 1,
                epoch,
                epochs,
                int(epoch_duration),
                int(train_time),
                current_lr,
                train_loss,
                validation_loss,
                float(validation_metrics[1]),
                validation_normalized_auprc,
                validation_score,
                validation_threshold,
            )

            validation_logger.info(
                "%s",
                metrics_to_dict(
                    validation_metrics,
                    score=validation_score,
                    threshold=validation_threshold,
                ),
            )

            if validation_text:
                validation_raw_logger.info(
                    "%s",
                    validation_text.replace("\n", " | "),
                )

            # Checkpoint, scheduler and early stopping all monitor:
            # best_score = (validation_AUROC + validation_AUPRC) / 2.
            improved = (
                current_score
                > best_score + loss_delta
            )

            if improved:
                best_score = float(current_score)
                best_validation_auroc = float(
                    validation_metrics[0]
                )
                best_validation_auprc = float(
                    validation_metrics[1]
                )
                best_validation_loss = float(
                    validation_loss
                )
                best_validation_threshold = float(
                    validation_threshold
                )
                best_validation_mcc = float(
                    validation_metrics[4]
                )
                best_epoch_round = int(epoch)
                best_model_state = copy.deepcopy(
                    model.state_dict()
                )
                epochs_without_improvement = 0

                best_logger.info(
                    "Fold %d new best: epoch=%04d, "
                    "best_score=%.6f, val_AUROC=%.6f, "
                    "val_AUPRC=%.6f, val_loss=%.6f, "
                    "val_MCC=%.6f, threshold=%.6f",
                    fold,
                    best_epoch_round - 1,
                    best_score,
                    best_validation_auroc,
                    best_validation_auprc,
                    best_validation_loss,
                    best_validation_mcc,
                    best_validation_threshold,
                )
            else:
                epochs_without_improvement += 1

                best_logger.info(
                    "Fold %d no best_score improvement: epoch=%04d, "
                    "counter=%d/%d, current_score=%.6f, "
                    "best_score=%.6f",
                    fold,
                    epoch - 1,
                    epochs_without_improvement,
                    early_stopping_patience,
                    current_score,
                    best_score,
                )

            if (
                checkpoint_period > 0
                and epoch % checkpoint_period == 0
            ):
                torch.save(
                    model.state_dict(),
                    last_model_path,
                )

            if (
                epochs_without_improvement
                >= early_stopping_patience
            ):
                best_logger.info(
                    "Fold %d early stopped at epoch=%04d. "
                    "Best epoch=%04d, best_score=%.6f",
                    fold,
                    epoch - 1,
                    best_epoch_round - 1,
                    best_score,
                )
                break

        # -------------------------
        # Save epoch summary
        # -------------------------
        epoch_score_df = pd.DataFrame(epoch_score_list)
        epoch_score_csv = os.path.join(
            prediction_dir,
            f"{run_name}_fold{fold}_epoch_score_summary.csv",
        )
        epoch_score_df.to_csv(
            epoch_score_csv,
            index=False,
        )

        score_summary_logger.info(
            "Fold %d epoch summary saved: %s",
            fold,
            epoch_score_csv,
        )

        if best_model_state is None:
            raise RuntimeError(
                f"Fold {fold} did not produce a best model."
            )

        best_epoch_zero_based = best_epoch_round - 1

        fold_model_name = (
            f"{run_name}_best_model_fold{fold}_"
            f"epoch{best_epoch_zero_based:04d}_"
            f"score{best_score:.4f}.pkl"
        )
        fold_model_path = os.path.join(
            model_dir,
            fold_model_name,
        )

        torch.save(
            best_model_state,
            fold_model_path,
        )

        best_logger.info(
            "Fold %d best model saved: %s",
            fold,
            fold_model_path,
        )

        # -------------------------
        # Retest fold best model
        # -------------------------
        model.load_state_dict(
            best_model_state,
            strict=True,
        )

        (
            retest_outputs,
            retest_loss,
            retest_metrics,
            retest_threshold,
            retest_text,
            _,
        ) = eval_step(
            model=model,
            loader=test_loader,
            device=device,
            loss_name=loss_name,
            class0_weight=class0_weight,
            class1_weight=class1_weight,
            positive_weight=positive_weight,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            name="test",
            fixed_threshold=best_validation_threshold,
        )

        retest_score = sum(retest_metrics[:2]) / 2.0

        fold_result_logger.info(
            "%s",
            fold_result_dict(
                metrics=retest_metrics,
                fold=fold,
                score=retest_score,
                best_epoch=best_epoch_zero_based,
                threshold=retest_threshold,
                model_path=fold_model_path,
            ),
        )

        prediction_df = pd.DataFrame()

        for column in (
            "tcr",
            "peptide",
            "hla",
            "HLA_sequence",
            "label",
        ):
            if column in test_data.columns:
                prediction_df[column] = test_data[column]

        prediction_df["prob"] = retest_outputs[2]
        prediction_df["pred_label"] = retest_outputs[1]
        prediction_df["threshold"] = retest_threshold

        fold_prediction_csv = os.path.join(
            prediction_dir,
            f"{run_name}_prediction_fold{fold}_"
            f"epoch{best_epoch_zero_based:04d}_"
            f"score{retest_score:.4f}.csv",
        )
        prediction_df.to_csv(
            fold_prediction_csv,
            index=False,
        )

        fold_summary_list.append(
            {
                "fold": fold,
                "best_epoch": best_epoch_zero_based,
                "best_epoch_round": best_epoch_round,
                "best_score": float(best_score),
                "score_definition": (
                    "score = "
                    "(validation_AUROC + validation_AUPRC) / 2"
                ),
                "best_validation_AUROC": float(
                    best_validation_auroc
                ),
                "best_validation_AUPRC": float(
                    best_validation_auprc
                ),
                "best_validation_loss": float(best_validation_loss),
                "best_validation_MCC": float(best_validation_mcc),
                "best_validation_threshold": float(
                    best_validation_threshold
                ),
                "best_test_score": float(retest_score),
                "test_loss": float(retest_loss),
                "best_test_threshold": float(retest_threshold),
                "test_AUROC": float(retest_metrics[0]),
                "test_AUPRC": float(retest_metrics[1]),
                "test_ACC": float(retest_metrics[2]),
                "test_F1": float(retest_metrics[3]),
                "test_MCC": float(retest_metrics[4]),
                "test_sensitivity": float(retest_metrics[5]),
                "test_specificity": float(retest_metrics[6]),
                "test_precision": float(retest_metrics[7]),
                "test_recall": float(retest_metrics[8]),
                "pkl_path": fold_model_path,
                "prediction_csv": fold_prediction_csv,
                "epoch_score_csv": epoch_score_csv,
            }
        )

        del model
        if use_cuda:
            torch.cuda.empty_cache()

    # =====================================================
    # Final summary and global Best_model.pkl
    # =====================================================
    if not fold_summary_list:
        raise RuntimeError("No fold results were generated.")

    summary_df = pd.DataFrame(fold_summary_list)
    summary_path = os.path.join(
        prediction_dir,
        f"{run_name}_fold_summary.csv",
    )
    summary_df.to_csv(summary_path, index=False)

    root_logger.info("Fold summary saved: %s", summary_path)

    for item in fold_summary_list:
        final_result_logger.info(
            "%s",
            fold_result_dict(
                metrics=[
                    item["test_AUROC"],
                    item["test_AUPRC"],
                    item["test_ACC"],
                    item["test_F1"],
                    item["test_MCC"],
                    item["test_sensitivity"],
                    item["test_specificity"],
                    item["test_precision"],
                    item["test_recall"],
                ],
                fold=item["fold"],
                score=item["best_score"],
                best_epoch=item["best_epoch"],
                threshold=item["best_test_threshold"],
                model_path=item["pkl_path"],
            ),
        )

    best_overall = max(
        fold_summary_list,
        key=lambda item: item["best_score"],
    )

    final_best_model_path = os.path.join(
        model_dir,
        "Best_model.pkl",
    )
    shutil.copyfile(
        best_overall["pkl_path"],
        final_best_model_path,
    )

    final_best_prediction_path = os.path.join(
        prediction_dir,
        "Best_model_prediction.csv",
    )
    shutil.copyfile(
        best_overall["prediction_csv"],
        final_best_prediction_path,
    )

    best_model_info = {
        "best_fold": int(best_overall["fold"]),
        "best_epoch": int(best_overall["best_epoch"]),
        "best_epoch_round": int(best_overall["best_epoch_round"]),
        "best_score": float(
            best_overall["best_score"]
        ),
        "score_definition": (
            "score = "
            "(validation_AUROC + validation_AUPRC) / 2"
        ),
        "selection_definition": (
            "maximum validation best_score"
        ),
        "best_validation_AUROC": float(
            best_overall["best_validation_AUROC"]
        ),
        "best_validation_AUPRC": float(
            best_overall["best_validation_AUPRC"]
        ),
        "best_validation_loss": float(
            best_overall["best_validation_loss"]
        ),
        "best_validation_MCC": float(
            best_overall["best_validation_MCC"]
        ),
        "best_validation_threshold": float(
            best_overall["best_validation_threshold"]
        ),
        "best_test_score": float(
            best_overall["best_test_score"]
        ),
        "test_score_definition": (
            "test_score = "
            "(test_AUROC + test_AUPRC) / 2"
        ),
        "source_model_path": best_overall["pkl_path"],
        "Best_model_path": final_best_model_path,
        "Best_model_prediction_csv": final_best_prediction_path,
        "pretrained_TranspMHC_path": pmhc_model_path,
        "pmhc_transfer_mode": "frozen pretrained feature extractor",
        "test_AUROC": float(best_overall["test_AUROC"]),
        "test_AUPRC": float(best_overall["test_AUPRC"]),
        "test_ACC": float(best_overall["test_ACC"]),
        "test_F1": float(best_overall["test_F1"]),
        "test_MCC": float(best_overall["test_MCC"]),
        "test_sensitivity": float(best_overall["test_sensitivity"]),
        "test_specificity": float(best_overall["test_specificity"]),
        "test_precision": float(best_overall["test_precision"]),
        "test_recall": float(best_overall["test_recall"]),
        "best_test_threshold": float(best_overall["best_test_threshold"]),
    }

    best_info_json_path = os.path.join(
        model_dir,
        "Best_model_info.json",
    )
    with open(
        best_info_json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_model_info,
            file,
            indent=4,
            ensure_ascii=False,
        )

    best_info_csv_path = os.path.join(
        model_dir,
        "Best_model_info.csv",
    )
    pd.DataFrame([best_model_info]).to_csv(
        best_info_csv_path,
        index=False,
    )

    metric_columns = [
        "best_score",
        "best_validation_AUROC",
        "best_validation_AUPRC",
        "best_validation_loss",
        "best_validation_MCC",
        "best_test_score",
        "test_loss",
        "test_AUROC",
        "test_AUPRC",
        "test_ACC",
        "test_F1",
        "test_MCC",
    ]

    mean_metrics = summary_df[metric_columns].mean().to_dict()
    std_metrics = summary_df[metric_columns].std().to_dict()

    root_logger.info("Mean metrics: %s", mean_metrics)
    root_logger.info("Std metrics: %s", std_metrics)
    root_logger.info("Best model: %s", final_best_model_path)
    root_logger.info("Best model info JSON: %s", best_info_json_path)
    root_logger.info("Best model info CSV: %s", best_info_csv_path)
    root_logger.info("Training finished at: %s", now_log_time())

    print("Training finished.")
    print("Log file:", log_path)
    print("Model folder:", model_dir)
    print("Result folder:", prediction_dir)
    print("Best model:", final_best_model_path)
    print("Fold mean metrics:", mean_metrics)
    print("Fold std metrics:", std_metrics)


if __name__ == "__main__":
    main()