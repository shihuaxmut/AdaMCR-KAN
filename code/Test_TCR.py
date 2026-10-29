from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "缺少 PyYAML，请执行：python -m pip install pyyaml"
    ) from exc

from DataProcessing import pMHCI_TCR_test_data_with_loader
from Performance import performances
from AdaMCR_KAN_imp_TCR import AdaMCR_KAN_TCR


warnings.filterwarnings("ignore")

LOCAL_TZ = timezone(timedelta(hours=8))
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "TCR_test.yaml"
)

def now_file_time() -> str:
    return datetime.now(LOCAL_TZ).strftime("%Y-%m-%d_%H-%M-%S")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return {} if config is None else config


def cfg_get(
    config: Dict[str, Any],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    for key in keys:
        current: Any = config
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


def resolve_input_path(
    path: str | os.PathLike | None,
) -> str | None:
    if path is None:
        return None

    expanded = Path(path).expanduser()

    if expanded.is_absolute():
        return str(expanded)

    candidates = [
        CODE_DIR / expanded,
        PROJECT_ROOT / expanded,
        Path.cwd() / expanded,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return str(candidates[0].resolve())


def resolve_output_path(
    path: str | os.PathLike,
) -> str:
    expanded = Path(path).expanduser()

    if expanded.is_absolute():
        return str(expanded)

    return str(
        (CODE_DIR / expanded).resolve()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test one manually specified "
            "AdaMCR_KAN_TCR Best_model.pkl."
        )
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="测试 YAML 路径。",
    )

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=None,
        help="覆盖 YAML 中 data.tcr_test。",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help=(
            "覆盖 YAML 中 model.pkl_path，"
            "直接指定要测试的 Best_model.pkl。"
        ),
    )

    parser.add_argument(
        "--pmhc-model",
        type=str,
        default=None,
        help="覆盖 YAML 中 model.pmhc_model_path。",
    )

    return parser.parse_args()


# =========================================================
# Checkpoint
# =========================================================
def safe_torch_load(
    checkpoint_path: str,
    device: torch.device,
) -> Any:
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location=device,
        )


def unwrap_state_dict(
    checkpoint: Any,
) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint 必须是 state_dict，"
            "或包含 state_dict 的字典。"
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

    cleaned_state_dict: Dict[str, torch.Tensor] = {}

    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned_state_dict[new_key] = value

    if not cleaned_state_dict:
        raise ValueError(
            "Checkpoint 中没有可用的 state_dict。"
        )

    return cleaned_state_dict


def validate_state_dict(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[list[str], list[str]]:
    model_state = model.state_dict()

    missing_keys = sorted(
        set(model_state) - set(state_dict)
    )
    unexpected_keys = sorted(
        set(state_dict) - set(model_state)
    )

    shape_mismatches = []

    for key in sorted(
        set(model_state) & set(state_dict)
    ):
        if model_state[key].shape != state_dict[key].shape:
            shape_mismatches.append(
                (
                    key,
                    tuple(state_dict[key].shape),
                    tuple(model_state[key].shape),
                )
            )

    if shape_mismatches:
        lines = [
            "Checkpoint 与当前测试模型形状不一致：",
        ]

        for (
            key,
            checkpoint_shape,
            model_shape,
        ) in shape_mismatches[:50]:
            lines.append(
                f"  {key}: "
                f"checkpoint={checkpoint_shape}, "
                f"model={model_shape}"
            )

        lines.extend(
            [
                "",
                "请检查测试 YAML 与训练 YAML 中：",
                "d_model、d_ff、d_k、d_v、n_layers、n_heads、",
                "dropout、mkan_order、cnn_kernel_size、",
                "pep_max_len、hla_max_len、tcr_max_len。",
            ]
        )

        raise RuntimeError("\n".join(lines))

    return missing_keys, unexpected_keys


def load_manual_best_model(
    model: torch.nn.Module,
    model_path: str,
    device: torch.device,
    strict_load: bool,
) -> None:
    checkpoint = safe_torch_load(
        model_path,
        device,
    )

    state_dict = unwrap_state_dict(
        checkpoint
    )

    missing_keys, unexpected_keys = (
        validate_state_dict(
            model,
            state_dict,
        )
    )

    if strict_load and (
        missing_keys or unexpected_keys
    ):
        raise RuntimeError(
            "严格加载失败。\n"
            f"Model path: {model_path}\n"
            f"Missing keys: {missing_keys}\n"
            f"Unexpected keys: {unexpected_keys}"
        )

    load_result = model.load_state_dict(
        state_dict,
        strict=strict_load,
    )

    print("已加载手动指定的模型：")
    print(model_path)
    print("Missing keys:", load_result.missing_keys)
    print("Unexpected keys:", load_result.unexpected_keys)


# =========================================================
# 指标与损失
# =========================================================
def transfer(
    y_prob: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return (
        np.asarray(y_prob, dtype=np.float64)
        >= float(threshold)
    ).astype(np.int64)


def classification_loss_sum(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_name: str,
    class0_weight: float,
    class1_weight: float,
    positive_weight: float,
) -> torch.Tensor:
    normalized_name = str(loss_name).lower()

    if normalized_name in {
        "cross_entropy",
        "ce",
    }:
        class_weights = torch.tensor(
            [class0_weight, class1_weight],
            dtype=logits.dtype,
            device=logits.device,
        )

        return F.cross_entropy(
            logits,
            labels.long(),
            weight=class_weights,
            reduction="sum",
        )

    if normalized_name in {
        "bce",
        "bce_logits",
        "bcewithlogits",
        "bce_with_logits",
    }:
        binary_logits = (
            logits[:, 1]
            - logits[:, 0]
        )

        pos_weight = torch.tensor(
            positive_weight,
            dtype=logits.dtype,
            device=logits.device,
        )

        return F.binary_cross_entropy_with_logits(
            binary_logits,
            labels.float(),
            pos_weight=pos_weight,
            reduction="sum",
        )

    raise ValueError(
        "loss.name 只能是 cross_entropy 或 bce_logits。"
    )


def predict_test_set(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_name: str,
    class0_weight: float,
    class1_weight: float,
    positive_weight: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()

    y_true_list: list[int] = []
    y_prob_list: list[float] = []

    loss_sum = 0.0
    sample_count = 0

    with torch.inference_mode():
        for (
            tcr_inputs,
            peptide_inputs,
            hla_inputs,
            labels,
        ) in tqdm(
            loader,
            desc="Testing",
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

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            batch_loss = classification_loss_sum(
                logits=logits,
                labels=labels,
                loss_name=loss_name,
                class0_weight=class0_weight,
                class1_weight=class1_weight,
                positive_weight=positive_weight,
            )

            loss_sum += float(
                batch_loss.detach().cpu().item()
            )
            sample_count += int(
                labels.numel()
            )

            y_true_list.extend(
                labels.detach().cpu().numpy().tolist()
            )
            y_prob_list.extend(
                probabilities.detach().cpu().numpy().tolist()
            )

    if sample_count == 0:
        raise RuntimeError(
            "测试 DataLoader 没有返回样本。"
        )

    return (
        np.asarray(
            y_true_list,
            dtype=np.int64,
        ),
        np.asarray(
            y_prob_list,
            dtype=np.float64,
        ),
        loss_sum / sample_count,
    )


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    test_loss: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    y_pred = transfer(
        y_prob,
        threshold,
    )

    metrics = performances(
        y_true,
        y_pred,
        y_prob,
        print_=False,
    )

    sensitivity = float(metrics[5])
    specificity = float(metrics[6])
    auroc = float(metrics[0])
    auprc = float(metrics[1])

    metrics_dict = {
        "test_loss": float(test_loss),
        "AUROC": auroc,
        "AUPRC": auprc,
        "score": (auroc + auprc) / 2.0,
        "ACC": float(metrics[2]),
        "BACC": (
            sensitivity + specificity
        ) / 2.0,
        "F1": float(metrics[3]),
        "MCC": float(metrics[4]),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": float(metrics[7]),
        "recall": float(metrics[8]),
        "threshold": float(threshold),
    }

    return y_pred, metrics_dict


# =========================================================
# 主函数
# =========================================================
def main() -> None:
    args = parse_args()
    config = load_yaml_config(
        args.config
    )

    experiment_name = str(
        cfg_get(
            config,
            ["experiment.name"],
            "AdaMCR_KAN_TCR_BEST",
        )
    )

    seed = int(
        cfg_get(
            config,
            ["test.random_seed"],
            42,
        )
    )
    seed_everything(seed)

    # -------------------------
    # 数据路径
    # -------------------------
    test_data_path = (
        args.data
        if args.data is not None
        else cfg_get(
            config,
            ["data.tcr_test", "data.test"],
            "../data/pmhc_tcr_test.csv",
        )
    )

    vocab_path = cfg_get(
        config,
        ["data.vocab", "vocab"],
        "vocab_dict.npy",
    )

    test_data_path = resolve_input_path(
        test_data_path
    )
    vocab_path = resolve_input_path(
        vocab_path
    )

    # -------------------------
    # 手动指定的 Best_model.pkl
    # -------------------------
    trained_tcr_model_path = (
        args.model
        if args.model is not None
        else cfg_get(
            config,
            [
                "model.pkl_path",
                "model.model_path",
                "pkl_path",
            ],
            None,
        )
    )

    if trained_tcr_model_path is None:
        raise ValueError(
            "请在测试 YAML 中填写 model.pkl_path，"
            "或者运行时使用 -m 指定 Best_model.pkl。"
        )

    trained_tcr_model_path = resolve_input_path(
        trained_tcr_model_path
    )

    pmhc_model_path = (
        args.pmhc_model
        if args.pmhc_model is not None
        else cfg_get(
            config,
            ["model.pmhc_model_path"],
            None,
        )
    )

    if pmhc_model_path is None:
        raise ValueError(
            "请在测试 YAML 中填写 "
            "model.pmhc_model_path。"
        )

    pmhc_model_path = resolve_input_path(
        pmhc_model_path
    )

    # -------------------------
    # 数据与模型结构参数
    # -------------------------
    pep_max_len = int(
        cfg_get(config, ["data.pep_max_len"], 25)
    )
    hla_max_len = int(
        cfg_get(config, ["data.hla_max_len"], 34)
    )
    tcr_max_len = int(
        cfg_get(config, ["data.tcr_max_len"], 27)
    )

    d_model = int(
        cfg_get(config, ["model.d_model"], 192)
    )
    d_ff = int(
        cfg_get(config, ["model.d_ff"], 512)
    )
    d_k = int(
        cfg_get(config, ["model.d_k"], 64)
    )
    d_v = int(
        cfg_get(config, ["model.d_v"], 64)
    )
    n_layers = int(
        cfg_get(config, ["model.n_layers"], 2)
    )
    n_heads = int(
        cfg_get(config, ["model.n_heads"], 8)
    )
    dropout = float(
        cfg_get(config, ["model.dropout"], 0.1)
    )
    mkan_order = int(
        cfg_get(config, ["model.mkan_order"], 3)
    )
    cnn_kernel_size = int(
        cfg_get(
            config,
            ["model.cnn_kernel_size"],
            3,
        )
    )
    freeze_pmhc = bool(
        cfg_get(
            config,
            ["model.freeze_pmhc"],
            True,
        )
    )
    strict_pmhc_load = bool(
        cfg_get(
            config,
            ["model.strict_pmhc_load"],
            True,
        )
    )
    strict_load = bool(
        cfg_get(
            config,
            ["test.strict_load"],
            True,
        )
    )

    tgt_len = (
        pep_max_len
        + hla_max_len
        + tcr_max_len
    )

    # -------------------------
    # 损失与测试参数
    # -------------------------
    loss_name = str(
        cfg_get(
            config,
            ["loss.name"],
            "cross_entropy",
        )
    )
    class0_weight = float(
        cfg_get(
            config,
            ["loss.class0_weight"],
            1.0,
        )
    )
    class1_weight = float(
        cfg_get(
            config,
            ["loss.class1_weight"],
            1.0,
        )
    )
    positive_weight = float(
        cfg_get(
            config,
            ["loss.positive_weight"],
            1.0,
        )
    )

    batch_size = int(
        cfg_get(
            config,
            ["test.batch_size"],
            1024,
        )
    )
    threshold = float(
        cfg_get(
            config,
            [
                "test.threshold",
                "eval.threshold",
            ],
            0.5,
        )
    )

    output_dir = resolve_output_path(
        cfg_get(
            config,
            ["test.output_dir"],
            "../result/AdaMCR_KAN_TCR",
        )
    )
    output_name = str(
        cfg_get(
            config,
            ["test.output_name"],
            "test_predictions.csv",
        )
    )

    # -------------------------
    # 文件检查
    # -------------------------
    required_paths = [
        (test_data_path, "测试数据"),
        (vocab_path, "词表"),
        (
            trained_tcr_model_path,
            "手动指定的 AdaMCR_KAN_TCR Best_model.pkl",
        ),
        (
            pmhc_model_path,
            "预训练 TranspMHC Best_model.pkl",
        ),
    ]

    for path, description in required_paths:
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(
                f"{description}不存在：{path}"
            )

    # -------------------------
    # 设备
    # -------------------------
    use_cuda = torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda else "cpu"
    )

    print("=" * 88)
    print("AdaMCR_KAN_TCR manual Best_model test")
    print("=" * 88)
    print("Experiment:", experiment_name)
    print("Config:", os.path.abspath(args.config))
    print("Device:", device)

    if use_cuda:
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Test data:", test_data_path)
    print("Manually specified Best_model.pkl:")
    print(trained_tcr_model_path)
    print("Pretrained TranspMHC:")
    print(pmhc_model_path)
    print("Loss:", loss_name)
    print("Fixed threshold:", threshold)
    print("=" * 88)

    # -------------------------
    # 加载数据
    # -------------------------
    vocab = np.load(
        vocab_path,
        allow_pickle=True,
    ).item()

    test_data = pd.read_csv(
        test_data_path
    )

    required_columns = {
        "tcr",
        "peptide",
        "HLA_sequence",
        "label",
    }

    missing_columns = (
        required_columns
        - set(test_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "测试CSV缺少字段："
            f"{sorted(missing_columns)}"
        )

    (
        _,
        _,
        _,
        test_labels,
        test_loader,
    ) = pMHCI_TCR_test_data_with_loader(
        test_data,
        vocab,
        batch_size=batch_size,
        pep_max_len=pep_max_len,
        hla_max_len=hla_max_len,
        tcr_max_len=tcr_max_len,
        shuffle=False,
    )

    labels_array = np.asarray(
        test_labels.cpu().numpy()
        if isinstance(test_labels, torch.Tensor)
        else test_labels,
        dtype=np.int64,
    ).reshape(-1)

    positive_count = int(
        (labels_array == 1).sum()
    )
    negative_count = int(
        (labels_array == 0).sum()
    )
    positive_prevalence = (
        positive_count / len(labels_array)
        if len(labels_array) > 0
        else float("nan")
    )

    print(
        "Test distribution:",
        {
            "positive": positive_count,
            "negative": negative_count,
            "positive_prevalence": positive_prevalence,
            "random_AUPRC_baseline": positive_prevalence,
        },
    )

    # -------------------------
    # 创建模型
    # -------------------------
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
        freeze_pmhc=freeze_pmhc,
        strict_pmhc_load=strict_pmhc_load,
    ).to(device)

    print(
        "Model parameters:",
        {
            "total": sum(
                parameter.numel()
                for parameter in model.parameters()
            ),
            "trainable": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "d_model": d_model,
            "d_ff": d_ff,
            "d_k": d_k,
            "d_v": d_v,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "dropout": dropout,
            "mkan_order": mkan_order,
            "cnn_kernel_size": cnn_kernel_size,
            "tgt_len": tgt_len,
        },
    )

    # -------------------------
    # 只加载手动指定的模型
    # -------------------------
    load_manual_best_model(
        model=model,
        model_path=trained_tcr_model_path,
        device=device,
        strict_load=strict_load,
    )

    # -------------------------
    # 测试
    # -------------------------
    y_true, y_prob, test_loss = (
        predict_test_set(
            model=model,
            loader=test_loader,
            device=device,
            loss_name=loss_name,
            class0_weight=class0_weight,
            class1_weight=class1_weight,
            positive_weight=positive_weight,
        )
    )

    y_pred, metrics_dict = compute_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        test_loss=test_loss,
    )

    metrics_dict.update(
        {
            "model_path": trained_tcr_model_path,
            "loss_name": loss_name,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_prevalence": positive_prevalence,
            "random_AUPRC_baseline": positive_prevalence,
        }
    )

    print("=" * 88)
    print("Test metrics")

    for key in (
        "test_loss",
        "AUROC",
        "AUPRC",
        "score",
        "ACC",
        "BACC",
        "F1",
        "MCC",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "threshold",
    ):
        print(
            f"{key}: {metrics_dict[key]}"
        )

    print("=" * 88)

    # -------------------------
    # 保存
    # -------------------------
    run_name = (
        f"{experiment_name}_TEST_{now_file_time()}"
    )
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df = test_data.copy().reset_index(
        drop=True
    )
    predictions_df["prob"] = y_prob
    predictions_df["pred_label"] = y_pred
    predictions_df["threshold"] = threshold

    predictions_path = (
        run_dir / output_name
    )
    metrics_csv_path = (
        run_dir / "test_metrics.csv"
    )
    metrics_json_path = (
        run_dir / "test_metrics.json"
    )
    config_copy_path = (
        run_dir / "used_test_config.yaml"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    pd.DataFrame(
        [metrics_dict]
    ).to_csv(
        metrics_csv_path,
        index=False,
    )

    with metrics_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics_dict,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    shutil.copy2(
        args.config,
        config_copy_path,
    )

    print("保存完成：")
    print("Predictions:", predictions_path)
    print("Metrics CSV:", metrics_csv_path)
    print("Metrics JSON:", metrics_json_path)
    print("Used YAML:", config_copy_path)


if __name__ == "__main__":
    main()