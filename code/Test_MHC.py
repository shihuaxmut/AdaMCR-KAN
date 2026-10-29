# -*- coding: utf-8 -*-
import argparse
import importlib
import inspect
import json
import os
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from tqdm import tqdm

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "缺少 PyYAML，请执行：python -m pip install pyyaml"
    ) from exc


LOCAL_TZ = timezone(timedelta(hours=8))
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CODE_DIR, ".."))
DEFAULT_CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "config",
    "pMHC_test_niu_dataset.yaml",
)
# DEFAULT_CONFIG_PATH = os.path.join(
#     PROJECT_ROOT,
#     "config",
#     "pMHC_test_mkan_fair.yaml",
# )
# DEFAULT_CONFIG_PATH = os.path.join(
#     PROJECT_ROOT,
#     "config",
#     "pMHC_test_mkan_fair_second_dataset.yaml",
# )
# DEFAULT_CONFIG_PATH = os.path.join(
#     PROJECT_ROOT,
#     "config",
#     "pMHC_test_mkan_fair_Anthem_dataset.yaml",
# )
def now_file_time():
    return datetime.now(LOCAL_TZ).strftime("%Y-%m-%d_%H-%M-%S")


def seed_everything(seed=1234):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_yaml_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

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


def resolve_input_path(path):
    if path is None:
        return None

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


def resolve_output_path(path):
    if os.path.isabs(path):
        return path

    return os.path.abspath(os.path.join(CODE_DIR, path))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test TranspMHC using YAML."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG_PATH,
        type=str,
    )
    parser.add_argument(
        "-d",
        "--data",
        default=None,
        type=str,
        help="Override YAML test-data path.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        type=str,
        help="Override YAML checkpoint path.",
    )
    return parser.parse_args()


def transfer(y_prob, threshold=0.5):
    y_prob = np.asarray(y_prob, dtype=np.float64)
    return (y_prob > float(threshold)).astype(np.int64)


def build_module(module_class, **kwargs):
    """
    只向 Encoder / Decoder 传入其构造函数支持的参数。
    """
    signature = inspect.signature(module_class.__init__)
    valid_keys = set(signature.parameters.keys())
    valid_keys.discard("self")

    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in valid_keys
    }
    return module_class(**filtered_kwargs)


class TranspMHC(nn.Module):
    """
    与训练时 TranspMHC_MKAN.py 相同的模型包装结构。
    """

    def __init__(
        self,
        encoder_class,
        decoder_class,
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
        mkan_order=2,
        cnn_kernel_size=3,
    ):
        super().__init__()

        if tgt_len != pep_len + hla_len:
            raise ValueError(
                f"tgt_len={tgt_len} 与 pep_len+hla_len="
                f"{pep_len + hla_len} 不一致。"
            )

        encoder_kwargs = {
            "d_model": d_model,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "d_k": d_k,
            "d_v": d_v,
            "n_heads": n_heads,
            "dropout": dropout,
            "use_cuda": use_cuda,
            "mkan_order": mkan_order,
            "cnn_kernel_size": cnn_kernel_size,
        }

        self.pep_encoder = build_module(
            encoder_class,
            **encoder_kwargs,
        )
        self.hla_encoder = build_module(
            encoder_class,
            **encoder_kwargs,
        )

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

        self.decoder = build_module(
            decoder_class,
            **decoder_kwargs,
        )

        # 必须与训练时分类头一致
        self.projection = nn.Sequential(
            nn.Linear(tgt_len * d_model, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, pep_inputs, hla_inputs):
        pep_outputs, pep_attns = self.pep_encoder(pep_inputs)
        hla_outputs, hla_attns = self.hla_encoder(hla_inputs)

        encoder_outputs = torch.cat(
            [pep_outputs, hla_outputs],
            dim=1,
        )

        decoder_outputs, decoder_attns = self.decoder(
            encoder_outputs
        )

        decoder_outputs = decoder_outputs.contiguous().view(
            decoder_outputs.size(0),
            -1,
        )

        logits = self.projection(decoder_outputs)

        return (
            logits.view(-1, logits.size(-1)),
            pep_attns,
            hla_attns,
            decoder_attns,
        )


def import_encoder_decoder(module_name):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"无法导入模块：{module_name}\n"
            f"请确认 {module_name}.py 位于：{CODE_DIR}"
        ) from exc

    if not hasattr(module, "HRIEncoder"):
        raise AttributeError(f"{module_name}.py 中没有 HRIEncoder。")
    if not hasattr(module, "ABGC_Decoder"):
        raise AttributeError(f"{module_name}.py 中没有 ABGC_Decoder。")

    return module.HRIEncoder, module.ABGC_Decoder


def unwrap_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint 不是有效字典。")

    for key in (
        "state_dict",
        "model_state_dict",
        "model",
        "net",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break

    cleaned = {}

    for key, value in checkpoint.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        cleaned[new_key] = value

    return cleaned


def validate_checkpoint_shapes(model, state_dict):
    model_state = model.state_dict()
    mismatches = []

    for key in sorted(
        set(model_state.keys()) & set(state_dict.keys())
    ):
        if model_state[key].shape != state_dict[key].shape:
            mismatches.append(
                (
                    key,
                    tuple(state_dict[key].shape),
                    tuple(model_state[key].shape),
                )
            )

    signature_keys = [
        "decoder.layers.0.pep_to_hla_attn.W_Q.weight",
        "decoder.layers.0.pep_to_hla_attn.W_K.weight",
        "decoder.layers.0.pep_to_hla_attn.W_V.weight",
        "decoder.layers.0.pep_to_hla_attn.fc.weight",
        "decoder.layers.0.pep_to_hla_attn.head_gate.weight",
        "projection.0.weight",
    ]

    print("-" * 80)
    print("Checkpoint architecture signature")

    for key in signature_keys:
        checkpoint_shape = (
            tuple(state_dict[key].shape)
            if key in state_dict
            else "NOT FOUND"
        )
        model_shape = (
            tuple(model_state[key].shape)
            if key in model_state
            else "NOT FOUND"
        )

        print(key)
        print("  checkpoint:", checkpoint_shape)
        print("  model:     ", model_shape)

    print("-" * 80)

    if mismatches:
        lines = ["Checkpoint 与当前模型形状不一致："]

        for key, checkpoint_shape, model_shape in mismatches[:40]:
            lines.append(
                f"{key}: checkpoint={checkpoint_shape}, "
                f"model={model_shape}"
            )

        raise RuntimeError("\n".join(lines))


def load_model_state(
    model,
    model_path,
    device,
    strict_load=True,
):
    checkpoint = torch.load(
        model_path,
        map_location=device,
    )
    state_dict = unwrap_checkpoint(checkpoint)

    validate_checkpoint_shapes(model, state_dict)

    load_result = model.load_state_dict(
        state_dict,
        strict=strict_load,
    )

    print("Model loaded from:", model_path)
    print("Missing keys:", load_result.missing_keys)
    print("Unexpected keys:", load_result.unexpected_keys)

    return model


def create_test_loader(
    test_data,
    vocab,
    batch_size,
    pep_max_len,
    hla_max_len,
):
    if "label" in test_data.columns:
        from DataProcessing import pMHCI_data_with_loader

        pep_inputs, hla_inputs, labels, loader = (
            pMHCI_data_with_loader(
                test_data,
                vocab,
                batch_size=batch_size,
                pep_max_len=pep_max_len,
                hla_max_len=hla_max_len,
                shuffle=False,
            )
        )
        return pep_inputs, hla_inputs, labels, loader

    from Test_DataProcessing import pMHCI_data_with_loader

    pep_inputs, hla_inputs, loader = (
        pMHCI_data_with_loader(
            test_data,
            vocab,
            batch_size=batch_size,
            pep_max_len=pep_max_len,
            hla_max_len=hla_max_len,
            shuffle=False,
        )
    )
    return pep_inputs, hla_inputs, None, loader


def evaluate(model, loader, device):
    model.eval()

    y_true_list = []
    y_prob_list = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Testing pMHC data"):
            if len(batch) == 3:
                pep_inputs, hla_inputs, labels = batch
                y_true_list.extend(
                    labels.cpu().numpy().tolist()
                )
            elif len(batch) == 2:
                pep_inputs, hla_inputs = batch
            else:
                raise ValueError(
                    f"Unexpected batch length: {len(batch)}"
                )

            pep_inputs = pep_inputs.to(device)
            hla_inputs = hla_inputs.to(device)

            logits = model(
                pep_inputs,
                hla_inputs,
            )[0]

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            y_prob_list.extend(
                probabilities.cpu().numpy().tolist()
            )

    return (
        np.asarray(y_true_list, dtype=np.int64),
        np.asarray(y_prob_list, dtype=np.float64),
    )


def safe_divide(numerator, denominator, default=0.0):
    if denominator == 0:
        return float(default)
    return float(numerator) / float(denominator)


def select_train_test_threshold(y_true, y_prob):
    precision_curve, recall_curve, thresholds = (
        precision_recall_curve(y_true, y_prob)
    )

    if len(thresholds) == 0:
        return 0.5

    objective = (
        precision_curve[:-1]
        * recall_curve[:-1]
    )

    return float(
        thresholds[int(np.argmax(objective))]
    )


def compute_metrics(y_true, y_prob, threshold):
    y_pred = transfer(y_prob, threshold)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    accuracy = safe_divide(
        tp + tn,
        tn + fp + fn + tp,
    )
    recall = safe_divide(tp, tp + fn)
    sensitivity = recall
    specificity = safe_divide(tn, tn + fp)
    precision_value = safe_divide(tp, tp + fp)

    f1 = safe_divide(
        2.0 * precision_value * recall,
        precision_value + recall,
    )

    mcc = float(matthews_corrcoef(y_true, y_pred))

    if np.unique(y_true).size == 2:
        auroc = float(roc_auc_score(y_true, y_prob))
    else:
        auroc = float("nan")

    precision_curve, recall_curve, _ = (
        precision_recall_curve(y_true, y_prob)
    )
    auprc = float(auc(recall_curve, precision_curve))

    bacc = (sensitivity + specificity) / 2.0
    score = (auroc + auprc) / 2.0

    metrics = {
        "AUROC": auroc,
        "AUPRC": auprc,
        "score": score,
        "ACC": accuracy,
        "BACC": bacc,
        "F1": f1,
        "MCC": mcc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_value,
        "recall": recall,
        "threshold": float(threshold),
        "score_definition": "score = (AUROC + AUPRC) / 2",
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return y_pred, metrics


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    seed = int(
        cfg_get(config, ["test.random_seed"], 1234)
    )
    seed_everything(seed)

    test_data_path = cfg_get(
        config,
        ["data.pmhc_test"],
        "../data/pmhc_test.csv",
    )
    vocab_path = cfg_get(
        config,
        ["data.vocab"],
        "vocab_dict.npy",
    )
    module_name = str(
        cfg_get(
            config,
            ["model.module_name"],
            "AdaMCR_KAN",
        )
    )
    model_path = cfg_get(
        config,
        ["model.pkl_path"],
        "../model/TranspMHC/Best_model.pkl",
    )

    pep_max_len = int(
        cfg_get(config, ["data.pep_max_len"], 25)
    )
    hla_max_len = int(
        cfg_get(config, ["data.hla_max_len"], 34)
    )

    d_model = int(cfg_get(config, ["model.d_model"], 64))
    d_ff = int(cfg_get(config, ["model.d_ff"], 512))
    d_k = int(cfg_get(config, ["model.d_k"], 64))
    d_v = int(cfg_get(config, ["model.d_v"], 64))
    n_layers = int(cfg_get(config, ["model.n_layers"], 1))
    n_heads = int(cfg_get(config, ["model.n_heads"], 1))
    dropout = float(cfg_get(config, ["model.dropout"], 0.1))
    mkan_order = int(
        cfg_get(config, ["model.mkan_order"], 2)
    )
    cnn_kernel_size = int(
        cfg_get(config, ["model.cnn_kernel_size"], 3)
    )

    batch_size = int(
        cfg_get(config, ["test.batch_size"], 1024)
    )
    threshold = float(
        cfg_get(config, ["test.threshold"], 0.5)
    )
    output_dir = cfg_get(
        config,
        ["test.output_dir"],
        "../result/TranspMHC_test",
    )
    output_name = str(
        cfg_get(
            config,
            ["test.output_name"],
            "pmhc_result.csv",
        )
    )
    compute_metrics_flag = bool(
        cfg_get(config, ["test.compute_metrics"], True)
    )
    metric_mode = str(
        cfg_get(config, ["test.metric_mode"], "fixed")
    ).lower()
    strict_load = bool(
        cfg_get(config, ["test.strict_load"], True)
    )

    if args.data is not None:
        test_data_path = args.data
    if args.model is not None:
        model_path = args.model

    test_data_path = resolve_input_path(test_data_path)
    vocab_path = resolve_input_path(vocab_path)
    model_path = resolve_input_path(model_path)
    output_dir = resolve_output_path(output_dir)

    run_dir = os.path.join(
        output_dir,
        f"TranspMHC_TEST_{now_file_time()}",
    )
    os.makedirs(run_dir, exist_ok=True)

    result_csv_path = os.path.join(
        run_dir,
        output_name,
    )
    metrics_csv_path = os.path.join(
        run_dir,
        "pmhc_metrics.csv",
    )
    metrics_json_path = os.path.join(
        run_dir,
        "pmhc_metrics.json",
    )
    config_copy_path = os.path.join(
        run_dir,
        "used_config.yaml",
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_cuda = device.type == "cuda"
    tgt_len = pep_max_len + hla_max_len

    print("=" * 80)
    print("TranspMHC Test")
    print("=" * 80)
    print("Config path:", args.config)
    print("Test data path:", test_data_path)
    print("Model module:", module_name)
    print("Model path:", model_path)
    print("Vocab path:", vocab_path)
    print("Output directory:", run_dir)
    print("Device:", device)

    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))

    print("-" * 80)
    print("d_model:", d_model)
    print("d_ff:", d_ff)
    print("d_k:", d_k)
    print("d_v:", d_v)
    print("n_layers:", n_layers)
    print("n_heads:", n_heads)
    print("dropout:", dropout)
    print("mkan_order:", mkan_order)
    print("cnn_kernel_size:", cnn_kernel_size)
    print("pep_max_len:", pep_max_len)
    print("hla_max_len:", hla_max_len)
    print("tgt_len:", tgt_len)
    print("batch_size:", batch_size)
    print("metric_mode:", metric_mode)
    print("=" * 80)

    for path, description in (
        (test_data_path, "Test data"),
        (vocab_path, "Vocab"),
        (model_path, "Checkpoint"),
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{description} not found: {path}"
            )

    vocab = np.load(
        vocab_path,
        allow_pickle=True,
    ).item()

    test_data = pd.read_csv(test_data_path)

    _, _, _, test_loader = create_test_loader(
        test_data=test_data,
        vocab=vocab,
        batch_size=batch_size,
        pep_max_len=pep_max_len,
        hla_max_len=hla_max_len,
    )

    Encoder, Decoder = import_encoder_decoder(module_name)

    model = TranspMHC(
        encoder_class=Encoder,
        decoder_class=Decoder,
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
        mkan_order=mkan_order,
        cnn_kernel_size=cnn_kernel_size,
    ).to(device)

    print(
        "Trainable parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )

    model = load_model_state(
        model=model,
        model_path=model_path,
        device=device,
        strict_load=strict_load,
    )

    y_true, y_prob = evaluate(
        model=model,
        loader=test_loader,
        device=device,
    )

    if len(y_true) == 0 and "label" in test_data.columns:
        y_true = test_data["label"].to_numpy(
            dtype=np.int64
        )

    if len(y_prob) != len(test_data):
        raise RuntimeError(
            f"预测数量 {len(y_prob)} 与测试样本数量 "
            f"{len(test_data)} 不一致。"
        )

    raw_pred = transfer(y_prob, threshold)

    result_df = pd.DataFrame()

    for column in (
        "peptide",
        "hla",
        "HLA_sequence",
        "label",
    ):
        if column in test_data.columns:
            result_df[column] = test_data[column]

    result_df["prob"] = y_prob
    result_df["pred_label"] = raw_pred
    result_df["threshold"] = threshold

    metrics_dict = None

    if compute_metrics_flag and len(y_true) > 0:
        if metric_mode == "train_test":
            metric_threshold = select_train_test_threshold(
                y_true,
                y_prob,
            )
        elif metric_mode == "fixed":
            metric_threshold = threshold
        else:
            raise ValueError(
                "test.metric_mode 只能是 fixed 或 train_test。"
            )

        metric_pred, metrics_dict = compute_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=metric_threshold,
        )

        result_df["pred_label_metric"] = metric_pred
        result_df["metric_threshold"] = metric_threshold

        pd.DataFrame([metrics_dict]).to_csv(
            metrics_csv_path,
            index=False,
        )

        with open(
            metrics_json_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metrics_dict,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False,
            )

        print("=" * 80)
        print("Test metrics")
        for key in (
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
            print(f"{key}: {metrics_dict[key]}")
        print("=" * 80)

    result_df.to_csv(result_csv_path, index=False)

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as source_file:
        config_text = source_file.read()

    with open(
        config_copy_path,
        "w",
        encoding="utf-8",
    ) as target_file:
        target_file.write(config_text)

    print("=" * 80)
    print("Prediction finished.")
    print("Prediction CSV:", result_csv_path)

    if metrics_dict is not None:
        print("Metrics CSV:", metrics_csv_path)
        print("Metrics JSON:", metrics_json_path)

    print("Used config:", config_copy_path)
    print("=" * 80)


if __name__ == "__main__":
    main()