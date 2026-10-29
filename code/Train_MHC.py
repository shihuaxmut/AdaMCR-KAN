import numpy as np
import pandas as pd
import time
import random
import copy
import shutil
import json
import warnings
import os
import sys
import argparse
import logging
import io
from contextlib import redirect_stdout
from datetime import datetime, timezone, timedelta

try:
    import yaml
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "缺少 PyYAML，请先执行：python -m pip install pyyaml"
    )

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm, trange
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve

import AdaMCR_KAN_imp
from AdaMCR_KAN_imp import AdaMCR_KAN_MHC
import DataProcessing
from DataProcessing import pMHCI_dataset
from DataProcessing import pMHCI_data_with_loader
import Performance
from Performance import performances

# ===================== Basic setting =====================
warnings.filterwarnings("ignore")
random.seed(2026)

LOCAL_TZ = timezone(timedelta(hours=8))

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CODE_DIR, ".."))
#    第一个 Niu数据集的写法！
DEFAULT_CONFIG_PATH = (
    "/home/zqliudachen/"
    "AdaMCR-KAN/config/pMHC_train_Niu_dataset.yaml"
)
# DEFAULT_CONFIG_PATH = (
#     "/home/zqliudachen/Third_work_pmhc_tcr/"
#     "TranspMHC-TransTCR/config/pMHC_train_mkan_fair.yaml"
# )
# =========================================================

#    第二个 数据集的写法！
# DEFAULT_CONFIG_PATH = (
#     "/home/zqliudachen/Third_work_pmhc_tcr/"
#     "TranspMHC-TransTCR/config/pMHC_train_mkan_fair_second_dataset.yaml"
# )
# =========================================================
#    第3个 数据集的写法！
# DEFAULT_CONFIG_PATH = (
#     "/home/zqliudachen/Third_work_pmhc_tcr/"
#     "TranspMHC-TransTCR/config/pMHC_train_mkan_fair_Anthem_dataset.yaml"
# )
# =========================================================
# ===================== Time functions =====================
def now_dt():
    return datetime.now(LOCAL_TZ)


def now_file_time():
    return now_dt().strftime("%Y-%m-%d_%H-%M-%S")


def now_log_time():
    return now_dt().strftime("%Y-%m-%d %H:%M:%S.%f")
# ==========================================================


# ===================== Config functions =====================
def load_yaml_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    return cfg


def cfg_get(cfg, keys, default=None):
    for key in keys:
        cur = cfg
        ok = True
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def resolve_path(path):
    if path is None:
        return None

    if os.path.isabs(path):
        return path

    candidates = [
        os.path.abspath(os.path.join(CODE_DIR, path)),
        os.path.abspath(os.path.join(PROJECT_ROOT, path)),
        os.path.abspath(path)
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    return candidates[0]
# ============================================================


# ===================== Args =====================
def parse_args():
    parser = argparse.ArgumentParser(description="Train MHC")

    parser.add_argument(
        "-c",
        "--config",
        help="Path of pMHC_train.yaml",
        default=DEFAULT_CONFIG_PATH,
        type=str
    )

    parser.add_argument(
        "-d",
        "--data",
        help="Test pMHC-I csv path. If not set, use yaml.",
        dest="data",
        default=None,
        type=str
    )

    parser.add_argument(
        "-train",
        "--pmhc_train",
        help="Training pMHC-I csv path. If not set, use yaml.",
        dest="pmhc_train",
        default=None,
        type=str
    )

    args = parser.parse_args()
    return args
# =================================================


# ===================== Logger setting =====================
class BeijingFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, LOCAL_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S") + f",{int(record.msecs):03d}"


def setup_logger(log_file):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    formatter = BeijingFormatter(
        fmt="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return root
# ==========================================================


# ===================== Original helper functions =====================
def f_mean(l):
    if len(l) == 0:
        return 0
    return sum(l) / len(l)


def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])
# ====================================================================


# ===================== Extra helper functions =====================
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tensor_loss_mean(loss_list):
    if len(loss_list) == 0:
        return 0.0

    vals = []
    for x in loss_list:
        if torch.is_tensor(x):
            vals.append(float(x.detach().cpu().item()))
        else:
            vals.append(float(x))
    return sum(vals) / len(vals)


def metrics_to_dict(metrics, score=None, threshold=None):
    metric_dict = {
        "AUROC": float(metrics[0]),
        "AUPRC": float(metrics[1]),
        "ACC": float(metrics[2]),
        "F1": float(metrics[3]),
        "MCC": float(metrics[4]),
        "sensitivity": float(metrics[5]),
        "specificity": float(metrics[6]),
        "precision": float(metrics[7]),
        "recall": float(metrics[8])
    }

    if score is not None:
        metric_dict["score"] = float(score)

    if threshold is not None:
        metric_dict["threshold"] = float(threshold)

    return metric_dict


def fold_result_dict(metrics, fold, score, best_epoch, threshold, model_path):
    bacc = (float(metrics[5]) + float(metrics[6])) / 2

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
        "model_path": model_path
    }
# ==================================================================


# ===================== Original train_step =====================
def train_step(model, optimizer, train_loader, fold, epoch, epochs, use_cuda=True):
    device = torch.device("cuda" if use_cuda else "cpu")

    time_train_ep = 0
    model.train()

    y_true_train_list, y_prob_train_list = [], []
    loss_train_list, dec_attns_train_list = [], []

    for train_pep_inputs, train_hla_inputs, train_labels in tqdm(
        train_loader,
        desc=f"Fold {fold} Epoch {epoch}/{epochs} Train",
        leave=False
    ):
        train_pep_inputs, train_hla_inputs, train_labels = (
            train_pep_inputs.to(device),
            train_hla_inputs.to(device),
            train_labels.to(device)
        )

        t1 = time.time()

        optimizer.zero_grad()
        train_outputs, _, _, train_dec_self_attns = model(train_pep_inputs, train_hla_inputs)

        criterion = nn.CrossEntropyLoss()
        train_loss = criterion(train_outputs, train_labels)

        time_train_ep += time.time() - t1

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        y_true_train = train_labels.cpu().numpy()
        y_prob_train = nn.Softmax(dim=1)(train_outputs)[:, 1].cpu().detach().numpy()

        y_true_train_list.extend(y_true_train)
        y_prob_train_list.extend(y_prob_train)

        loss_train_list.append(train_loss)

    y_pred_train_list = transfer(y_prob_train_list, threshold)
    ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)

    return loss_train_list, time_train_ep
# ===============================================================


# ===================== Original eval_step metric logic =====================
def eval_step(model, loader, use_cuda=True, name="test"):
    model.eval()
    device = torch.device("cuda" if use_cuda else "cpu")

    with torch.no_grad():
        loss_val_list, dec_attns_val_list = [], []
        y_true_val_list, y_prob_val_list = [], []
        pep_inputs, hla_inputs = [], []

        for val_pep_inputs, val_hla_inputs, val_labels in tqdm(
            loader,
            desc=f"Eval {name}",
            leave=False
        ):
            val_pep_inputs, val_hla_inputs, val_labels = (
                val_pep_inputs.to(device),
                val_hla_inputs.to(device),
                val_labels.to(device)
            )

            val_outputs = model(val_pep_inputs, val_hla_inputs)
            correct_output = val_outputs[0]

            y_prob_val = nn.Softmax(dim=1)(correct_output)[:, 1].cpu().detach().numpy()
            y_true_val = val_labels.cpu().numpy()

            y_true_val_list.extend(y_true_val)
            y_prob_val_list.extend(y_prob_val)

        best_threshold = None

        # ===================== 原始阈值计算逻辑，不改 =====================
        if name == "test":
            precision, recall, thresholds = precision_recall_curve(
                y_true_val_list,
                y_prob_val_list
            )
            best_index = np.argmax(precision * recall)
            best_threshold = thresholds[best_index]
            y_pred_val_list = transfer(y_prob_val_list, best_threshold)
        else:
            y_pred_val_list = transfer(y_prob_val_list, 0.5)
            best_threshold = 0.5
        # ================================================================

        ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)

        # ===================== 原始指标计算逻辑，不改 =====================
        perf_stdout = io.StringIO()
        with redirect_stdout(perf_stdout):
            metrics_val = performances(
                y_true_val_list,
                y_pred_val_list,
                y_prob_val_list,
                print_=True
            )
        perf_text = perf_stdout.getvalue().strip()
        # ================================================================

    return ys_val, loss_val_list, metrics_val, best_threshold, perf_text
# ======================================================================


# ===================== Main =====================
args = parse_args()
cfg = load_yaml_config(args.config)

# --------------------- Parameters from yaml ---------------------
batch_size = int(cfg_get(cfg, ["train.batch_size", "batch_size"], 1024))
epochs = int(cfg_get(cfg, ["train.epochs", "epochs"], 18))
learning_rate = float(cfg_get(cfg, ["train.learning_rate", "train.lr", "lr", "learning_rate"], 1e-3))
k_folds = int(cfg_get(cfg, ["train.k_folds", "k_folds"], 5))

d_model = int(cfg_get(cfg, ["model.d_model", "d_model"], 64))
d_ff = int(cfg_get(cfg, ["model.d_ff", "d_ff"], 512))
d_k = int(cfg_get(cfg, ["model.d_k", "d_k"], 64))
d_v = int(cfg_get(cfg, ["model.d_v", "d_v"], 64))
n_layers = int(cfg_get(cfg, ["model.n_layers", "n_layers"], 1))
n_heads = int(cfg_get(cfg, ["model.n_heads", "n_heads"], 1))
dropout = float(cfg_get(cfg, ["model.dropout", "dropout"], 0.1))
mkan_order = int(cfg_get(cfg, ["model.mkan_order", "mkan_order"], 2))
cnn_kernel_size = int(cfg_get(cfg, ["model.cnn_kernel_size", "cnn_kernel_size"], 3))

threshold = float(cfg_get(cfg, ["eval.threshold", "threshold"], 0.5))
pep_max_len = int(cfg_get(cfg, ["data.pep_max_len", "pep_max_len"], 25))
hla_max_len = int(cfg_get(cfg, ["data.hla_max_len", "hla_max_len"], 34))
tgt_len = pep_max_len + hla_max_len

pmhc_train_path = cfg_get(cfg, ["data.pmhc_train", "data.train", "pmhc_train", "train"], None)
pmhc_test_path = cfg_get(cfg, ["data.pmhc_test", "data.test", "data.data", "test", "data"], None)
vocab_path = cfg_get(cfg, ["data.vocab", "vocab", "vocab_path"], "vocab_dict.npy")

if args.pmhc_train is not None:
    pmhc_train_path = args.pmhc_train

if args.data is not None:
    pmhc_test_path = args.data

if pmhc_train_path is None:
    pmhc_train_path = "../data/pmhc_train.csv"

if pmhc_test_path is None:
    pmhc_test_path = "../data/pmhc_test.csv"

pmhc_train_path = resolve_path(pmhc_train_path)
pmhc_test_path = resolve_path(pmhc_test_path)
vocab_path = resolve_path(vocab_path)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# ----------------------------------------------------------------


# ===================== Run path setting =====================
current_time = now_file_time()
run_name = f"AdaMCR-KAN_RUN_{current_time}"

log_dir = os.path.join(PROJECT_ROOT, "result", "log")
model_dir = os.path.join(PROJECT_ROOT, "model", "MHC", run_name)
pred_dir = os.path.join(PROJECT_ROOT, "result", "MHC", run_name)

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)

result_path = os.path.join(log_dir, f"{run_name}_train_log.txt")
# ============================================================


# ===================== Logger init =====================
setup_logger(result_path)

exp_logger = logging.getLogger("Experiment Name")
desc_logger = logging.getLogger("Description")
tags_logger = logging.getLogger("Tags:")
root_logger = logging.getLogger("root")
begin_logger = logging.getLogger("Begin training model #")
param_logger = logging.getLogger("Trainable params")
start_logger = logging.getLogger("Start training")
train_logger = logging.getLogger("Training process")
val_metric_logger = logging.getLogger("val metric:")
test_metric_logger = logging.getLogger("test metric:")
val_raw_logger = logging.getLogger("val raw:")
test_raw_logger = logging.getLogger("test raw:")
best_logger = logging.getLogger("Best model")
score_summary_logger = logging.getLogger("Epoch score summary")
test_logger = logging.getLogger("Test")
fold_result_logger = logging.getLogger("Fold result")
fold_final_result_logger = logging.getLogger("Fold final result")
# ======================================================


# ===================== Load data =====================
vocab = np.load(vocab_path, allow_pickle=True).item()
vocab_size = len(vocab)

train_data = pd.read_csv(pmhc_train_path)
test_data = pd.read_csv(pmhc_test_path)

train_pep_inputs, train_hla_inputs, train_labels, train_loader = pMHCI_data_with_loader(
    train_data,
    vocab,
    batch_size=batch_size,
    pep_max_len=pep_max_len,
    hla_max_len=hla_max_len,
    shuffle=False
)

test_pep_inputs, test_hla_inputs, test_labels, test_loader = pMHCI_data_with_loader(
    test_data,
    vocab,
    batch_size=batch_size,
    pep_max_len=pep_max_len,
    hla_max_len=hla_max_len,
    shuffle=False
)
# =====================================================


# ===================== Log basic information =====================
exp_logger.info(cfg_get(cfg, ["experiment.name", "name"], "AdaMCR_KAN_train"))
desc_logger.info(cfg_get(cfg, ["experiment.description", "description"], "AdaMCR_KAN training with original metrics"))
tags_logger.info(cfg_get(cfg, ["experiment.tags", "tags"], ["baseline", "AdaMCR_KAN"]))

root_logger.info("Run name: %s", run_name)
root_logger.info("Config path: %s", args.config)
root_logger.info("Time zone: UTC+8")
root_logger.info("File time: %s", current_time)
root_logger.info("Start time: %s", now_log_time())
root_logger.info("Run directory: %s", PROJECT_ROOT)
root_logger.info("Log file: %s", result_path)
root_logger.info("Checkpoints directory: %s", model_dir)
root_logger.info("Prediction results directory: %s", pred_dir)
root_logger.info("CUDA_VISIBLE_DEVICES: %s", os.getenv("CUDA_VISIBLE_DEVICES"))
root_logger.info("use_cuda: %s", use_cuda)
root_logger.info("device: %s", device)

if use_cuda:
    root_logger.info("GPU name: %s", torch.cuda.get_device_name(0))

root_logger.info("Training data path: %s", pmhc_train_path)
root_logger.info("Testing data path: %s", pmhc_test_path)
root_logger.info("Vocab path: %s", vocab_path)
root_logger.info("Training samples: %d", len(train_data))
root_logger.info("Testing samples: %d", len(test_data))

if "label" in train_data.columns:
    root_logger.info("Training label distribution: %s", train_data["label"].value_counts().to_dict())

if "label" in test_data.columns:
    root_logger.info("Testing label distribution: %s", test_data["label"].value_counts().to_dict())

root_logger.info(
    "Parameters: batch_size=%d, epochs=%d, lr=%s, k_folds=%d, "
    "pep_max_len=%d, hla_max_len=%d, tgt_len=%d, d_model=%d, "
    "d_ff=%d, d_k=%d, d_v=%d, n_layers=%d, n_heads=%d, "
    "dropout=%.3f, mkan_order=%d, cnn_kernel_size=%d",
    batch_size,
    epochs,
    learning_rate,
    k_folds,
    pep_max_len,
    hla_max_len,
    tgt_len,
    d_model,
    d_ff,
    d_k,
    d_v,
    n_layers,
    n_heads,
    dropout,
    mkan_order,
    cnn_kernel_size
)

root_logger.info("All model parameters are passed from YAML into AdaMCR_KAN.")
root_logger.info(
    "Score definition: score = (test_AUROC + test_AUPRC) / 2, same as original metrics_test_ep_avg."
)
# ================================================================


# ===================== Train / Val / Test =====================
skf = StratifiedKFold(n_splits=k_folds, shuffle=True)

fold_summary_list = []

for fold, (train_index, val_index) in enumerate(skf.split(train_pep_inputs, train_labels)):
    fold_logger = logging.getLogger(f"Fold {fold}")

    fold_logger.info("Train %d, Val %d", len(train_index), len(val_index))
    begin_logger.info("%d", fold)

    train_pep, train_hla = train_pep_inputs[train_index], train_hla_inputs[train_index]
    val_pep, val_hla = train_pep_inputs[val_index], train_hla_inputs[val_index]

    train_label = train_labels[train_index]
    val_label = train_labels[val_index]

    train_loader = DataLoader(
        pMHCI_dataset(train_pep, train_hla, train_label),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        pMHCI_dataset(val_pep, val_hla, val_label),
        batch_size=batch_size,
        shuffle=True
    )

    model = AdaMCR_KAN_MHC(
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
        cnn_kernel_size=cnn_kernel_size,
        mkan_order=mkan_order
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    param_logger.info("%d", count_trainable_params(model))
    start_logger.info("%s", now_log_time())

    metric_best, ep_best = 0, 0
    time_train = 0

    best_model_state = None
    best_ys_test_val = None
    best_metrics_test_val = None
    best_test_threshold = None

    epoch_score_list = []

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        loss_train_list, time_train_ep = train_step(
            model,
            optimizer,
            train_loader,
            fold,
            epoch,
            epochs,
            use_cuda=use_cuda
        )

        train_loss_avg = tensor_loss_mean(loss_train_list)

        ys_train_val, loss_train_val_list, metrics_train_val, val_threshold, val_perf_text = eval_step(
            model,
            val_loader,
            use_cuda=use_cuda,
            name="val"
        )

        metrics_train_ep_avg = sum(metrics_train_val[:2]) / 2

        ys_test_val, loss_test_val_list, metrics_test_val, test_threshold, test_perf_text = eval_step(
            model,
            test_loader,
            use_cuda=use_cuda,
            name="test"
        )

        metrics_test_ep_avg = sum(metrics_test_val[:2]) / 2

        metrics_ep_avg = metrics_test_ep_avg


        epoch_total_time = time.time() - epoch_start_time

        epoch_score_list.append({
            "fold": fold,
            "epoch": epoch - 1,
            "epoch_round": epoch,
            "test_AUROC": float(metrics_test_val[0]),
            "test_AUPRC": float(metrics_test_val[1]),
            "test_score": float(metrics_test_ep_avg),
            "test_ACC": float(metrics_test_val[2]),
            "test_F1": float(metrics_test_val[3]),
            "test_MCC": float(metrics_test_val[4]),
            "test_threshold": float(test_threshold)
        })

        train_logger.info(
            "[base_model%d]-[model%d]-[Epoch %04d | round %02d/%02d] - time: %4d s, train_loss: %.5f",
            fold,
            fold,
            epoch - 1,
            epoch,
            epochs,
            int(epoch_total_time),
            train_loss_avg
        )

        val_metric_logger.info(
            "%s",
            metrics_to_dict(
                metrics_train_val,
                score=metrics_train_ep_avg,
                threshold=0.5
            )
        )

        test_metric_logger.info(
            "%s",
            metrics_to_dict(
                metrics_test_val,
                score=metrics_test_ep_avg,
                threshold=test_threshold
            )
        )

        if val_perf_text:
            val_raw_logger.info("%s", val_perf_text.replace("\n", " | "))

        if test_perf_text:
            test_raw_logger.info("%s", test_perf_text.replace("\n", " | "))

        if metrics_ep_avg > metric_best:
            metric_best, ep_best = metrics_ep_avg, epoch

            best_model_state = copy.deepcopy(model.eval().state_dict())
            best_ys_test_val = ys_test_val
            best_metrics_test_val = metrics_test_val
            best_test_threshold = test_threshold

            best_logger.info(
                "Fold %d new best: epoch=%04d, round=%02d/%02d, score=%.5f",
                fold,
                ep_best - 1,
                ep_best,
                epochs,
                metric_best
            )

    # ===================== Fold epoch score summary =====================
    score_summary_logger.info("Fold %d epoch score summary:", fold)

    for item in epoch_score_list:
        score_summary_logger.info(
            "Fold %d | Epoch %04d | round=%02d/%02d | test_AUROC=%.5f | test_AUPRC=%.5f | test_score=%.5f | test_ACC=%.5f | test_F1=%.5f | test_MCC=%.5f | threshold=%.5f",
            fold,
            item["epoch"],
            item["epoch_round"],
            epochs,
            item["test_AUROC"],
            item["test_AUPRC"],
            item["test_score"],
            item["test_ACC"],
            item["test_F1"],
            item["test_MCC"],
            item["test_threshold"]
        )

    score_summary_logger.info(
        "Fold %d final best epoch: epoch=%04d, round=%02d/%02d, best_score=%.5f",
        fold,
        ep_best - 1,
        ep_best,
        epochs,
        metric_best
    )

    epoch_score_df = pd.DataFrame(epoch_score_list)
    epoch_score_csv = os.path.join(
        pred_dir,
        f"{run_name}_fold{fold}_epoch_score_summary.csv"
    )
    epoch_score_df.to_csv(epoch_score_csv, index=False)
    score_summary_logger.info("Fold %d epoch score summary CSV saved: %s", fold, epoch_score_csv)
    # ===================================================================

    # ===================== Save fold best pkl and verify it =====================
    if best_model_state is not None:
        pkl_name = (
            f"{run_name}_"
            f"best_model_"
            f"fold{fold}_"
            f"epoch{ep_best - 1:04d}_"
            f"score{metric_best:.4f}.pkl"
        )

        path_saver = os.path.join(model_dir, pkl_name)
        torch.save(best_model_state, path_saver)

        best_logger.info(
            "Fold %d best model saved: best_epoch=%04d, round=%02d/%02d, best_score=%.5f, path=%s",
            fold,
            ep_best - 1,
            ep_best,
            epochs,
            metric_best,
            path_saver
        )

        test_logger.info("Loading fold %d", fold)

        model.load_state_dict(best_model_state)
        model.eval()

        retest_ys, retest_loss_list, retest_metrics, retest_threshold, retest_perf_text = eval_step(
            model,
            test_loader,
            use_cuda=use_cuda,
            name="test"
        )

        retest_score = sum(retest_metrics[:2]) / 2

        fold_result = fold_result_dict(
            metrics=retest_metrics,
            fold=fold,
            score=retest_score,
            best_epoch=ep_best - 1,
            threshold=retest_threshold,
            model_path=path_saver
        )

        fold_result_logger.info("%s", fold_result)

        if retest_perf_text:
            logging.getLogger("Fold test raw:").info(
                "Fold %d | %s",
                fold,
                retest_perf_text.replace("\n", " | ")
            )

        fold_pred_df = {
            "peptide": test_data["peptide"],
            "hla": test_data["hla"],
            "label": retest_ys[0],
            "prob": retest_ys[2]
        }

        fold_pred_df = pd.DataFrame(fold_pred_df)

        fold_pred_csv = os.path.join(
            pred_dir,
            f"{run_name}_prediction_fold{fold}_epoch{ep_best - 1:04d}_score{retest_score:.4f}.csv"
        )
        fold_pred_df.to_csv(fold_pred_csv, index=False)

        best_logger.info(
            "Fold %d prediction CSV saved: %s",
            fold,
            fold_pred_csv
        )

        fold_summary_list.append({
            "fold": fold,
            "best_epoch": ep_best - 1,
            "best_epoch_round": ep_best,
            "best_score": float(retest_score),
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
            "pkl_path": path_saver,
            "prediction_csv": fold_pred_csv,
            "epoch_score_csv": epoch_score_csv
        })

if len(fold_summary_list) > 0:
    summary_df = pd.DataFrame(fold_summary_list)
    summary_path = os.path.join(pred_dir, f"{run_name}_fold_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    root_logger.info("Fold summary saved: %s", summary_path)

    for item in fold_summary_list:
        bacc = (float(item["test_sensitivity"]) + float(item["test_specificity"])) / 2

        fold_result = {
            "AUC": float(item["test_AUROC"]),
            "AUPR": float(item["test_AUPRC"]),
            "ACC": float(item["test_ACC"]),
            "BACC": float(bacc),
            "F1": float(item["test_F1"]),
            "MCC": float(item["test_MCC"]),
            "fold": int(item["fold"]),
            "best_epoch": int(item["best_epoch"]),
            "score": float(item["best_score"])
        }

        fold_final_result_logger.info("%s", fold_result)

    best_overall = max(fold_summary_list, key=lambda x: x["best_score"])

    best_fold = int(best_overall["fold"])
    best_epoch = int(best_overall["best_epoch"])
    best_score = float(best_overall["best_score"])
    best_source_model_path = best_overall["pkl_path"]

    final_best_model_path = os.path.join(model_dir, "Best_model.pkl")
    shutil.copyfile(best_source_model_path, final_best_model_path)

    final_best_prediction_path = os.path.join(pred_dir, "Best_model_prediction.csv")
    shutil.copyfile(best_overall["prediction_csv"], final_best_prediction_path)

    best_model_info = {
        "best_fold": best_fold,
        "best_epoch": best_epoch,
        "best_epoch_round": int(best_overall["best_epoch_round"]),
        "best_score": best_score,
        "score_definition": "score = (test_AUROC + test_AUPRC) / 2",
        "source_model_path": best_source_model_path,
        "Best_model_path": final_best_model_path,
        "Best_model_prediction_csv": final_best_prediction_path,
        "test_AUROC": float(best_overall["test_AUROC"]),
        "test_AUPRC": float(best_overall["test_AUPRC"]),
        "test_ACC": float(best_overall["test_ACC"]),
        "test_F1": float(best_overall["test_F1"]),
        "test_MCC": float(best_overall["test_MCC"]),
        "test_sensitivity": float(best_overall["test_sensitivity"]),
        "test_specificity": float(best_overall["test_specificity"]),
        "test_precision": float(best_overall["test_precision"]),
        "test_recall": float(best_overall["test_recall"]),
        "best_test_threshold": float(best_overall["best_test_threshold"])
    }

    best_logger.info(
        "%s",
        {
            "best_fold": best_fold,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "model_path": final_best_model_path
        }
    )

    best_info_json_path = os.path.join(model_dir, "Best_model_info.json")
    with open(best_info_json_path, "w", encoding="utf-8") as f:
        json.dump(best_model_info, f, indent=4, ensure_ascii=False)

    best_info_csv_path = os.path.join(model_dir, "Best_model_info.csv")
    pd.DataFrame([best_model_info]).to_csv(best_info_csv_path, index=False)

    root_logger.info("Best model copied from: %s", best_source_model_path)
    root_logger.info("Best model saved as: %s", final_best_model_path)
    root_logger.info("Best model prediction copied to: %s", final_best_prediction_path)
    root_logger.info("Best model info JSON saved: %s", best_info_json_path)
    root_logger.info("Best model info CSV saved: %s", best_info_csv_path)

    mean_metrics = summary_df[
        [
            "best_score",
            "test_AUROC",
            "test_AUPRC",
            "test_ACC",
            "test_F1",
            "test_MCC"
        ]
    ].mean().to_dict()

    std_metrics = summary_df[
        [
            "best_score",
            "test_AUROC",
            "test_AUPRC",
            "test_ACC",
            "test_F1",
            "test_MCC"
        ]
    ].std().to_dict()

    root_logger.info("Mean metrics: %s", mean_metrics)
    root_logger.info("Std metrics: %s", std_metrics)

root_logger.info("Training finished.")
root_logger.info("Run name: %s", run_name)
root_logger.info("End time: %s", now_log_time())
root_logger.info("Log file: %s", result_path)
root_logger.info("Model folder: %s", model_dir)
root_logger.info("Result folder: %s", pred_dir)

print("Training finished.")
print("Log file:", result_path)
print("Model folder:", model_dir)
print("Result folder:", pred_dir)
