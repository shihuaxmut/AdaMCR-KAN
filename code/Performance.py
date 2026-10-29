import numpy as np
import pandas as pd

from collections import Counter

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import matthews_corrcoef


def safe_divide(numerator, denominator, default=0.0):

    if denominator == 0:
        return float(default)

    return float(numerator) / float(denominator)


def performances(y_true, y_pred, y_prob, print_=True):


    # 转换成 NumPy 数组，保证输入格式统一
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)

    # 固定 labels=[0, 1]，保证始终得到 2×2 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel().tolist()

    # Accuracy
    accuracy = safe_divide(
        tp + tn,
        tn + fp + fn + tp
    )

    # Recall / Sensitivity
    recall = safe_divide(
        tp,
        tp + fn
    )

    sensitivity = recall

    # Specificity
    specificity = safe_divide(
        tn,
        tn + fp
    )

    # Precision
    precision = safe_divide(
        tp,
        tp + fp
    )

    # F1
    f1 = safe_divide(
        2.0 * precision * recall,
        precision + recall
    )

    # MCC
    # 使用 sklearn 标准实现，避免旧代码中的 np.float 问题
    mcc = float(
        matthews_corrcoef(
            y_true,
            y_pred
        )
    )

    # AUROC
    # 只有真实标签同时包含 0 和 1 时才能正常计算
    if np.unique(y_true).size == 2:
        roc_auc = float(
            roc_auc_score(
                y_true,
                y_prob
            )
        )
    else:
        roc_auc = 0.0

    # AUPRC
    prec_curve, recall_curve, _ = precision_recall_curve(
        y_true,
        y_prob
    )

    roc_prc = float(
        auc(
            recall_curve,
            prec_curve
        )
    )

    if print_:
        print(
            "tn = {}, fp = {}, fn = {}, tp = {}".format(
                tn,
                fp,
                fn,
                tp
            )
        )

        print(
            "y_pred: 0 = {} | 1 = {}".format(
                Counter(y_pred)[0],
                Counter(y_pred)[1]
            )
        )

        print(
            "y_true: 0 = {} | 1 = {}".format(
                Counter(y_true)[0],
                Counter(y_true)[1]
            )
        )

        print(
            (
                "auc={:.4f}|auprc={:.4f}|"
                "sensitivity={:.4f}|specificity={:.4f}|"
                "acc={:.4f}|mcc={:.4f}"
            ).format(
                roc_auc,
                roc_prc,
                sensitivity,
                specificity,
                accuracy,
                mcc
            )
        )

        print(
            "precision={:.4f}|recall={:.4f}|f1={:.4f}".format(
                precision,
                recall,
                f1
            )
        )

    return (
        roc_auc,
        roc_prc,
        accuracy,
        f1,
        mcc,
        sensitivity,
        specificity,
        precision,
        recall
    )


def performances_to_pd(performances_list):
    """
    将多次实验结果转换为 DataFrame。

    performances() 返回 9 个值，因此这里必须有 9 个列名。
    """

    metrics_name = [
        "roc_auc",
        "auprc",
        "accuracy",
        "f1",
        "mcc",
        "sensitivity",
        "specificity",
        "precision",
        "recall"
    ]

    performances_pd = pd.DataFrame(
        performances_list,
        columns=metrics_name
    )

    performances_pd.loc["mean"] = performances_pd.mean(
        axis=0,
        numeric_only=True
    )

    performances_pd.loc["std"] = performances_pd.std(
        axis=0,
        numeric_only=True
    )

    return performances_pd