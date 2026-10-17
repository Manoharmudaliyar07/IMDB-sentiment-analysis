"""
Model Evaluation and Metrics Module.
Calculates leak-free classification metrics, confusion matrices, and ROC-AUC.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute full classification evaluation metrics.
    """
    if class_names is None:
        class_names = ["Negative", "Positive"]

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class metrics
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred).tolist()

    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            roc_auc = None

    per_class_dict = {}
    for idx, name in enumerate(class_names):
        per_class_dict[name] = {
            "precision": float(prec_per_class[idx]) if idx < len(prec_per_class) else 0.0,
            "recall": float(rec_per_class[idx]) if idx < len(rec_per_class) else 0.0,
            "f1_score": float(f1_per_class[idx]) if idx < len(f1_per_class) else 0.0,
        }

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "per_class": per_class_dict,
        "total_samples": int(len(y_true)),
    }


def format_metrics_table(models_metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a dictionary of model metrics into a clean presentation DataFrame.
    """
    rows = []
    for model_name, metrics in models_metrics.items():
        auc_val = f"{metrics['roc_auc']:.4f}" if metrics.get("roc_auc") is not None else "N/A"
        neg_metrics = metrics.get("per_class", {}).get("Negative", {})
        pos_metrics = metrics.get("per_class", {}).get("Positive", {})
        rows.append({
            "Model": model_name,
            "Accuracy": f"{metrics['accuracy']:.2%}",
            "Macro F1": f"{metrics['f1_macro']:.4f}",
            "ROC-AUC": auc_val,
            "Negative Recall": f"{neg_metrics.get('recall', 0.0):.2%}",
            "Positive Recall": f"{pos_metrics.get('recall', 0.0):.2%}",
            "Samples": metrics.get("total_samples", 0),
        })
    return pd.DataFrame(rows)
