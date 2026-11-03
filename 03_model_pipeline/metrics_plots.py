"""Metrics and plotting utilities for binary peptide classification models.

This module computes the main classification metrics and generates the plots
used by the modeling pipeline:
- Confusion matrix
- ROC curve
- Precision-Recall curve

The functions assume a binary classification task where the positive class is
encoded as 1 and the negative class is encoded as 0.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
)


def evaluate_model(y_true, y_pred):
    """Compute classification metrics from true and predicted labels.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.

    Returns
    -------
    dict
        Dictionary containing accuracy, sensitivity, specificity, MCC, F1 score,
        and the confusion matrix.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    sensitivity = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if (true_negative + false_positive) > 0
        else 0
    )

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "Confusion Matrix": matrix,
    }


def plot_confusion_matrix(confusion_matrix_values, class_labels, output_file):
    """Save a confusion matrix plot.

    Parameters
    ----------
    confusion_matrix_values : array-like
        Confusion matrix values, usually obtained from `evaluate_model()`.
    class_labels : list[str]
        Labels to display on the x-axis and y-axis.
    output_file : str or pathlib.Path
        Path where the PNG file will be saved.
    """
    output_file = Path(output_file)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(confusion_matrix_values)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    for row_index in range(confusion_matrix_values.shape[0]):
        for col_index in range(confusion_matrix_values.shape[1]):
            ax.text(
                col_index,
                row_index,
                str(confusion_matrix_values[row_index, col_index]),
                ha="center",
                va="center",
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_scores, output_file):
    """Save the ROC curve and return the ROC AUC.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_scores : array-like
        Continuous scores for the positive class, usually probabilities.
    output_file : str or pathlib.Path
        Path where the PNG file will be saved.

    Returns
    -------
    float
        ROC AUC value.
    """
    output_file = Path(output_file)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(false_positive_rate, true_positive_rate, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    return roc_auc


def plot_precision_recall_curve(y_true, y_scores, output_png, output_csv):
    """Save the Precision-Recall curve and its values.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_scores : array-like
        Continuous scores for the positive class, usually probabilities or
        decision scores.
    output_png : str or pathlib.Path
        Path where the PNG file will be saved.
    output_csv : str or pathlib.Path
        Path where the curve values will be saved as CSV.

    Returns
    -------
    tuple[float, float]
        Average precision and Precision-Recall AUC.
    """
    output_png = Path(output_png)
    output_csv = Path(output_csv)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    pr_auc = auc(recall, precision)

    aligned_thresholds = np.r_[[np.nan], thresholds]
    curve_dataframe = pd.DataFrame(
        {
            "threshold": aligned_thresholds,
            "recall": recall,
            "precision": precision,
        }
    )
    curve_dataframe.to_csv(output_csv, index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.step(recall, precision, where="post")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        f"Precision-Recall Curve (AP={average_precision:.4f}, AUC={pr_auc:.4f})"
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    return average_precision, pr_auc
