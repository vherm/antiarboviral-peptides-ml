"""Utilities for saving model evaluation reports."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from fpdf import FPDF


def _format_metric_value(value: Any) -> str:
    """Convert a metric value to a readable string for PDF reports."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.4f}"

    if isinstance(value, np.ndarray):
        return np.array2string(value)

    return str(value)


def save_excel_report(
    metrics: Mapping[str, Any],
    output_file: str | Path,
    column_name: str = "Value",
) -> None:
    """Save model metrics to an Excel file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_dataframe = pd.DataFrame.from_dict(
        metrics,
        orient="index",
        columns=[column_name],
    )

    metrics_dataframe.to_excel(output_path)
    print(f"[OK] Excel report saved: {output_path}")


def save_pdf_report(metrics: Mapping[str, Any], output_file: str | Path) -> None:
    """Save model metrics to a PDF file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, txt="MODEL EVALUATION REPORT", ln=True, align="C")
    pdf.ln(5)

    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_name = metrics.get("Dataset", metrics.get("CSV file", ""))
    algorithm_name = metrics.get("Algorithm", "Unknown")
    feature_selection_method = metrics.get("Feature selection method", "")

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, txt=f"Generated on: {generation_time}", ln=True)

    if dataset_name:
        pdf.cell(0, 10, txt=f"Dataset: {Path(str(dataset_name)).name}", ln=True)

    pdf.cell(0, 10, txt=f"Algorithm: {algorithm_name}", ln=True)

    if feature_selection_method:
        pdf.cell(0, 10, txt=f"Feature selection method: {feature_selection_method}", ln=True)

    pdf.ln(5)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, txt="Evaluation metrics:", ln=True)
    pdf.set_font("Arial", "", 12)

    metadata_keys = {
        "Dataset",
        "CSV file",
        "Algorithm",
        "Feature selection method",
    }

    for metric_name, metric_value in metrics.items():
        if metric_name in metadata_keys:
            continue

        formatted_value = _format_metric_value(metric_value)
        pdf.multi_cell(0, 8, txt=f"{metric_name}: {formatted_value}")

    pdf.ln(10)

    pdf.set_font("Arial", "I", 10)
    pdf.cell(
        0,
        10,
        txt="Generated automatically by the peptide model evaluation pipeline",
        ln=True,
        align="C",
    )

    pdf.output(str(output_path))
    print(f"[OK] PDF report saved: {output_path}")
