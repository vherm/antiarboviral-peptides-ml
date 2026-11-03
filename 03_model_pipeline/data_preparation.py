"""Data loading and preparation utilities for peptide ML experiments.

Expected CSV format:
#, sequence, label, feature_1, feature_2, ..., feature_n

This module assumes that train/test splitting has already been performed by a
separate script. It does not create train/test splits internally.
"""

from pathlib import Path
import re

import pandas as pd


METADATA_COLUMNS = {
    "#",
    "id",
    "ids",
    "name",
    "nombre",
    "peptide",
    "peptide_id",
    "peptido",
    "péptido",
    "sequence",
    "secuencia",
    "sequence_id",
    "seq_id",
    "accession",
    "header",
    "description",
    "function",
    "split",
    "training",
    "train",
    "test",
    "group",
    "grupo",
    "unnamed: 0",
    "unnamed: 0.1",
}

DANGEROUS_COLUMN_PATTERNS = [
    r"^label(\.|_|$)",
    r"^target(\.|_|$)",
    r"^class(\.|_|$)",
    r"^category(\.|_|$)",
    r"^categoria(\.|_|$)",
    r"^sequence(_id)?$",
    r"^secuencia$",
    r"^peptide(_id)?$",
    r"^peptido$",
    r"^péptido$",
    r"^seq_id$",
    r"^function$",
    r"^split$",
    r"^training$",
    r"^test$",
    r"^train$",
    r"^#$",
    r"^id$",
    r"^name$",
    r"^nombre$",
]


TRUE_FALSE_DTYPES = ["number", "bool"]


def safe_read_csv(csv_path):
    """Read a CSV file using fallback separators and encodings."""
    csv_path = Path(csv_path)

    try:
        return pd.read_csv(csv_path, low_memory=False)
    except Exception as error:
        print(f"[WARN] Standard CSV reading failed for {csv_path}: {error}")

    for separator in [None, ",", ";", "\t", "|"]:
        try:
            dataframe = pd.read_csv(
                csv_path,
                sep=separator,
                engine="python",
                encoding="utf-8-sig",
                on_bad_lines="skip",
            )
            print(
                f"[INFO] CSV read with separator={separator!r} "
                "and encoding='utf-8-sig'"
            )
            return dataframe
        except Exception as error:
            print(f"[WARN] Reading attempt with separator={separator!r} failed: {error}")

    for encoding in ["latin-1", "cp1252"]:
        try:
            dataframe = pd.read_csv(
                csv_path,
                engine="python",
                encoding=encoding,
                on_bad_lines="skip",
            )
            print(f"[INFO] CSV read with fallback encoding={encoding!r}")
            return dataframe
        except Exception as error:
            print(f"[WARN] Reading attempt with encoding={encoding!r} failed: {error}")

    raise RuntimeError(f"Could not read CSV file: {csv_path}")


def normalize_column_name(column_name):
    """Normalize a column name for robust metadata-column detection."""
    column_name = str(column_name).strip().lower()
    column_name = column_name.replace("\ufeff", "").strip()
    column_name = column_name.strip('"').strip("'").strip()
    return column_name


def normalize_dataframe_columns(dataframe):
    """Return a copy of a dataframe with normalized column names."""
    dataframe = dataframe.copy()
    dataframe.columns = [normalize_column_name(column) for column in dataframe.columns]
    return dataframe


def load_descriptors(csv_path):
    """Load one descriptor CSV file and normalize its column names."""
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Input file must be a CSV file: {csv_path}")

    print(f"[INFO] Loading descriptor file: {csv_path}")
    dataframe = safe_read_csv(csv_path)
    dataframe = normalize_dataframe_columns(dataframe)

    duplicated_label_columns = [
        column for column in dataframe.columns
        if column == "label" or column.startswith("label.")
    ]

    if len(duplicated_label_columns) > 1:
        print(f"[INFO] Duplicate label columns found: {duplicated_label_columns}")
        dataframe = dataframe.drop(columns=duplicated_label_columns[1:])

    if "label" not in dataframe.columns:
        raise ValueError("The input CSV file must contain a 'label' column.")

    return dataframe


def remove_non_predictive_columns(dataframe, dataset_name="DATASET"):
    """Remove metadata and non-numeric columns while preserving the label column."""
    dataframe = normalize_dataframe_columns(dataframe)

    print(f"[DEBUG] {dataset_name} - First 10 columns before filtering:", list(dataframe.columns[:10]))

    if "label" not in dataframe.columns:
        raise ValueError(f"The 'label' column was not found in {dataset_name}.")

    metadata_columns_present = [
        column for column in dataframe.columns
        if column in METADATA_COLUMNS and column != "label"
    ]

    if metadata_columns_present:
        print(f"[INFO] {dataset_name} - Metadata columns removed: {metadata_columns_present}")
        dataframe = dataframe.drop(columns=metadata_columns_present)
    else:
        print(f"[INFO] {dataset_name} - No explicit metadata columns were found.")

    dataframe["label"] = pd.to_numeric(dataframe["label"], errors="coerce")

    invalid_label_count = int(dataframe["label"].isna().sum())
    if invalid_label_count > 0:
        raise ValueError(
            f"{dataset_name} contains {invalid_label_count} rows with missing "
            "or non-numeric labels."
        )

    non_numeric_columns = dataframe.select_dtypes(exclude=TRUE_FALSE_DTYPES).columns.tolist()
    non_numeric_columns = [column for column in non_numeric_columns if column != "label"]

    if non_numeric_columns:
        print(f"[INFO] {dataset_name} - Non-numeric columns removed: {non_numeric_columns}")
        dataframe = dataframe.drop(columns=non_numeric_columns)
    else:
        print(f"[INFO] {dataset_name} - No additional non-numeric columns were found.")

    remaining_metadata_columns = [
        column for column in dataframe.columns
        if column in METADATA_COLUMNS and column != "label"
    ]

    if remaining_metadata_columns:
        raise ValueError(
            f"Metadata columns are still present in {dataset_name}: "
            f"{remaining_metadata_columns}"
        )

    if dataframe.shape[1] < 2:
        raise ValueError(
            f"{dataset_name} must contain the label column and at least one feature column."
        )

    print(f"[INFO] {dataset_name} - First 10 final columns:", list(dataframe.columns[:10]))
    return dataframe


def find_suspicious_feature_columns(columns):
    """Identify feature columns that look like labels, metadata, or split markers."""
    suspicious_columns = []

    for column in columns:
        normalized_column = normalize_column_name(column)

        if normalized_column in METADATA_COLUMNS or normalized_column == "label":
            suspicious_columns.append(column)
            continue

        if any(re.search(pattern, normalized_column) for pattern in DANGEROUS_COLUMN_PATTERNS):
            suspicious_columns.append(column)

    return suspicious_columns


def validate_feature_matrices(X_train, X_test, y_train, y_test):
    """Run defensive checks before model fitting."""
    suspicious_train_columns = find_suspicious_feature_columns(X_train.columns)
    suspicious_test_columns = find_suspicious_feature_columns(X_test.columns)

    print("[CHECK] Suspicious columns in X_train:", suspicious_train_columns)
    print("[CHECK] Suspicious columns in X_test:", suspicious_test_columns)

    if suspicious_train_columns or suspicious_test_columns:
        raise ValueError(
            "Metadata or label-like columns were detected in the feature matrices. "
            f"X_train={suspicious_train_columns}, X_test={suspicious_test_columns}"
        )

    non_numeric_train_columns = X_train.select_dtypes(exclude=TRUE_FALSE_DTYPES).columns.tolist()
    non_numeric_test_columns = X_test.select_dtypes(exclude=TRUE_FALSE_DTYPES).columns.tolist()

    print("[CHECK] Non-numeric columns in X_train:", non_numeric_train_columns)
    print("[CHECK] Non-numeric columns in X_test:", non_numeric_test_columns)

    if non_numeric_train_columns or non_numeric_test_columns:
        raise ValueError(
            "Non-numeric columns are still present in the feature matrices. "
            f"X_train={non_numeric_train_columns}, X_test={non_numeric_test_columns}"
        )

    train_missing_values = int(X_train.isna().sum().sum())
    test_missing_values = int(X_test.isna().sum().sum())

    print("[CHECK] Missing values in X_train:", train_missing_values)
    print("[CHECK] Missing values in X_test:", test_missing_values)

    print("[CHECK] y_train distribution:")
    print(y_train.value_counts(dropna=False).sort_index())

    print("[CHECK] y_test distribution:")
    print(y_test.value_counts(dropna=False).sort_index())

    duplicated_train_rows = int(X_train.duplicated().sum())
    duplicated_test_rows = int(X_test.duplicated().sum())

    print("[CHECK] Exact duplicate rows within X_train:", duplicated_train_rows)
    print("[CHECK] Exact duplicate rows within X_test:", duplicated_test_rows)

    train_check = X_train.copy()
    test_check = X_test.copy()

    train_check["_split_origin_"] = "train"
    test_check["_split_origin_"] = "test"

    combined = pd.concat([train_check, test_check], ignore_index=True)
    feature_columns = [column for column in combined.columns if column != "_split_origin_"]

    duplicated_across_splits = combined.duplicated(subset=feature_columns, keep=False)
    duplicated_rows_across_splits = combined.loc[duplicated_across_splits]

    if duplicated_rows_across_splits.empty:
        repeated_feature_profiles = 0
    else:
        repeated_groups = duplicated_rows_across_splits.groupby(
            feature_columns,
            dropna=False,
        )["_split_origin_"].nunique()
        repeated_feature_profiles = int((repeated_groups >= 2).sum())

    print(
        "[CHECK] Exact feature rows present in both train and test:",
        repeated_feature_profiles,
    )

    if repeated_feature_profiles > 0:
        print(
            "[WARN] Some exact feature rows are present in both train and test. "
            "This may inflate metrics if they correspond to identical or highly similar peptides."
        )


def prepare_train_test_data(train_dataframe, test_dataframe):
    """Convert externally generated train/test dataframes into X_train, X_test, y_train, and y_test.

    This function assumes that train and test files were generated from the same
    original CSV file and preserve the same feature columns in the same order.
    Users should verify this before running the modeling pipeline.
    """
    train_dataframe = remove_non_predictive_columns(train_dataframe, dataset_name="TRAIN")
    test_dataframe = remove_non_predictive_columns(test_dataframe, dataset_name="TEST")


    y_train = train_dataframe["label"].astype(int)
    y_test = test_dataframe["label"].astype(int)

    X_train = train_dataframe.drop(columns=["label"])
    X_test = test_dataframe.drop(columns=["label"])

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    validate_feature_matrices(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test
