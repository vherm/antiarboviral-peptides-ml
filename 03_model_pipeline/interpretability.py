"""
Interpretability utilities for the anti-arboviral peptide ML pipeline.

This module provides complementary interpretability analyses for the finalist
models selected after Round 2:

1. Permutation importance (PI)
   - Model-agnostic.
   - Used for XGBoost, Random Forest, and SVM.
   - By default, importance is measured as the decrease in Average Precision (AP).
   - Parallelized using get_parallel_jobs() from model_tuning.py.

2. Grouped permutation importance for iLearn
   - Permutes all retained features from the same descriptor family together.
   - Uses the same row permutation for every feature in a family so the internal
     multivariate structure of the family is preserved while its association
     with the target is disrupted.

3. SHAP importance
   - Used only for tree-based models: XGBoost and Random Forest.
   - Uses TreeExplainer on the full transformed test set.
   - For iLearn, SHAP values are also aggregated by descriptor family.

Expected outputs (depending on model/representation/mode):
    <algorithm>_<feature_selection_method>_permutation_importance.csv
    <algorithm>_<feature_selection_method>_top<N>_permutation_importance.csv
    <algorithm>_<feature_selection_method>_top<N>_permutation_importance.png

    <algorithm>_<feature_selection_method>_family_permutation_importance.csv
    <algorithm>_<feature_selection_method>_family_permutation_importance.png

    <algorithm>_<feature_selection_method>_shap_importance.csv
    <algorithm>_<feature_selection_method>_top<N>_shap_importance.csv
    <algorithm>_<feature_selection_method>_top<N>_shap_importance.png

    <algorithm>_<feature_selection_method>_shap_family_importance.csv
    <algorithm>_<feature_selection_method>_shap_family_importance.png

    <algorithm>_<feature_selection_method>_interpretability_metadata.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, f1_score, make_scorer, matthews_corrcoef
from sklearn.pipeline import Pipeline

from seed_utils import SEED
from model_tuning import get_parallel_jobs


PERMUTATION_SUPPORTED_ALGORITHMS = {"XGBoost", "RF", "SVM"}
SHAP_SUPPORTED_ALGORITHMS = {"XGBoost", "RF"}
SUPPORTED_REPRESENTATIONS = {"esm-2", "ilearn"}

MCC_SCORER = make_scorer(matthews_corrcoef)
F1_SCORER = make_scorer(f1_score, zero_division=0)

# The order is intentionally longest/most specific first. This avoids ambiguous
# matches such as DPC inside GDPC, TPC inside GTPC, and PAAC inside APAAC.
ILEARN_FAMILY_ALIASES = [
    ("cksaagp", "CKSAAGP"),
    ("cksaap", "CKSAAP"),
    ("ksctriad", "KSCTriad"),
    ("socnumber", "SOCNumber"),
    ("nmbroto", "NMBroto"),
    ("qsorder", "QSOrder"),
    ("ctriad", "CTRIAD"),
    ("apaac", "APAAC"),
    ("geary", "Geary"),
    ("moran", "Moran"),
    ("ctdc", "CTDC"),
    ("ctdd", "CTDD"),
    ("ctdt", "CTDT"),
    ("gdpc", "GDPC"),
    ("gtpc", "GTPC"),
    ("paac", "PAAC"),
    ("gaac", "GAAC"),
    ("dde", "DDE"),
    ("dpc", "DPC"),
    ("tpc", "TPC"),
    ("aac", "AAC"),
]


def sanitize_name(value: str) -> str:
    """Return a filesystem-friendly version of a name."""
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "value"


def normalize_representation(value: str) -> str:
    """Normalize and validate the descriptor representation name."""
    normalized = str(value).strip().lower().replace("_", "-")

    if normalized in {"esm2", "esm-2"}:
        return "esm-2"
    if normalized == "ilearn":
        return "ilearn"

    raise ValueError(
        "Unsupported representation. Expected 'ESM-2' or 'iLearn', "
        f"received: {value!r}"
    )


def get_feature_names(X) -> list[str]:
    """Return feature names from a DataFrame or generate generic names for arrays."""
    if isinstance(X, pd.DataFrame):
        return list(X.columns)

    n_features = X.shape[1]
    return [f"f{i}" for i in range(n_features)]


def get_final_estimator_step_name(pipeline: Pipeline) -> str:
    """Return the step name of the final estimator in a fitted sklearn Pipeline."""
    if not isinstance(pipeline, Pipeline):
        raise TypeError("The fitted model must be a sklearn Pipeline.")

    return pipeline.steps[-1][0]


def get_final_estimator(pipeline: Pipeline):
    """Return the final estimator from a fitted sklearn Pipeline."""
    if not isinstance(pipeline, Pipeline):
        raise TypeError("The fitted model must be a sklearn Pipeline.")

    return pipeline.steps[-1][1]


def get_preprocessing_pipeline(pipeline: Pipeline) -> Pipeline:
    """Return all fitted pipeline steps except the final estimator."""
    if not isinstance(pipeline, Pipeline):
        raise TypeError("The fitted model must be a sklearn Pipeline.")

    if len(pipeline.steps) <= 1:
        return Pipeline([])

    return Pipeline(pipeline.steps[:-1])


def transform_features_before_estimator(pipeline: Pipeline, X):
    """Apply all fitted preprocessing/feature-selection steps before the estimator."""
    preprocessing_pipeline = get_preprocessing_pipeline(pipeline)

    if len(preprocessing_pipeline.steps) == 0:
        return X

    return preprocessing_pipeline.transform(X)


def get_transformed_feature_names(
    pipeline: Pipeline,
    original_feature_names: Iterable[str],
) -> list[str]:
    """
    Track feature names through fitted pipeline steps that expose get_support().

    This handles VarianceThreshold, CorrelationFilter, SelectKBest/
    SelectPercentile, RFE, RFECV, and SelectFromModel. Scalers and imputers keep
    names unchanged.
    """
    feature_names = list(original_feature_names)

    if not isinstance(pipeline, Pipeline):
        return feature_names

    for step_name, step in pipeline.steps[:-1]:
        if hasattr(step, "get_support"):
            support_mask = np.asarray(step.get_support(), dtype=bool)

            if len(support_mask) != len(feature_names):
                print(
                    f"[WARN] Could not align feature names after step '{step_name}'. "
                    "Using generic transformed feature names."
                )
                return [
                    f"feature_{i}"
                    for i in range(int(support_mask.sum()))
                ]

            feature_names = [
                name
                for name, keep in zip(feature_names, support_mask)
                if keep
            ]

    return feature_names


def get_transformed_test_data(pipeline: Pipeline, X_test):
    """Return transformed test data and aligned feature names."""
    original_feature_names = get_feature_names(X_test)
    transformed_feature_names = get_transformed_feature_names(
        pipeline,
        original_feature_names,
    )

    transformed_X_test = transform_features_before_estimator(
        pipeline,
        X_test,
    )
    transformed_X_test = np.asarray(transformed_X_test)

    if transformed_X_test.shape[1] != len(transformed_feature_names):
        print(
            "[WARN] Number of transformed features does not match the number "
            "of tracked feature names. Using generic transformed feature names."
        )
        transformed_feature_names = [
            f"feature_{i}"
            for i in range(transformed_X_test.shape[1])
        ]

    return transformed_X_test, transformed_feature_names


def get_positive_class_values(values):
    """
    Return the positive-class SHAP matrix for binary classification when needed.

    SHAP can return:
    - ndarray with shape (n_samples, n_features)
    - ndarray with shape (n_samples, n_features, n_classes)
    - list of arrays, one per class
    """
    if isinstance(values, list):
        if len(values) == 1:
            return values[0]
        return values[1]

    values = np.asarray(values)

    if values.ndim == 3:
        return values[:, :, 1]

    return values


def build_importance_dataframe(
    feature_names: list[str],
    importances: np.ndarray,
    std_values: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Build a ranked feature-importance DataFrame."""
    importances = np.asarray(importances)

    data = {
        "feature": feature_names,
        "importance": importances,
    }

    if std_values is not None:
        data["importance_std"] = np.asarray(std_values)

    df = pd.DataFrame(data)
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def save_top_features_plot(
    importance_dataframe: pd.DataFrame,
    output_file: str | Path,
    title: str,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar plot with the top feature importances."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_df = importance_dataframe.head(top_n).copy()

    plt.figure(figsize=(9, max(5, 0.35 * len(top_df))))
    plt.barh(top_df["feature"][::-1], top_df["importance"][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_family_importance_plot(
    family_dataframe: pd.DataFrame,
    importance_column: str,
    output_file: str | Path,
    title: str,
) -> None:
    """Save a horizontal bar plot for descriptor-family importance."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = family_dataframe.sort_values(
        importance_column,
        ascending=True,
    ).copy()

    plt.figure(figsize=(9, max(5, 0.38 * len(plot_df))))
    plt.barh(plot_df["family"], plot_df[importance_column])
    plt.xlabel(importance_column.replace("_", " ").title())
    plt.ylabel("iLearn descriptor family")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def normalize_ilearn_name(text: str) -> str:
    """Normalize a feature/family name exactly as in the Round-2 family logic."""
    return re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()


def infer_ilearn_family(feature_name: str) -> Optional[str]:
    """
    Infer the iLearn descriptor family from a feature-column name.

    This intentionally follows the same matching strategy used in the
    group-normalized Euclidean negative-selection scripts from Round 2:
    1. normalized prefix match;
    2. delimiter-aware token match;
    3. return None if no known family can be identified.
    """
    raw = str(feature_name)
    normalized = normalize_ilearn_name(raw)

    known_families = [family for _, family in ILEARN_FAMILY_ALIASES]
    families_sorted = sorted(
        known_families,
        key=lambda value: len(normalize_ilearn_name(value)),
        reverse=True,
    )

    for family in families_sorted:
        family_normalized = normalize_ilearn_name(family)
        if normalized.startswith(family_normalized):
            return family

    tokens = [
        token
        for token in re.split(r"[\s_:\-|./()\[\]]+", raw)
        if token
    ]
    normalized_tokens = {normalize_ilearn_name(token) for token in tokens}

    for family in families_sorted:
        if normalize_ilearn_name(family) in normalized_tokens:
            return family

    return None

def map_ilearn_families(
    feature_names: Iterable[str],
    strict: bool = True,
) -> list[str]:
    """Map transformed iLearn feature names to descriptor families."""
    families = []
    unknown = []

    for feature_name in feature_names:
        family = infer_ilearn_family(feature_name)
        if family is None:
            unknown.append(str(feature_name))
            families.append("UNKNOWN")
        else:
            families.append(family)

    if unknown and strict:
        preview = unknown[:20]
        extra_count = max(0, len(unknown) - len(preview))
        extra_text = (
            f" (+{extra_count} additional feature(s))"
            if extra_count
            else ""
        )
        raise ValueError(
            "Could not infer an iLearn descriptor family for the following "
            f"selected feature(s): {preview}{extra_text}. "
            "Check the feature-name convention and update ILEARN_FAMILY_ALIASES "
            "or infer_ilearn_family()."
        )

    return families


def normalize_permutation_scoring(scoring: str) -> str:
    """Normalize and validate the metric used for permutation importance."""
    scoring_normalized = str(scoring).strip().lower()

    aliases = {
        "ap": "average_precision",
        "average-precision": "average_precision",
        "average_precision": "average_precision",
        "mcc": "mcc",
        "f1": "f1",
    }

    if scoring_normalized not in aliases:
        raise ValueError(
            "Unsupported permutation scoring. Expected one of: "
            "'average_precision', 'mcc', 'f1'."
        )

    return aliases[scoring_normalized]


def resolve_permutation_scoring(scoring: str):
    """Return the scorer used by sklearn permutation_importance()."""
    scoring_normalized = normalize_permutation_scoring(scoring)

    if scoring_normalized == "average_precision":
        return "average_precision"
    if scoring_normalized == "mcc":
        return MCC_SCORER
    return F1_SCORER


def get_positive_class_scores(final_estimator, X):
    """Return continuous positive-class scores for rank-based metrics."""
    if hasattr(final_estimator, "predict_proba"):
        try:
            probabilities = np.asarray(final_estimator.predict_proba(X))
            if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                return probabilities[:, 1]
        except Exception:
            pass

    if hasattr(final_estimator, "decision_function"):
        try:
            scores = np.asarray(final_estimator.decision_function(X))
            if scores.ndim == 2 and scores.shape[1] >= 2:
                return scores[:, 1]
            return scores.ravel()
        except Exception:
            pass

    raise ValueError(
        "The estimator does not provide predict_proba() or decision_function(), "
        "which is required for average_precision permutation importance."
    )


def compute_metric_score(
    final_estimator,
    X,
    y_true,
    scoring: str,
) -> float:
    """Compute the selected permutation-importance metric on a feature matrix."""
    scoring_normalized = normalize_permutation_scoring(scoring)

    if scoring_normalized == "average_precision":
        y_scores = get_positive_class_scores(final_estimator, X)
        return float(average_precision_score(y_true, y_scores))

    y_pred = final_estimator.predict(X)

    if scoring_normalized == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))

    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_baseline_mcc(final_estimator, transformed_X_test, y_test) -> float:
    """Compute baseline MCC on the transformed external test set."""
    y_pred = final_estimator.predict(transformed_X_test)
    return float(matthews_corrcoef(y_test, y_pred))


def compute_permutation_importance(
    pipeline: Pipeline,
    X_test,
    y_test,
    algorithm: str,
    feature_selection_method: str,
    representation: str,
    output_dir: str | Path = ".",
    top_n: int = 20,
    n_repeats: int = 10,
    scoring: str = "average_precision",
    strict_ilearn_family_mapping: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Compute individual-feature permutation importance using the transformed
    features received by the final estimator.
    """
    if algorithm not in PERMUTATION_SUPPORTED_ALGORITHMS:
        print(
            f"[INFO] Permutation importance skipped for algorithm '{algorithm}'."
        )
        return None

    representation = normalize_representation(representation)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transformed_X_test, transformed_feature_names = get_transformed_test_data(
        pipeline,
        X_test,
    )

    final_estimator = get_final_estimator(pipeline)
    n_jobs = get_parallel_jobs()
    scorer = resolve_permutation_scoring(scoring)

    print(
        f"[INFO] Computing permutation importance for {algorithm} "
        f"using n_jobs={n_jobs}, n_repeats={n_repeats}, "
        f"scoring='{scoring}'."
    )

    result = permutation_importance(
        final_estimator,
        transformed_X_test,
        y_test,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=SEED,
        n_jobs=n_jobs,
    )

    importance_df = build_importance_dataframe(
        feature_names=transformed_feature_names,
        importances=result.importances_mean,
        std_values=result.importances_std,
    )

    if representation == "ilearn":
        family_by_feature = dict(
            zip(
                transformed_feature_names,
                map_ilearn_families(
                    transformed_feature_names,
                    strict=strict_ilearn_family_mapping,
                ),
            )
        )
        importance_df.insert(
            2,
            "family",
            importance_df["feature"].map(family_by_feature),
        )

    prefix = (
        f"{sanitize_name(algorithm)}_"
        f"{sanitize_name(feature_selection_method)}"
    )

    full_csv = output_dir / f"{prefix}_permutation_importance.csv"
    top_csv = output_dir / f"{prefix}_top{top_n}_permutation_importance.csv"
    top_png = output_dir / f"{prefix}_top{top_n}_permutation_importance.png"

    importance_df.to_csv(full_csv, index=False, encoding="utf-8")
    importance_df.head(top_n).to_csv(top_csv, index=False, encoding="utf-8")

    save_top_features_plot(
        importance_dataframe=importance_df,
        output_file=top_png,
        title=(
            f"Top {top_n} permutation importance features "
            f"({algorithm}, {feature_selection_method}; "
            f"{normalize_permutation_scoring(scoring).replace('_', ' ').title()})"
        ),
        top_n=top_n,
    )

    print(f"[OK] Permutation importance saved to: {full_csv}")
    print(f"[OK] Top permutation features saved to: {top_csv}")
    print(f"[OK] Permutation importance plot saved to: {top_png}")

    return importance_df


def compute_grouped_permutation_importance(
    pipeline: Pipeline,
    X_test,
    y_test,
    algorithm: str,
    feature_selection_method: str,
    output_dir: str | Path = ".",
    n_repeats: int = 10,
    scoring: str = "average_precision",
    strict_ilearn_family_mapping: bool = True,
) -> pd.DataFrame:
    """
    Compute grouped iLearn descriptor-family permutation importance.

    All retained features belonging to one descriptor family are moved together
    using the same row permutation. Importance is the baseline score minus the
    score obtained after permuting that family.

    For Round 4A, ``average_precision`` is recommended because it uses
    continuous prediction scores and therefore remains sensitive when a
    permutation changes ranking/confidence without changing the final 0/1 class.
    """
    if algorithm not in PERMUTATION_SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Grouped permutation importance is not supported for '{algorithm}'."
        )

    scoring_normalized = normalize_permutation_scoring(scoring)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transformed_X_test, transformed_feature_names = get_transformed_test_data(
        pipeline,
        X_test,
    )
    families = map_ilearn_families(
        transformed_feature_names,
        strict=strict_ilearn_family_mapping,
    )

    final_estimator = get_final_estimator(pipeline)
    baseline_score = compute_metric_score(
        final_estimator,
        transformed_X_test,
        y_test,
        scoring=scoring_normalized,
    )

    family_to_indices: dict[str, list[int]] = {}
    for column_index, family in enumerate(families):
        family_to_indices.setdefault(family, []).append(column_index)

    rng = np.random.default_rng(SEED)
    rows = []

    print(
        f"[INFO] Computing grouped iLearn permutation importance for {algorithm} "
        f"using {len(family_to_indices)} retained family/families, "
        f"n_repeats={n_repeats}, scoring='{scoring_normalized}', "
        f"baseline_score={baseline_score:.6f}."
    )

    for family, indices in family_to_indices.items():
        repeat_importances = []
        repeat_scores = []

        for _ in range(n_repeats):
            row_permutation = rng.permutation(transformed_X_test.shape[0])
            permuted_X = transformed_X_test.copy()

            permuted_X[:, indices] = transformed_X_test[
                row_permutation
            ][:, indices]

            permuted_score = compute_metric_score(
                final_estimator,
                permuted_X,
                y_test,
                scoring=scoring_normalized,
            )
            repeat_scores.append(permuted_score)
            repeat_importances.append(baseline_score - permuted_score)

        rows.append(
            {
                "family": family,
                "n_features": len(indices),
                "importance": float(np.mean(repeat_importances)),
                "importance_std": float(np.std(repeat_importances, ddof=0)),
                "scoring": scoring_normalized,
                "baseline_score": float(baseline_score),
                "permuted_score_mean": float(np.mean(repeat_scores)),
                "permuted_score_std": float(np.std(repeat_scores, ddof=0)),
                "n_repeats": int(n_repeats),
            }
        )

    family_df = pd.DataFrame(rows)
    family_df = family_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)
    family_df.insert(0, "rank", np.arange(1, len(family_df) + 1))

    prefix = (
        f"{sanitize_name(algorithm)}_"
        f"{sanitize_name(feature_selection_method)}"
    )
    output_csv = output_dir / f"{prefix}_family_permutation_importance.csv"
    output_png = output_dir / f"{prefix}_family_permutation_importance.png"

    family_df.to_csv(output_csv, index=False, encoding="utf-8")
    save_family_importance_plot(
        family_dataframe=family_df,
        importance_column="importance",
        output_file=output_png,
        title=(
            "iLearn family permutation importance "
            f"({algorithm}, {feature_selection_method}; "
            f"{scoring_normalized.replace('_', ' ').title()})"
        ),
    )

    print(f"[OK] Family permutation importance saved to: {output_csv}")
    print(f"[OK] Family permutation importance plot saved to: {output_png}")

    return family_df


def aggregate_shap_by_family(
    importance_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate mean absolute SHAP values by iLearn descriptor family."""
    family_df = (
        importance_df.groupby("family", as_index=False)
        .agg(
            n_features=("feature", "size"),
            total_mean_abs_shap=("mean_abs_shap", "sum"),
            mean_abs_shap_per_feature=("mean_abs_shap", "mean"),
        )
    )

    total_shap = float(family_df["total_mean_abs_shap"].sum())
    if total_shap > 0:
        family_df["fraction_total_shap"] = (
            family_df["total_mean_abs_shap"] / total_shap
        )
    else:
        family_df["fraction_total_shap"] = 0.0

    family_df = family_df.sort_values(
        "total_mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)
    family_df.insert(0, "rank", np.arange(1, len(family_df) + 1))
    return family_df


def compute_shap_importance(
    pipeline: Pipeline,
    X_test,
    algorithm: str,
    feature_selection_method: str,
    representation: str,
    output_dir: str | Path = ".",
    top_n: int = 20,
    strict_ilearn_family_mapping: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Compute SHAP feature importance for tree-based models using the full test set.

    SHAP is applied to the final estimator after all fitted preprocessing and
    feature-selection transformations. For iLearn, individual SHAP importance is
    also aggregated by descriptor family.
    """
    if algorithm not in SHAP_SUPPORTED_ALGORITHMS:
        print(f"[INFO] SHAP skipped for algorithm '{algorithm}'.")
        return None

    try:
        import shap
    except ImportError:
        print("[WARN] SHAP is not installed. Install it with: pip install shap")
        return None

    representation = normalize_representation(representation)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transformed_X_test, transformed_feature_names = get_transformed_test_data(
        pipeline,
        X_test,
    )

    final_estimator = get_final_estimator(pipeline)

    print(
        f"[INFO] Computing SHAP importance for {algorithm} using the full test set "
        f"({transformed_X_test.shape[0]} samples, "
        f"{transformed_X_test.shape[1]} features)."
    )

    explainer = shap.TreeExplainer(final_estimator)
    raw_shap_values = explainer.shap_values(transformed_X_test)
    shap_values = np.asarray(get_positive_class_values(raw_shap_values))

    if shap_values.ndim != 2:
        raise ValueError(
            "Unexpected SHAP value shape after positive-class extraction: "
            f"{shap_values.shape}"
        )

    if shap_values.shape[1] != len(transformed_feature_names):
        raise ValueError(
            "SHAP feature dimension does not match the transformed feature names: "
            f"{shap_values.shape[1]} vs {len(transformed_feature_names)}"
        )

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = build_importance_dataframe(
        feature_names=transformed_feature_names,
        importances=mean_abs_shap,
        std_values=None,
    )
    importance_df = importance_df.rename(
        columns={"importance": "mean_abs_shap"}
    )

    if representation == "ilearn":
        family_by_feature = dict(
            zip(
                transformed_feature_names,
                map_ilearn_families(
                    transformed_feature_names,
                    strict=strict_ilearn_family_mapping,
                ),
            )
        )
        importance_df.insert(
            2,
            "family",
            importance_df["feature"].map(family_by_feature),
        )

    # Shared plotting helper expects an 'importance' column.
    plot_df = importance_df.rename(
        columns={"mean_abs_shap": "importance"}
    )

    prefix = (
        f"{sanitize_name(algorithm)}_"
        f"{sanitize_name(feature_selection_method)}"
    )

    full_csv = output_dir / f"{prefix}_shap_importance.csv"
    top_csv = output_dir / f"{prefix}_top{top_n}_shap_importance.csv"
    top_png = output_dir / f"{prefix}_top{top_n}_shap_importance.png"

    importance_df.to_csv(full_csv, index=False, encoding="utf-8")
    importance_df.head(top_n).to_csv(top_csv, index=False, encoding="utf-8")

    save_top_features_plot(
        importance_dataframe=plot_df,
        output_file=top_png,
        title=f"Top {top_n} SHAP features ({algorithm}, {feature_selection_method})",
        top_n=top_n,
    )

    print(f"[OK] SHAP importance saved to: {full_csv}")
    print(f"[OK] Top SHAP features saved to: {top_csv}")
    print(f"[OK] SHAP importance plot saved to: {top_png}")

    if representation == "ilearn":
        family_df = aggregate_shap_by_family(importance_df)
        family_csv = output_dir / f"{prefix}_shap_family_importance.csv"
        family_png = output_dir / f"{prefix}_shap_family_importance.png"

        family_df.to_csv(family_csv, index=False, encoding="utf-8")
        save_family_importance_plot(
            family_dataframe=family_df,
            importance_column="total_mean_abs_shap",
            output_file=family_png,
            title=(
                "iLearn family SHAP importance "
                f"({algorithm}, {feature_selection_method})"
            ),
        )

        print(f"[OK] SHAP family importance saved to: {family_csv}")
        print(f"[OK] SHAP family importance plot saved to: {family_png}")

    return importance_df


def save_interpretability_metadata(
    pipeline: Pipeline,
    X_test,
    y_test,
    algorithm: str,
    feature_selection_method: str,
    representation: str,
    mode: str,
    output_dir: str | Path,
    top_n: int,
    permutation_repeats: int,
    permutation_scoring: str,
) -> Path:
    """Save metadata needed to audit and aggregate interpretability results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transformed_X_test, transformed_feature_names = get_transformed_test_data(
        pipeline,
        X_test,
    )
    final_estimator = get_final_estimator(pipeline)
    baseline_mcc = compute_baseline_mcc(
        final_estimator,
        transformed_X_test,
        y_test,
    )
    normalized_permutation_scoring = normalize_permutation_scoring(
        permutation_scoring
    )
    baseline_permutation_score = compute_metric_score(
        final_estimator,
        transformed_X_test,
        y_test,
        scoring=normalized_permutation_scoring,
    )

    metadata = {
        "algorithm": algorithm,
        "feature_selection_method": feature_selection_method,
        "representation": representation,
        "mode": mode,
        "n_test_samples": int(transformed_X_test.shape[0]),
        "n_original_features": int(X_test.shape[1]),
        "n_selected_features": int(transformed_X_test.shape[1]),
        "baseline_mcc": baseline_mcc,
        "baseline_permutation_score": baseline_permutation_score,
        "top_n": int(top_n),
        "permutation_repeats": int(permutation_repeats),
        "permutation_scoring": normalized_permutation_scoring,
        "seed": int(SEED),
        "final_estimator_step": get_final_estimator_step_name(pipeline),
        "selected_feature_names": [str(name) for name in transformed_feature_names],
    }

    prefix = (
        f"{sanitize_name(algorithm)}_"
        f"{sanitize_name(feature_selection_method)}"
    )
    output_file = output_dir / f"{prefix}_interpretability_metadata.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print(f"[OK] Interpretability metadata saved to: {output_file}")
    return output_file


def run_interpretability_analysis(
    pipeline: Pipeline,
    X_test,
    y_test,
    algorithm: str,
    feature_selection_method: str,
    representation: str,
    mode: str = "none",
    output_dir: str | Path = ".",
    top_n: int = 20,
    permutation_repeats: int = 10,
    permutation_scoring: str = "average_precision",
    grouped_ilearn: bool = True,
    strict_ilearn_family_mapping: bool = True,
) -> None:
    """
    Run the requested interpretability analysis.

    Parameters
    ----------
    mode:
        One of: "none", "permutation", "shap", "all".
    representation:
        "ESM-2" or "iLearn".
    permutation_scoring:
        "average_precision" (recommended for Round 4A), "mcc", or "f1".
    grouped_ilearn:
        If True, grouped family permutation importance is computed whenever PI
        is requested for iLearn.
    """
    mode = (mode or "none").lower()
    representation_normalized = normalize_representation(representation)

    if mode == "none":
        print("[INFO] Interpretability analysis skipped.")
        return

    if mode not in {"none", "permutation", "shap", "all"}:
        raise ValueError(
            "Invalid interpretability mode. Expected one of: "
            "none, permutation, shap, all."
        )

    # Validate scoring early so a long model run does not end with a typo here.
    resolve_permutation_scoring(permutation_scoring)

    save_interpretability_metadata(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        algorithm=algorithm,
        feature_selection_method=feature_selection_method,
        representation=representation,
        mode=mode,
        output_dir=output_dir,
        top_n=top_n,
        permutation_repeats=permutation_repeats,
        permutation_scoring=permutation_scoring,
    )

    if mode in {"permutation", "all"}:
        compute_permutation_importance(
            pipeline=pipeline,
            X_test=X_test,
            y_test=y_test,
            algorithm=algorithm,
            feature_selection_method=feature_selection_method,
            representation=representation,
            output_dir=output_dir,
            top_n=top_n,
            n_repeats=permutation_repeats,
            scoring=permutation_scoring,
            strict_ilearn_family_mapping=strict_ilearn_family_mapping,
        )

        if representation_normalized == "ilearn" and grouped_ilearn:
            compute_grouped_permutation_importance(
                pipeline=pipeline,
                X_test=X_test,
                y_test=y_test,
                algorithm=algorithm,
                feature_selection_method=feature_selection_method,
                output_dir=output_dir,
                n_repeats=permutation_repeats,
                scoring=permutation_scoring,
                strict_ilearn_family_mapping=strict_ilearn_family_mapping,
            )

    if mode in {"shap", "all"}:
        compute_shap_importance(
            pipeline=pipeline,
            X_test=X_test,
            algorithm=algorithm,
            feature_selection_method=feature_selection_method,
            representation=representation,
            output_dir=output_dir,
            top_n=top_n,
            strict_ilearn_family_mapping=strict_ilearn_family_mapping,
        )
