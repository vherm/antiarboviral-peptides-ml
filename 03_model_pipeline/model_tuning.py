"""
Model tuning, feature selection, and result-export utilities for peptide classification.

This module builds scikit-learn pipelines that keep preprocessing, feature selection,
and model fitting inside cross-validation. This avoids information leakage from the
external test set.
"""

from __future__ import annotations

import ast
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFECV,
    RFE,
    SelectFromModel,
    SelectPercentile,
    VarianceThreshold,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

from seed_utils import SEED


ALGORITHMS = ["SVM", "RF", "KNN", "NB", "MLP", "XGBoost"]
FEATURE_SELECTION_METHODS = [
    "none",
    "prefilter",
    "percentile",
    "percentile_chi2",
    "rfe",
    "rfecv",
    "sfm_l1",
    "sfm_rf",
    "sfm_xgb",
]

FEATURE_PERCENTILES = [1, 5, 10, 20, 30, 40, 50]


def max_features_2_percent(X):
    return max(1, int(np.ceil(X.shape[1] * 0.02)))


def max_features_5_percent(X):
    return max(1, int(np.ceil(X.shape[1] * 0.05)))


def max_features_10_percent(X):
    return max(1, int(np.ceil(X.shape[1] * 0.10)))


SFM_MAX_FEATURES_GRID = [
    max_features_2_percent,
    max_features_5_percent,
    max_features_10_percent,
    None,
]


def get_parallel_jobs(default_local: Optional[int] = None) -> int:
    """
    Return the number of CPU jobs to use locally or inside a Slurm allocation.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

    if slurm_cpus is not None:
        return int(slurm_cpus)

    if default_local is not None:
        return int(default_local)

    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Remove highly correlated features using only the data seen during fit.

    This transformer is intended to be used inside a scikit-learn Pipeline so
    that correlation filtering is learned independently within each CV fold.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_array = np.asarray(X)

        if X_array.ndim != 2:
            raise ValueError("CorrelationFilter expects a 2D feature matrix.")

        n_features = X_array.shape[1]

        if n_features == 0:
            self.support_mask_ = np.array([], dtype=bool)
            self.features_to_drop_ = []
            return self

        correlation_matrix = pd.DataFrame(X_array).corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        self.features_to_drop_ = [
            column_index
            for column_index in upper_triangle.columns
            if any(upper_triangle[column_index] > self.threshold)
        ]

        self.support_mask_ = np.ones(n_features, dtype=bool)
        self.support_mask_[self.features_to_drop_] = False

        return self

    def transform(self, X):
        X_array = np.asarray(X)
        return X_array[:, self.support_mask_]

    def get_support(self):
        return self.support_mask_


def get_feature_selection_stats_and_export(
    pipeline,
    X,
    algorithm: str,
    feature_selection_method: str,
    output_dir: str = ".",
):
    """
    Count the final number of features retained by all preprocessing
    and feature-selection steps, and save a text summary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_features = X.shape[1]

    selected_mask = get_global_feature_selection_mask(
        pipeline=pipeline,
        X=X,
    )
    selected_features = int(selected_mask.sum())

    message = (
        f"[INFO] Final selected features: "
        f"{selected_features} / {total_features}"
    )
    print(message)

    output_file = output_path / (
        f"{algorithm}_{feature_selection_method}_features_count.txt"
    ).replace(" ", "_")

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f"Algorithm: {algorithm}\n")
        file.write(
            f"Feature selection method: "
            f"{feature_selection_method}\n"
        )
        file.write(
            f"Final selected features: "
            f"{selected_features} / {total_features}\n"
        )

    return selected_features, total_features


def export_predictions(
    pipeline,
    X_test,
    y_test,
    algorithm: str,
    feature_selection_method: str,
    output_dir: str = ".",
    test_metadata=None,
):
    """
    Generate test predictions and export them to CSV.

    When test_metadata is provided, the original peptide identifier
    and sequence are included in the exported prediction table.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    y_pred = pipeline.predict(X_test)
    y_probabilities = None

    if hasattr(pipeline, "predict_proba"):
        try:
            y_probabilities = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_probabilities = None

    prediction_data = {}

    if test_metadata is not None:
        if not X_test.index.isin(test_metadata.index).all():
            raise ValueError(
                "The test metadata index does not match the X_test index."
            )

        aligned_metadata = test_metadata.reindex(X_test.index)

        if "#" in aligned_metadata.columns:
            prediction_data["#"] = aligned_metadata["#"].to_numpy()

        if "sequence" in aligned_metadata.columns:
            prediction_data["sequence"] = (
                aligned_metadata["sequence"].to_numpy()
            )

    prediction_data["test_row_index"] = np.asarray(X_test.index)
    prediction_data["y_true"] = np.asarray(y_test)
    prediction_data["y_pred"] = np.asarray(y_pred)

    prediction_data["positive_class_probability"] = (
        y_probabilities
        if y_probabilities is not None
        else np.full(len(y_pred), np.nan)
    )

    predictions_df = pd.DataFrame(prediction_data)

    output_file = output_path / (
        f"{algorithm}_{feature_selection_method}_predictions.csv"
    ).replace(" ", "_")

    predictions_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    print(f"[INFO] Predictions saved to: {output_file}")

    return y_pred, y_probabilities


def get_feature_names(X) -> list[str]:
    """
    Return feature names from a DataFrame or generate generic names for arrays.
    """
    if isinstance(X, pd.DataFrame):
        return list(X.columns)

    return [f"feature_{index}" for index in range(X.shape[1])]


def get_coefficients_or_importances(estimator):
    """
    Return feature importances or coefficients when available.
    """
    if estimator is None:
        return None

    if hasattr(estimator, "feature_importances_"):
        return getattr(estimator, "feature_importances_")

    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(getattr(estimator, "coef_"))
        if coefficients.ndim == 1:
            return coefficients
        return np.sum(np.abs(coefficients), axis=0)

    return None


def get_indices_before_step(pipeline, X, stop_step_name: Optional[str] = None) -> list[int]:
    """
    Track original feature indices that remain before a given pipeline step.

    If stop_step_name is None, the function processes all steps and returns the
    original indices that remain at the end of the pipeline preprocessing and
    feature-selection steps. Estimator steps without get_support() are ignored.
    """
    current_indices = list(range(X.shape[1]))

    for step_name, step_object in pipeline.steps:
        if stop_step_name is not None and step_name == stop_step_name:
            break

        if hasattr(step_object, "get_support"):
            support_mask = np.asarray(step_object.get_support()).astype(bool)
            if len(support_mask) == len(current_indices):
                current_indices = [
                    index for index, keep in zip(current_indices, support_mask) if keep
                ]

    return current_indices


def get_indices_before_feature_selection(pipeline, X) -> list[int]:
    """
    Track which original feature indices remain immediately before the 'fs' step.
    """
    return get_indices_before_step(pipeline, X, stop_step_name="fs")


def expand_step_values_to_original_features(values, current_indices, total_features):
    """
    Expand selector-level values to the original feature space.
    """
    expanded_values = np.full(total_features, np.nan, dtype=object)

    if values is None:
        return expanded_values.tolist()

    values = np.asarray(values, dtype=object)
    usable_length = min(len(values), len(current_indices))

    for local_index in range(usable_length):
        original_index = current_indices[local_index]
        expanded_values[original_index] = values[local_index]

    return expanded_values.tolist()


def get_global_feature_selection_mask(pipeline, X) -> np.ndarray:
    """
    Return a boolean mask of selected features in the original feature space.

    The mask includes unsupervised prefilters such as VarianceThreshold and
    CorrelationFilter, and supervised selectors in the 'fs' step when present.
    """
    total_features = X.shape[1]
    selected_indices = get_indices_before_step(pipeline, X, stop_step_name=None)

    global_mask = np.zeros(total_features, dtype=bool)
    for original_index in selected_indices:
        global_mask[original_index] = True

    return global_mask


def export_selected_features(
    pipeline,
    X,
    algorithm: str,
    feature_selection_method: str,
    output_dir: str = ".",
):
    """
    Export feature-selection metadata for all original input features.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_names = get_feature_names(X)
    total_features = len(feature_names)
    selected_mask = get_global_feature_selection_mask(pipeline, X)

    scores = [np.nan] * total_features
    pvalues = [np.nan] * total_features
    rankings = [np.nan] * total_features
    importances = [np.nan] * total_features

    if "fs" in pipeline.named_steps:
        feature_selector = pipeline.named_steps["fs"]
        current_indices = get_indices_before_feature_selection(pipeline, X)

        scores = expand_step_values_to_original_features(
            getattr(feature_selector, "scores_", None), current_indices, total_features
        )
        pvalues = expand_step_values_to_original_features(
            getattr(feature_selector, "pvalues_", None), current_indices, total_features
        )
        rankings = expand_step_values_to_original_features(
            getattr(feature_selector, "ranking_", None), current_indices, total_features
        )

        internal_estimator = getattr(feature_selector, "estimator_", None) or getattr(
            feature_selector, "estimator", None
        )
        importances = expand_step_values_to_original_features(
            get_coefficients_or_importances(internal_estimator), current_indices, total_features
        )
    else:
        rankings = [1] * total_features

    selected_features_df = pd.DataFrame(
        {
            "feature": feature_names,
            "selected": selected_mask.tolist(),
            "score": scores,
            "pvalue": pvalues,
            "ranking": rankings,
            "importance": importances,
        }
    )

    output_file = output_path / f"{algorithm}_{feature_selection_method}_selected_features.csv".replace(" ", "_")
    selected_features_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"[INFO] Selected-feature metadata saved to: {output_file}")

    return output_file


def export_selected_feature_names(
    pipeline,
    X,
    algorithm: str,
    feature_selection_method: str,
    output_dir: str = ".",
):
    """
    Export only the names of the selected features.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_names = get_feature_names(X)
    selected_mask = get_global_feature_selection_mask(pipeline, X)
    selected_names = [name for name, selected in zip(feature_names, selected_mask) if selected]

    output_file = output_path / f"{algorithm}_{feature_selection_method}_selected_feature_names.txt".replace(" ", "_")
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(selected_names))

    print(f"[INFO] Selected feature names saved to: {output_file}")
    return output_file


def jsonify_parameters(parameters: dict) -> dict:
    """
    Convert scikit-learn parameters to JSON-safe values.
    """

    def convert(value):
        if callable(value):
            return getattr(value, "__name__", str(value))
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, (np.ndarray, list, tuple, set)):
            return [convert(item) for item in list(value)]
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    return {key: convert(value) for key, value in parameters.items()}


def parse_feature_selection_parameter_grid(feature_selection_param_grid_extra):
    """
    Parse optional feature-selection parameters from a dictionary or string.
    """
    if feature_selection_param_grid_extra is None:
        return {}

    if isinstance(feature_selection_param_grid_extra, str):
        raw_value = feature_selection_param_grid_extra.strip()
        try:
            feature_selection_param_grid_extra = json.loads(raw_value)
        except Exception:
            feature_selection_param_grid_extra = ast.literal_eval(raw_value)

    if not isinstance(feature_selection_param_grid_extra, dict):
        raise ValueError(
            "feature_selection_param_grid_extra must be a dictionary or a string representing a dictionary."
        )

    score_function_mapping = {
        "f_classif": f_classif,
        "mutual_info_classif": mutual_info_classif,
        "chi2": chi2,
    }

    def map_value(value):
        if isinstance(value, str) and value in score_function_mapping:
            return score_function_mapping[value]
        if isinstance(value, list):
            return [map_value(item) for item in value]
        return value

    return {key: map_value(value) for key, value in feature_selection_param_grid_extra.items()}


def build_feature_selection_step(feature_selection_method: str):
    """
    Build the supervised/model-based feature-selection step and its default grid.

    The 'none' and 'prefilter' methods do not add a supervised selector.
    Prefiltering is handled separately in build_training_pipeline().
    """
    method = (feature_selection_method or "none").lower()

    if method in ("none", "prefilter"):
        return None, None, {}

    if method == "percentile":
        selector = SelectPercentile(score_func=f_classif, percentile=10)
        grid = {
            "fs__percentile": FEATURE_PERCENTILES,
            "fs__score_func": [f_classif, mutual_info_classif],
        }
        return None, ("fs", selector), grid

    if method == "percentile_chi2":
        pre_feature_selection_step = ("pre_fs", MinMaxScaler())
        selector = SelectPercentile(score_func=chi2, percentile=10)
        grid = {"fs__percentile": FEATURE_PERCENTILES}
        return pre_feature_selection_step, ("fs", selector), grid

    if method == "rfe":
        selector = RFE(
            estimator=RandomForestClassifier(
                n_estimators=50,
                random_state=SEED,
                n_jobs=1,
            ),
            n_features_to_select=0.10,
            step=0.10,
        )
        grid = {
            "fs__n_features_to_select": [0.05, 0.10, 0.20, 0.40],
            "fs__step": [0.10, 0.20],
        }
        return None, ("fs", selector), grid

    if method == "rfecv":
        rfecv_cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=SEED,
        )
        selector = RFECV(
            estimator=RandomForestClassifier(
                n_estimators=100,
                random_state=SEED,
                n_jobs=1,
            ),
            step=0.05,
            min_features_to_select=50,
            cv=rfecv_cv,
            scoring="f1",
            n_jobs=1,
        )
        grid = {"fs__step": [0.05, 0.10]}
        return None, ("fs", selector), grid

    if method == "sfm_l1":
        selector = SelectFromModel(
            LinearSVC(
                penalty="l1",
                dual=False,
                C=0.5,
                random_state=SEED,
                max_iter=100000,
            )
        )
        grid = {
            "fs__max_features": SFM_MAX_FEATURES_GRID,
            "fs__estimator__C": [0.1, 0.5, 1.0],
        }
        return None, ("fs", selector), grid

    if method == "sfm_rf":
        selector = SelectFromModel(
            RandomForestClassifier(
                n_estimators=300,
                random_state=SEED,
                n_jobs=1,
            )
        )
        grid = {
            "fs__max_features": SFM_MAX_FEATURES_GRID,
            "fs__estimator__max_depth": [None, 10, 20],
        }
        return None, ("fs", selector), grid

    if method == "sfm_xgb":
        selector = SelectFromModel(
            XGBClassifier(
                eval_metric="logloss",
                n_estimators=200,
                learning_rate=0.10,
                random_state=SEED,
                n_jobs=1,
                device="cpu",
            )
        )
        grid = [
            {
                "fs__max_features": SFM_MAX_FEATURES_GRID,
                "fs__estimator__n_estimators": [200],
                "fs__estimator__learning_rate": [0.10],
                "fs__estimator__max_depth": [3, 6],
            },
            {
                "fs__max_features": SFM_MAX_FEATURES_GRID,
                "fs__estimator__n_estimators": [400],
                "fs__estimator__learning_rate": [0.03],
                "fs__estimator__max_depth": [3, 6],
            },
        ]
        return None, ("fs", selector), grid

    raise ValueError(f"Unsupported feature selection method: {feature_selection_method}")


def get_base_model_and_grid(algorithm: str):
    """
    Return the base estimator and its hyperparameter grid.
    """
    models = {
        "SVM": (
            SVC(probability=False, random_state=SEED),
            [
                {
                    "svc__C": [0.1, 1, 10],
                    "svc__kernel": ["linear"],
                },
                {
                    "svc__C": [0.1, 1, 10],
                    "svc__kernel": ["rbf"],
                    "svc__gamma": ["scale", "auto"],
                },
            ],
        ),
        "RF": (
            RandomForestClassifier(random_state=SEED, n_jobs=1),
            {
                "randomforestclassifier__n_estimators": [200, 400],
                "randomforestclassifier__max_depth": [None, 10, 20],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "kneighborsclassifier__n_neighbors": [3, 5, 7],
                "kneighborsclassifier__weights": ["uniform", "distance"],
            },
        ),
        "NB": (GaussianNB(), {}),
        "MLP": (
            MLPClassifier(max_iter=500, random_state=SEED),
            {
                "mlpclassifier__hidden_layer_sizes": [(100,), (150,), (100, 50)],
                "mlpclassifier__alpha": [0.0001, 0.001],
                "mlpclassifier__solver": ["adam", "lbfgs"],
            },
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1, device="cpu"),
            {
                "xgbclassifier__n_estimators": [200, 400],
                "xgbclassifier__learning_rate": [0.03, 0.1],
                "xgbclassifier__max_depth": [3, 6],
            },
        ),
    }

    if algorithm not in models:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return models[algorithm]


def prepare_rfe_selector_for_algorithm(feature_selection_step, algorithm: str, base_model):
    """
    Replace the internal RFE/RFECV estimator when the final model has no importances.
    """
    if feature_selection_step is None:
        return None

    step_name, selector = feature_selection_step

    if not isinstance(selector, (RFE, RFECV)):
        return feature_selection_step

    updated_selector = clone(selector)

    if algorithm in ("SVM", "MLP", "KNN", "NB"):
        embedded_estimator = LinearSVC(C=1.0, random_state=SEED, dual=True, max_iter=100000)
    else:
        embedded_estimator = base_model

    updated_selector.set_params(estimator=embedded_estimator)
    return step_name, updated_selector


def build_training_pipeline(algorithm: str, feature_selection_method: str):
    """
    Build the complete preprocessing, feature-selection, and estimator pipeline.

    Feature-selection method semantics:
    - none: no prefilter and no feature-selection step.
    - prefilter: low-variance filter + correlation filter only.
    - all other methods: prefilter + selected supervised/model-based selector.

    StandardScaler is added only for algorithms/selectors that need scaling.
    """
    method = (feature_selection_method or "none").lower()

    base_model, model_param_grid = get_base_model_and_grid(algorithm)
    model_step_name = base_model.__class__.__name__.lower()

    pre_feature_selection_step, feature_selection_step, feature_selection_grid = build_feature_selection_step(
        method
    )
    feature_selection_step = prepare_rfe_selector_for_algorithm(
        feature_selection_step, algorithm, base_model
    )

    steps = [("imputer", SimpleImputer(strategy="mean"))]

    use_prefilter = method != "none"
    if use_prefilter:
        steps.append(("low_variance_filter", VarianceThreshold(threshold=0.0)))
        steps.append(("correlation_filter", CorrelationFilter(threshold=0.95)))

    if pre_feature_selection_step is not None:
        steps.append(pre_feature_selection_step)

    needs_linear_embedded_scaler = (
        algorithm in ("SVM", "MLP", "KNN", "NB")
        and feature_selection_step is not None
        and isinstance(feature_selection_step[1], (RFE, RFECV))
    )
    add_scaler_before_feature_selection = (
        needs_linear_embedded_scaler or method == "sfm_l1"
    )

    if add_scaler_before_feature_selection:
        steps.append(("scaler", StandardScaler()))

    if feature_selection_step is not None:
        steps.append(feature_selection_step)

    model_needs_scaler = algorithm in ("SVM", "KNN", "MLP")

    if model_needs_scaler and not add_scaler_before_feature_selection:
        steps.append(("scaler", StandardScaler()))

    if algorithm == "SVM":
        steps.append(("svc", base_model))
    else:
        steps.append((model_step_name, base_model))

    pipeline = Pipeline(steps)
    return pipeline, model_param_grid, feature_selection_grid, feature_selection_step


def filter_feature_selection_grid(feature_selection_grid, pipeline, feature_selection_step):
    """
    Remove feature-selection parameters that are not valid for the current pipeline.
    """
    valid_parameters = set(pipeline.get_params().keys())

    def filter_single_grid(single_grid):
        filtered_grid = {
            key: value
            for key, value in single_grid.items()
            if key in valid_parameters
        }

        if feature_selection_step and isinstance(
            feature_selection_step[1],
            (RFE, RFECV),
        ):
            filtered_grid = {
                key: value
                for key, value in filtered_grid.items()
                if not key.startswith("fs__estimator__")
            }

        return filtered_grid

    if isinstance(feature_selection_grid, list):
        return [
            filter_single_grid(single_grid)
            for single_grid in feature_selection_grid
        ]

    return filter_single_grid(feature_selection_grid)


def export_best_parameters(best_parameters, algorithm: str, feature_selection_method: str, output_dir: str = "."):
    """
    Save the best hyperparameters from the search object as JSON.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_parameters_json = jsonify_parameters(best_parameters)
    output_file = output_path / f"{algorithm}_{feature_selection_method}_best_parameters.json".replace(" ", "_")

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(best_parameters_json, json_file, indent=4)

    print(f"[OK] Best parameters saved to: {output_file}")
    return output_file


def export_training_log(algorithm: str, feature_selection_method: str, output_dir: str = "."):
    """
    Save a lightweight training log for the current experiment.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"{algorithm}_{feature_selection_method}_log.txt".replace(" ", "_")

    with open(output_file, "w", encoding="utf-8") as log_file:
        log_file.write(f"Algorithm: {algorithm}\n")
        log_file.write(f"Feature selection method: {feature_selection_method}\n")
        log_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    return output_file


def tune_model(
    algorithm,
    X_train,
    y_train,
    search_mode="grid",
    cv=5,
    use_probabilities=True,
    feature_selection_method: str = "none",
    feature_selection_param_grid_extra: Optional[Dict[str, Any]] = None,
):
    """
    Tune and fit a full ML pipeline using only the training set.
    """
    pipeline, model_param_grid, feature_selection_grid, feature_selection_step = build_training_pipeline(
        algorithm=algorithm,
        feature_selection_method=feature_selection_method,
    )

    if feature_selection_param_grid_extra:
        parsed_grid = parse_feature_selection_parameter_grid(
            feature_selection_param_grid_extra
        )
        if isinstance(feature_selection_grid, list):
            feature_selection_grid = [
                {**single_grid, **parsed_grid}
                for single_grid in feature_selection_grid
            ]
        else:
            feature_selection_grid.update(parsed_grid)

    feature_selection_grid = filter_feature_selection_grid(
        feature_selection_grid,
        pipeline,
        feature_selection_step,
    )

    model_grids = (
        model_param_grid
        if isinstance(model_param_grid, list)
        else [model_param_grid]
    )
    feature_selection_grids = (
        feature_selection_grid
        if isinstance(feature_selection_grid, list)
        else [feature_selection_grid]
    )

    combined_grids = [
        {**model_grid, **feature_selection_subgrid}
        for model_grid in model_grids
        for feature_selection_subgrid in feature_selection_grids
    ]
    parameter_grid = (
        combined_grids[0]
        if len(combined_grids) == 1
        else combined_grids
    )

    stratified_cv = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=SEED,
    )

    print(f"[INFO] Feature selection: {feature_selection_method} | Search mode: {search_mode} | CV: {cv}")

    if search_mode == "grid":
        total_combinations = len(list(ParameterGrid(parameter_grid))) if parameter_grid else 0
        print(
            f"[INFO] Total combinations to evaluate: {total_combinations} × {cv}-fold = {total_combinations * cv}"
        )

    if not parameter_grid:
        print(f"[INFO] Algorithm '{algorithm}' has no configured hyperparameter grid. Fitting directly.")
        pipeline.fit(X_train, y_train)
        export_selected_features(pipeline, X_train, algorithm, feature_selection_method, output_dir=".")
        export_selected_feature_names(pipeline, X_train, algorithm, feature_selection_method, output_dir=".")
        return pipeline

    print(f"[INFO] Tuning algorithm '{algorithm}' with {search_mode} search...")
    n_jobs = get_parallel_jobs()

    if search_mode == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="f1",
            n_jobs=n_jobs,
            verbose=1,
            cv=stratified_cv,
        )
    elif search_mode == "random":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameter_grid,
            scoring="f1",
            n_jobs=n_jobs,
            verbose=1,
            cv=stratified_cv,
            random_state=SEED,
        )
    else:
        raise ValueError(f"Unsupported search mode: {search_mode}")

    print(f"[INFO] GridSearchCV parallel jobs: {n_jobs}", flush=True)
    print("[INFO] Starting cross-validation search and final refit...", flush=True)
    
    search.fit(X_train, y_train)
    print(f"[INFO] Automatic refit time: {search.refit_time_:.2f} seconds", flush=True)
    print("[INFO] Cross-validation search and automatic refit completed.", flush=True)
    print(f"[INFO] Best configuration: {search.best_params_}")

    export_training_log(algorithm, feature_selection_method, output_dir=".")
    export_best_parameters(search.best_params_, algorithm, feature_selection_method, output_dir=".")

    best_pipeline = search.best_estimator_

    try:
        export_selected_features(best_pipeline, X_train, algorithm, feature_selection_method, output_dir=".")
        export_selected_feature_names(best_pipeline, X_train, algorithm, feature_selection_method, output_dir=".")
    except Exception as exc:
        print(f"[WARN] Could not export selected features from the best pipeline: {exc}")

    if algorithm == "SVM" and use_probabilities:
        print("[INFO] Refitting the best SVM pipeline with probability=True.", flush=True)
        final_pipeline = clone(best_pipeline)
        final_pipeline.set_params(svc__probability=True)
        final_pipeline.fit(X_train, y_train)
        print("[INFO] Final SVM refit completed.", flush=True)
        return final_pipeline

    return best_pipeline
