import argparse
import ast
import json
import sys
from pathlib import Path

from seed_utils import set_global_seed
from data_preparation import load_descriptors, prepare_train_test_data
from metrics_plots import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)
from reports import save_excel_report, save_pdf_report
from model_tuning import (
    tune_model,
    export_predictions,
    get_feature_selection_stats_and_export,
)
from interpretability import run_interpretability_analysis


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
SEARCH_MODES = ["grid", "random"]
INTERPRETABILITY_MODES = ["none", "permutation", "shap", "all"]
REPRESENTATIONS = ["ESM-2", "iLearn"]
PERMUTATION_SCORING_OPTIONS = ["average_precision", "mcc", "f1"]
CLASS_NAMES = ["Negative", "Positive"]


def validate_input_files(train_file, test_file):
    """
    Validate that both external train and test CSV files were provided and exist.
    """
    if not train_file or not test_file:
        print("[ERROR] Both --train_file and --test_file are required.")
        sys.exit(1)

    train_path = Path(train_file)
    test_path = Path(test_file)

    if not train_path.is_file():
        print(f"[ERROR] Training file not found: {train_path}")
        sys.exit(1)

    if not test_path.is_file():
        print(f"[ERROR] Test file not found: {test_path}")
        sys.exit(1)

    return train_path, test_path


def parse_feature_selection_params(raw_params):
    """
    Parse optional feature-selection parameters passed as JSON or Python literal syntax.
    """
    if raw_params is None:
        return None

    try:
        return json.loads(raw_params)
    except Exception:
        try:
            return ast.literal_eval(raw_params)
        except Exception as exc:
            raise ValueError(
                "Invalid --feature_selection_params format. Use JSON or a valid Python dictionary literal."
            ) from exc


def load_train_test_datasets(train_file, test_file):
    """
    Load external training and test descriptor datasets.
    """
    print("[INFO] Loading descriptor datasets...")
    train_df = load_descriptors(train_file)
    test_df = load_descriptors(test_file)
    return train_df, test_df


def prepare_model_data(train_df, test_df):
    """
    Build X_train, X_test, y_train, and y_test from precomputed
    train/test datasets.

    Test metadata are preserved separately so that predictions can
    later be linked to the original peptide identifiers and sequences.
    """
    metadata_columns = [
        column
        for column in ["#", "sequence"]
        if column in test_df.columns
    ]

    test_metadata = test_df[metadata_columns].copy()

    X_train, X_test, y_train, y_test = prepare_train_test_data(
        train_df,
        test_df,
    )

    # Preserve the same row order and index used by X_test.
    test_metadata = test_metadata.reindex(X_test.index)

    print("[INFO] Data shapes:")
    print(f"       X_train: {X_train.shape}")
    print(f"       X_test : {X_test.shape}")
    print(f"       y_train: {y_train.shape}")
    print(f"       y_test : {y_test.shape}")

    print("[INFO] Test metadata columns:", list(test_metadata.columns))

    return X_train, X_test, y_train, y_test, test_metadata


def fit_best_pipeline(args, X_train, y_train, feature_selection_params):
    """
    Tune and fit the full machine-learning pipeline using only the training set.

    IMPORTANT FOR ROUND 4A:
    The GridSearchCV scoring criterion remains unchanged from Round 2 (F1 inside
    model_tuning.py). Average Precision is used as the default scoring metric for
    permutation importance in Round 4A.
    """
    print(f"[INFO] Algorithm: {args.algorithm}")
    print(
        f"[INFO] Feature selection method: "
        f"{args.feature_selection_method}"
    )
    print(f"[INFO] Search mode: {args.search_mode}")
    print(f"[INFO] Cross-validation folds: {args.cv}")
    print(f"[INFO] Representation: {args.representation}")

    model = tune_model(
        algorithm=args.algorithm,
        X_train=X_train,
        y_train=y_train,
        search_mode=args.search_mode,
        cv=args.cv,
        use_probabilities=True,
        feature_selection_method=args.feature_selection_method,
        feature_selection_param_grid_extra=feature_selection_params,
    )

    print(f"[DEBUG] Fitted pipeline type: {type(model)}")
    return model


def evaluate_pipeline(
    model,
    X_train,
    X_test,
    y_test,
    test_metadata,
    algorithm,
    feature_selection_method,
):
    """
    Evaluate the fitted pipeline on the external test set and generate base metrics.
    """
    selected_count, total_count = get_feature_selection_stats_and_export(
        pipeline=model,
        X=X_train,
        algorithm=algorithm,
        feature_selection_method=feature_selection_method,
        output_dir=".",
    )

    y_pred, y_probabilities = export_predictions(
        pipeline=model,
        X_test=X_test,
        y_test=y_test,
        test_metadata=test_metadata,
        algorithm=algorithm,
        feature_selection_method=feature_selection_method,
        output_dir=".",
    )

    metrics = evaluate_model(y_test, y_pred)
    metrics["Selected features"] = f"{selected_count} / {total_count}"

    return metrics, y_probabilities


def add_roc_metrics(
    metrics,
    y_test,
    y_probabilities,
    algorithm,
    feature_selection_method,
):
    """
    Generate the ROC curve and add AUC when class probabilities are available.
    """
    if y_probabilities is None:
        print(
            "[WARN] The model does not provide predict_proba(); "
            "ROC curve was not generated."
        )
        return metrics

    roc_png = f"{algorithm}_{feature_selection_method}_roc_curve.png"
    auc_score = plot_roc_curve(
        y_true=y_test,
        y_scores=y_probabilities,
        output_file=roc_png,
    )
    metrics["AUC"] = auc_score

    return metrics


def add_precision_recall_metrics(
    metrics,
    model,
    X_test,
    y_test,
    y_probabilities,
    algorithm,
    feature_selection_method,
):
    """
    Generate the Precision-Recall curve using probabilities or decision scores.
    """
    y_scores = None

    if y_probabilities is not None:
        y_scores = y_probabilities
    elif hasattr(model, "decision_function"):
        try:
            y_scores = model.decision_function(X_test)
        except Exception:
            y_scores = None

    if y_scores is None:
        print(
            "[WARN] No continuous scores available for the Precision-Recall curve."
        )
        return metrics

    precision_recall_png = (
        f"{algorithm}_{feature_selection_method}_precision_recall_curve.png"
    )
    precision_recall_csv = (
        f"{algorithm}_{feature_selection_method}_precision_recall_curve.csv"
    )

    average_precision, precision_recall_auc = plot_precision_recall_curve(
        y_true=y_test,
        y_scores=y_scores,
        output_png=precision_recall_png,
        output_csv=precision_recall_csv,
    )

    metrics["AP"] = average_precision
    metrics["PR_AUC"] = precision_recall_auc

    return metrics


def add_experiment_metadata(
    metrics,
    train_file,
    test_file,
    algorithm,
    feature_selection_method,
):
    """
    Add experiment metadata to the metrics dictionary.

    These values are included in the Excel and PDF reports.
    """
    dataset_label = f"{Path(train_file).stem}__{Path(test_file).stem}"

    metrics_without_metadata = {
        key: value
        for key, value in metrics.items()
        if key not in {
            "Dataset",
            "Algorithm",
            "Feature selection method",
        }
    }

    return {
        "Dataset": dataset_label,
        "Algorithm": algorithm,
        "Feature selection method": feature_selection_method,
        **metrics_without_metadata,
    }


def save_outputs(metrics, algorithm, feature_selection_method):
    """
    Save the confusion matrix, Excel report, and PDF report for the current experiment.
    """
    confusion_matrix_png = (
        f"{algorithm}_{feature_selection_method}_confusion_matrix.png"
    )
    excel_report = f"{algorithm}_{feature_selection_method}_model_report.xlsx"
    pdf_report = f"{algorithm}_{feature_selection_method}_model_report.pdf"

    plot_confusion_matrix(
        confusion_matrix_values=metrics["Confusion Matrix"],
        class_labels=CLASS_NAMES,
        output_file=confusion_matrix_png,
    )

    print("[INFO] Saving reports...")
    save_excel_report(
        metrics,
        excel_report,
        f"{algorithm}_{feature_selection_method}",
    )
    save_pdf_report(metrics, pdf_report)

    print("[OK] Experiment completed.")
    print(f"     Excel report     : {excel_report}")
    print(f"     PDF report       : {pdf_report}")
    print(f"     Confusion matrix : {confusion_matrix_png}")


def run_experiment(args):
    """
    Run a complete supervised ML experiment using external train/test files.
    """
    set_global_seed()

    train_file, test_file = validate_input_files(
        args.train_file,
        args.test_file,
    )
    feature_selection_params = parse_feature_selection_params(
        args.feature_selection_params
    )

    train_df, test_df = load_train_test_datasets(train_file, test_file)
    X_train, X_test, y_train, y_test, test_metadata = prepare_model_data(
        train_df,
        test_df,
    )

    model = fit_best_pipeline(
        args,
        X_train,
        y_train,
        feature_selection_params,
    )

    metrics, y_probabilities = evaluate_pipeline(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        test_metadata=test_metadata,
        algorithm=args.algorithm,
        feature_selection_method=args.feature_selection_method,
    )

    metrics = add_roc_metrics(
        metrics=metrics,
        y_test=y_test,
        y_probabilities=y_probabilities,
        algorithm=args.algorithm,
        feature_selection_method=args.feature_selection_method,
    )

    metrics = add_precision_recall_metrics(
        metrics=metrics,
        model=model,
        X_test=X_test,
        y_test=y_test,
        y_probabilities=y_probabilities,
        algorithm=args.algorithm,
        feature_selection_method=args.feature_selection_method,
    )

    metrics = add_experiment_metadata(
        metrics=metrics,
        train_file=train_file,
        test_file=test_file,
        algorithm=args.algorithm,
        feature_selection_method=args.feature_selection_method,
    )

    save_outputs(
        metrics=metrics,
        algorithm=args.algorithm,
        feature_selection_method=args.feature_selection_method,
    )

    run_interpretability_analysis(
        pipeline=model,
        X_test=X_test,
        y_test=y_test,
        algorithm=args.algorithm,
        feature_selection_method=args.feature_selection_method,
        representation=args.representation,
        mode=args.interpretability,
        output_dir=".",
        top_n=args.top_features,
        permutation_repeats=args.permutation_repeats,
        permutation_scoring=args.permutation_scoring,
    )


def parse_arguments():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a peptide classification model using external "
            "train/test CSV files."
        )
    )

    parser.add_argument(
        "--train_file",
        required=True,
        help="Training CSV file with descriptors.",
    )
    parser.add_argument(
        "--test_file",
        required=True,
        help="Test CSV file with descriptors.",
    )
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=ALGORITHMS,
        help="Machine learning algorithm.",
    )
    parser.add_argument(
        "--feature_selection_method",
        default="none",
        choices=FEATURE_SELECTION_METHODS,
        help="Feature selection method.",
    )
    parser.add_argument(
        "--feature_selection_params",
        default=None,
        help=(
            "Optional JSON/dict with feature-selection hyperparameters, "
            'e.g. {"fs__k": [50, 100]}'
        ),
    )
    parser.add_argument(
        "--search_mode",
        default="grid",
        choices=SEARCH_MODES,
        help="Hyperparameter search strategy.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--representation",
        required=True,
        choices=REPRESENTATIONS,
        help=(
            "Descriptor representation used by the input files: ESM-2 or iLearn. "
            "This is required so iLearn family-level interpretability can be "
            "enabled safely."
        ),
    )
    parser.add_argument(
        "--interpretability",
        default="none",
        choices=INTERPRETABILITY_MODES,
        help=(
            "Interpretability analysis to run: none, permutation, shap, or all."
        ),
    )
    parser.add_argument(
        "--top_features",
        type=int,
        default=20,
        help=(
            "Number of top features to export and plot for interpretability analyses."
        ),
    )
    parser.add_argument(
        "--permutation_repeats",
        type=int,
        default=10,
        help="Number of repeats for permutation importance.",
    )
    parser.add_argument(
        "--permutation_scoring",
        default="average_precision",
        choices=PERMUTATION_SCORING_OPTIONS,
        help=(
            "Metric used to quantify permutation importance. "
            "Default for Round 4A: average_precision. "
            "This does not change GridSearchCV scoring."
        ),
    )

    return parser.parse_args()


def main():
    """Entry point of the script."""
    args = parse_arguments()
    run_experiment(args)


if __name__ == "__main__":
    main()
