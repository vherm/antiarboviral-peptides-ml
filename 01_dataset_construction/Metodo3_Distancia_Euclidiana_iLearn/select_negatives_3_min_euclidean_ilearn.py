"""
Select negative peptides using GROUP-NORMALIZED Euclidean distance over
preprocessed iLearn features.

This is a controlled modification of the original Min-Euclidean selector.
Everything is kept unchanged except the distance definition:

Original Euclidean distance
---------------------------
    D^2(x, y) = sum_j (x_j - y_j)^2

Group-normalized Euclidean distance
-----------------------------------
For every iLearn descriptor family g with p_g retained features:

    D_g^2(x, y) = (1 / p_g) * sum_{j in g} (x_j - y_j)^2

and the global distance is

    D^2(x, y) = sum_g D_g^2(x, y)

Equivalently, every retained feature in family g is multiplied by
1 / sqrt(p_g) before applying the ordinary Euclidean norm.

Consequences:
- StandardScaler still gives every individual feature a comparable scale.
- Dividing each family's squared contribution by p_g prevents a family from
  dominating merely because it contains many more features.
- Each descriptor family has the same total structural weight (= 1) before
  the actual positive-negative differences are considered.

The following parts of the original method are intentionally unchanged:
- all positives and candidate negatives are processed together;
- constant features are removed;
- features are standardized with StandardScaler;
- features with |Pearson correlation| >= 0.95 are greedily removed;
- candidate negatives are restricted to the same +/-10% length window;
- 5 negatives are selected per positive;
- selected negatives cannot be reused;
- positives are processed greedily in their input order.

A family report is automatically written next to the output dataset:
    <output_stem>_family_report.csv

Usage
-----
    python select_negatives_3_group_normalized_euclidean_ilearn.py \
        positives.csv negatives.csv Dataset_Min_Euclidean_GroupNormalized.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

N_NEGATIVES_PER_POSITIVE = 5

CORRELATION_THRESHOLD = 0.95
CORRELATION_BLOCK_SIZE = 256

LENGTH_WINDOW_FRACTION = 0.10
MINIMUM_LENGTH_DIFFERENCE = 1

METADATA_COLUMNS = {"#", "sequence", "label"}

# Known iLearn descriptor families used in this project.
# Matching is conservative and more-specific names are checked first.
KNOWN_FAMILIES = [
    "SOCNumber",
    "CKSAAGP",
    "CKSAAP",
    "KSCTriad",
    "QSOrder",
    "NMBroto",
    "CTRIAD",
    "APAAC",
    "Geary",
    "Moran",
    "CTDC",
    "CTDD",
    "CTDT",
    "GDPC",
    "GTPC",
    "PAAC",
    "AAC",
    "DDE",
    "DPC",
    "GAAC",
    "TPC",
]


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments():
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Select negative peptides using group-normalized Euclidean "
            "distance over preprocessed iLearn features and a relative "
            "sequence-length window."
        )
    )

    parser.add_argument(
        "positive_csv",
        help=(
            "CSV file containing positive peptides "
            "and iLearn features."
        ),
    )

    parser.add_argument(
        "negative_csv",
        help=(
            "CSV file containing candidate negative peptides "
            "and iLearn features."
        ),
    )

    parser.add_argument(
        "output_csv",
        help="CSV file where the resulting dataset will be saved.",
    )

    return parser.parse_args()


# ============================================================
# Length window
# ============================================================

def compute_maximum_length_difference(sequence_length):
    """
    Calculate the maximum allowed length difference.

    The window corresponds to 10% of the positive peptide length,
    using conventional rounding and a minimum difference of one residue.

    Examples
    --------
    Length 5  -> maximum difference 1
    Length 10 -> maximum difference 1
    Length 20 -> maximum difference 2
    Length 30 -> maximum difference 3
    Length 40 -> maximum difference 4
    Length 50 -> maximum difference 5
    """
    relative_difference = int(
        LENGTH_WINDOW_FRACTION * sequence_length + 0.5
    )

    return max(
        MINIMUM_LENGTH_DIFFERENCE,
        relative_difference,
    )


# ============================================================
# Correlation filter
# ============================================================

def remove_highly_correlated_features(
    X_scaled,
    threshold=CORRELATION_THRESHOLD,
    block_size=CORRELATION_BLOCK_SIZE,
):
    """
    Apply the same greedy Pearson correlation filter as the original script.

    Features are evaluated in their original column order. A feature is
    removed when its absolute Pearson correlation with any previously retained
    feature is greater than or equal to the specified threshold.
    """
    n_samples, n_features = X_scaled.shape

    dropped = np.zeros(
        n_features,
        dtype=bool,
    )

    kept_indices = []

    for start in range(0, n_features, block_size):
        end = min(
            start + block_size,
            n_features,
        )

        current_indices = np.arange(
            start,
            end,
        )

        current_indices = current_indices[
            ~dropped[current_indices]
        ]

        if current_indices.size == 0:
            continue

        current_matrix = X_scaled[
            :,
            current_indices,
        ]

        correlations_within_block = np.abs(
            current_matrix.T @ current_matrix
        ) / n_samples

        accepted_local_positions = []

        for local_position, feature_index in enumerate(
            current_indices
        ):
            if accepted_local_positions:
                previous_correlations = (
                    correlations_within_block[
                        accepted_local_positions,
                        local_position,
                    ]
                )

                if np.any(
                    previous_correlations >= threshold
                ):
                    dropped[feature_index] = True
                    continue

            accepted_local_positions.append(
                local_position
            )

        accepted_indices = current_indices[
            accepted_local_positions
        ]

        if accepted_indices.size == 0:
            continue

        kept_indices.extend(
            accepted_indices.tolist()
        )

        accepted_matrix = X_scaled[
            :,
            accepted_indices,
        ]

        # Compare retained features in the current block against all
        # subsequent feature blocks.
        for later_start in range(
            end,
            n_features,
            block_size,
        ):
            later_end = min(
                later_start + block_size,
                n_features,
            )

            later_indices = np.arange(
                later_start,
                later_end,
            )

            later_indices = later_indices[
                ~dropped[later_indices]
            ]

            if later_indices.size == 0:
                continue

            correlations = np.abs(
                accepted_matrix.T
                @ X_scaled[:, later_indices]
            ) / n_samples

            correlated_columns = np.any(
                correlations >= threshold,
                axis=0,
            )

            dropped[
                later_indices[correlated_columns]
            ] = True

        print(
            f"[CORRELATION] Processed features "
            f"{start + 1}-{end} of {n_features}. "
            f"Retained so far: {len(kept_indices)}"
        )

    return np.asarray(
        kept_indices,
        dtype=int,
    )


# ============================================================
# Descriptor-family handling
# ============================================================

def normalize_name(text):
    """Normalize a column/family name for conservative matching."""
    return re.sub(
        r"[^A-Za-z0-9]",
        "",
        str(text),
    ).upper()


def infer_feature_family(feature_name):
    """
    Infer the iLearn descriptor family from a retained feature-column name.

    Matching strategy:
    1. normalized prefix match;
    2. delimiter-aware token match;
    3. return OTHER if no known family can be identified.
    """
    raw = str(feature_name)
    normalized = normalize_name(raw)

    families_sorted = sorted(
        KNOWN_FAMILIES,
        key=lambda x: len(normalize_name(x)),
        reverse=True,
    )

    for family in families_sorted:
        family_normalized = normalize_name(
            family
        )
        if normalized.startswith(
            family_normalized
        ):
            return family

    tokens = [
        token
        for token in re.split(
            r"[\s_:\-|./()\[\]]+",
            raw,
        )
        if token
    ]

    normalized_tokens = {
        normalize_name(token)
        for token in tokens
    }

    for family in families_sorted:
        if normalize_name(family) in normalized_tokens:
            return family

    return "OTHER"


def build_group_normalization(
    retained_feature_names,
):
    """
    Build per-feature weights for group-normalized Euclidean distance.

    For a descriptor family g containing p_g retained features, every feature
    receives squared-distance weight 1/p_g. Therefore the transformed feature
    value is multiplied by 1/sqrt(p_g).

    Returns
    -------
    sqrt_weights : numpy.ndarray
        Multiplicative factor for every retained feature.

    family_report_df : pandas.DataFrame
        Audit table with feature counts and weights for every family.
    """
    feature_families = np.asarray(
        [
            infer_feature_family(feature_name)
            for feature_name in retained_feature_names
        ],
        dtype=object,
    )

    other_mask = feature_families == "OTHER"

    if np.any(other_mask):
        unknown_features = np.asarray(
            retained_feature_names
        )[other_mask]

        examples = "\n    ".join(
            str(name)
            for name in unknown_features[:20]
        )

        raise ValueError(
            "Some retained feature columns could not be assigned to a "
            "known iLearn descriptor family. Group normalization is stopped "
            "to avoid silently assigning an incorrect weight.\n"
            f"Unrecognized features: {len(unknown_features)}\n"
            f"First examples:\n    {examples}\n"
            "Add the corresponding family name to KNOWN_FAMILIES or update "
            "infer_feature_family() before rerunning."
        )

    sqrt_weights = np.empty(
        len(retained_feature_names),
        dtype=np.float64,
    )

    report_rows = []
    total_features = len(
        retained_feature_names
    )

    # Preserve first-occurrence family order from the retained columns.
    family_order = list(
        dict.fromkeys(
            feature_families.tolist()
        )
    )

    for family in family_order:
        family_indices = np.flatnonzero(
            feature_families == family
        )

        n_family_features = len(
            family_indices
        )

        squared_distance_weight_per_feature = (
            1.0 / n_family_features
        )

        sqrt_weight_per_feature = (
            1.0 / np.sqrt(n_family_features)
        )

        sqrt_weights[
            family_indices
        ] = sqrt_weight_per_feature

        report_rows.append(
            {
                "family": family,
                "n_retained_features": n_family_features,
                "feature_share_pct": (
                    100.0
                    * n_family_features
                    / total_features
                ),
                "squared_distance_weight_per_feature": (
                    squared_distance_weight_per_feature
                ),
                "sqrt_weight_applied_to_feature": (
                    sqrt_weight_per_feature
                ),
                # Sum_j in family (1 / p_g) = 1.
                "total_structural_family_weight": 1.0,
            }
        )

    family_report_df = pd.DataFrame(
        report_rows
    )

    family_report_df = (
        family_report_df
        .sort_values(
            "n_retained_features",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return (
        sqrt_weights,
        feature_families,
        family_report_df,
    )


# ============================================================
# Feature preprocessing
# ============================================================

def prepare_feature_space(
    positive_df,
    negative_df,
):
    """
    Prepare the iLearn feature space used to calculate distances.

    Processing steps (unchanged from the original method):
    1. Combine positives and candidate negatives.
    2. Remove constant features.
    3. Standardize remaining features.
    4. Remove highly correlated features.

    Then, for the NEW distance only:
    5. Assign each retained feature to its descriptor family.
    6. Multiply each feature from family g by 1/sqrt(p_g).

    Returns
    -------
    X_positive_weighted : numpy.ndarray
        Standardized and group-weighted positive feature matrix.

    X_negative_weighted : numpy.ndarray
        Standardized and group-weighted candidate-negative feature matrix.

    retained_feature_names : numpy.ndarray
        Names of retained features.

    feature_families : numpy.ndarray
        Family assigned to each retained feature.

    family_report_df : pandas.DataFrame
        Family counts and normalization weights.
    """
    feature_columns = [
        column
        for column in positive_df.columns
        if column not in METADATA_COLUMNS
    ]

    if not feature_columns:
        raise ValueError(
            "No feature columns were found after excluding "
            "'#', 'sequence', and 'label'."
        )

    print(
        f"[INFO] Original number of features: "
        f"{len(feature_columns)}"
    )

    combined_features = pd.concat(
        [
            positive_df[feature_columns],
            negative_df[feature_columns],
        ],
        ignore_index=True,
    )

    X_all = combined_features.to_numpy(
        dtype=np.float64,
        copy=True,
    )

    # --------------------------------------------------------
    # Remove constant features -- unchanged
    # --------------------------------------------------------

    variance_filter = VarianceThreshold(
        threshold=0.0
    )

    X_nonconstant = variance_filter.fit_transform(
        X_all
    )

    nonconstant_feature_names = np.asarray(
        feature_columns
    )[variance_filter.get_support()]

    n_constant_removed = (
        len(feature_columns)
        - len(nonconstant_feature_names)
    )

    print(
        f"[INFO] Constant features removed: "
        f"{n_constant_removed}"
    )

    print(
        f"[INFO] Features remaining after variance filter: "
        f"{len(nonconstant_feature_names)}"
    )

    # --------------------------------------------------------
    # Standardize each feature -- unchanged
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_nonconstant
    )

    # --------------------------------------------------------
    # Remove highly correlated features -- unchanged
    # --------------------------------------------------------

    retained_indices = remove_highly_correlated_features(
        X_scaled=X_scaled,
        threshold=CORRELATION_THRESHOLD,
        block_size=CORRELATION_BLOCK_SIZE,
    )

    X_final = X_scaled[
        :,
        retained_indices,
    ]

    retained_feature_names = nonconstant_feature_names[
        retained_indices
    ]

    n_correlated_removed = (
        len(nonconstant_feature_names)
        - len(retained_feature_names)
    )

    print(
        f"[INFO] Correlated features removed: "
        f"{n_correlated_removed}"
    )

    print(
        f"[INFO] Final number of features before group weighting: "
        f"{len(retained_feature_names)}"
    )

    # --------------------------------------------------------
    # NEW: descriptor-family normalization
    # --------------------------------------------------------

    (
        sqrt_weights,
        feature_families,
        family_report_df,
    ) = build_group_normalization(
        retained_feature_names
    )

    print(
        f"[INFO] Descriptor families represented after filtering: "
        f"{len(family_report_df)}"
    )

    print("\n[FAMILY NORMALIZATION]")
    print(
        family_report_df.to_string(
            index=False,
            formatters={
                "feature_share_pct": "{:.3f}".format,
                "squared_distance_weight_per_feature": "{:.8f}".format,
                "sqrt_weight_applied_to_feature": "{:.8f}".format,
                "total_structural_family_weight": "{:.3f}".format,
            },
        )
    )

    # Multiplying by 1/sqrt(p_g) means that a subsequent ordinary Euclidean
    # norm is exactly the group-normalized Euclidean distance.
    X_group_weighted = (
        X_final * sqrt_weights
    )

    n_positive = len(
        positive_df
    )

    X_positive_weighted = X_group_weighted[
        :n_positive
    ]

    X_negative_weighted = X_group_weighted[
        n_positive:
    ]

    return (
        X_positive_weighted,
        X_negative_weighted,
        retained_feature_names,
        feature_families,
        family_report_df,
    )


# ============================================================
# Negative selection
# ============================================================

def select_negative_peptides(
    positive_df,
    negative_df,
    X_positive_weighted,
    X_negative_weighted,
    n_negatives_per_positive=N_NEGATIVES_PER_POSITIVE,
):
    """
    Select closest negatives using GROUP-NORMALIZED Euclidean distance.

    Apart from the distance definition, the selection procedure is identical
    to the original Min-Euclidean method.
    """
    required_negatives = (
        len(positive_df)
        * n_negatives_per_positive
    )

    if len(negative_df) < required_negatives:
        raise ValueError(
            f"{required_negatives} negative peptides are required, "
            f"but only {len(negative_df)} are available."
        )

    positive_sequences = (
        positive_df["sequence"]
        .astype(str)
        .to_numpy()
    )

    negative_sequences = (
        negative_df["sequence"]
        .astype(str)
        .to_numpy()
    )

    negative_lengths = np.fromiter(
        (
            len(sequence)
            for sequence in negative_sequences
        ),
        dtype=np.int32,
        count=len(negative_sequences),
    )

    available_mask = np.ones(
        len(negative_df),
        dtype=bool,
    )

    selected_negative_indices = []

    for positive_index, positive_sequence in enumerate(
        positive_sequences
    ):
        positive_length = len(
            positive_sequence
        )

        maximum_length_difference = (
            compute_maximum_length_difference(
                positive_length
            )
        )

        minimum_allowed_length = max(
            1,
            positive_length - maximum_length_difference,
        )

        maximum_allowed_length = (
            positive_length + maximum_length_difference
        )

        length_differences = np.abs(
            negative_lengths - positive_length
        )

        candidate_mask = (
            available_mask
            & (
                length_differences
                <= maximum_length_difference
            )
        )

        candidate_indices = np.flatnonzero(
            candidate_mask
        )

        if len(candidate_indices) < n_negatives_per_positive:
            raise RuntimeError(
                f"Positive peptide '{positive_sequence}' "
                f"(length {positive_length}) has only "
                f"{len(candidate_indices)} unused negative "
                f"candidates in the allowed length range "
                f"{minimum_allowed_length}-"
                f"{maximum_allowed_length}. "
                f"{n_negatives_per_positive} candidates "
                f"are required."
            )

        # ----------------------------------------------------
        # GROUP-NORMALIZED Euclidean distance
        # ----------------------------------------------------
        # X_*_weighted already contains the factor 1/sqrt(p_g)
        # for every feature belonging to family g. Therefore this
        # ordinary norm is exactly:
        # sqrt(sum_g mean_{j in g}((x_j-y_j)^2)).
        differences = (
            X_negative_weighted[candidate_indices]
            - X_positive_weighted[positive_index]
        )

        distances = np.linalg.norm(
            differences,
            axis=1,
        )

        # Stable sorting resolves exact ties according to the original
        # order in the negative CSV -- unchanged from the original method.
        ordered_positions = np.argsort(
            distances,
            kind="stable",
        )

        selected_positions = ordered_positions[
            :n_negatives_per_positive
        ]

        selected_indices = candidate_indices[
            selected_positions
        ]

        selected_distances = distances[
            selected_positions
        ]

        # Selected negatives cannot be reused -- unchanged.
        available_mask[
            selected_indices
        ] = False

        selected_negative_indices.extend(
            selected_indices.tolist()
        )

        print(
            f"\n[POSITIVE {positive_index + 1}/"
            f"{len(positive_df)}] "
            f"{positive_sequence} "
            f"(length: {positive_length})"
        )

        print(
            f"  Allowed negative length range: "
            f"{minimum_allowed_length}-"
            f"{maximum_allowed_length}"
        )

        print(
            f"  Available candidates in window: "
            f"{len(candidate_indices)}"
        )

        for rank, (
            negative_index,
            distance,
        ) in enumerate(
            zip(
                selected_indices,
                selected_distances,
            ),
            start=1,
        ):
            negative_sequence = negative_sequences[
                negative_index
            ]

            negative_length = len(
                negative_sequence
            )

            print(
                f"  {rank}. "
                f"{negative_sequence} | "
                f"group-normalized distance: {distance:.6f} | "
                f"length: {negative_length} | "
                f"length difference: "
                f"{abs(positive_length - negative_length)}"
            )

    # Restore the relative order from the original negative CSV -- unchanged.
    selected_negative_indices = np.asarray(
        selected_negative_indices,
        dtype=np.int64,
    )

    selected_negative_indices.sort()

    selected_negative_df = negative_df.iloc[
        selected_negative_indices
    ]

    # Positives first, selected negatives afterwards -- unchanged.
    result_df = pd.concat(
        [
            positive_df,
            selected_negative_df,
        ],
        ignore_index=True,
    )

    return result_df


# ============================================================
# Main program
# ============================================================

def main():
    args = parse_arguments()

    print("[INFO] Reading input files...")

    positive_df = pd.read_csv(
        args.positive_csv
    )

    negative_df = pd.read_csv(
        args.negative_csv
    )

    # Use the same columns and column order as the positive CSV.
    negative_df = negative_df.loc[
        :,
        positive_df.columns,
    ]

    print(
        f"[INFO] Positive peptides: "
        f"{len(positive_df)}"
    )

    print(
        f"[INFO] Candidate negative peptides: "
        f"{len(negative_df)}"
    )

    print(
        f"[INFO] Negatives selected per positive: "
        f"{N_NEGATIVES_PER_POSITIVE}"
    )

    print(
        f"[INFO] Correlation threshold: "
        f"{CORRELATION_THRESHOLD}"
    )

    print(
        f"[INFO] Relative length window: "
        f"{LENGTH_WINDOW_FRACTION * 100:.0f}%"
    )

    print(
        f"[INFO] Minimum allowed length difference: "
        f"{MINIMUM_LENGTH_DIFFERENCE}"
    )

    print(
        "[INFO] Distance: group-normalized Euclidean "
        "(equal structural weight per descriptor family)"
    )

    (
        X_positive_weighted,
        X_negative_weighted,
        retained_feature_names,
        feature_families,
        family_report_df,
    ) = prepare_feature_space(
        positive_df=positive_df,
        negative_df=negative_df,
    )

    result_df = select_negative_peptides(
        positive_df=positive_df,
        negative_df=negative_df,
        X_positive_weighted=X_positive_weighted,
        X_negative_weighted=X_negative_weighted,
        n_negatives_per_positive=(
            N_NEGATIVES_PER_POSITIVE
        ),
    )

    output_path = Path(
        args.output_csv
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    # Save an auditable report of the detected descriptor families and the
    # normalization actually applied.
    family_report_path = output_path.with_name(
        f"{output_path.stem}_family_report.csv"
    )

    family_report_df.to_csv(
        family_report_path,
        index=False,
    )

    # Optional retained-feature audit table, useful if family naming changes.
    retained_feature_report_path = output_path.with_name(
        f"{output_path.stem}_retained_features_with_family.csv"
    )

    retained_feature_report_df = pd.DataFrame(
        {
            "feature": retained_feature_names,
            "family": feature_families,
        }
    )

    retained_feature_report_df.to_csv(
        retained_feature_report_path,
        index=False,
    )

    n_selected_negatives = (
        len(positive_df)
        * N_NEGATIVES_PER_POSITIVE
    )

    print("\n[RESULT]")

    print(
        f"Positive peptides included: "
        f"{len(positive_df)}"
    )

    print(
        f"Negative peptides selected: "
        f"{n_selected_negatives}"
    )

    print(
        f"Total rows in resulting dataset: "
        f"{len(result_df)}"
    )

    print(
        f"Features used for distance calculation: "
        f"{len(retained_feature_names)}"
    )

    print(
        f"Descriptor families used: "
        f"{len(family_report_df)}"
    )

    print(
        f"Unused negative candidates: "
        f"{len(negative_df) - n_selected_negatives}"
    )

    print(
        f"Result saved to: "
        f"{output_path}"
    )

    print(
        f"Family report saved to: "
        f"{family_report_path}"
    )

    print(
        f"Retained-feature family map saved to: "
        f"{retained_feature_report_path}"
    )


if __name__ == "__main__":
    main()
