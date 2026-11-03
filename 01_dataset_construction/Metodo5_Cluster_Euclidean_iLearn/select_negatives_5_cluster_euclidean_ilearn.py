"""
Cluster-based negative selection using GROUP-NORMALIZED Euclidean distance
over preprocessed iLearn descriptors.

This is a controlled modification of the original Cluster-Euclidean method.
The clustering/sampling algorithm is intentionally unchanged. The only
methodological change is the geometry used wherever Euclidean distance is
computed.

For each retained iLearn descriptor family g with p_g features:

    D_g^2(x, y) = (1 / p_g) * sum_{j in g} (x_j - y_j)^2

and
    D^2(x, y) = sum_g D_g^2(x, y)

Equivalently, every standardized retained feature in family g is multiplied
by 1/sqrt(p_g), after which ordinary Euclidean distance is applied.

The following behavior remains unchanged from the original script:
- positives and candidate negatives are processed together;
- constant features are removed;
- StandardScaler is fitted on the combined feature matrix;
- features with absolute Pearson correlation >= 0.95 are greedily removed;
- each negative is assigned to its nearest length-compatible positive;
- clusters are processed from smallest to largest;
- sampling uses the same 0-2%, 2-5%, and outer length intervals with weights
  0.50, 0.33, and 0.17;
- clusters with fewer than 5 negatives borrow from the nearest compatible
  positive cluster with lendable negatives;
- random sampling remains controlled by the positional seed argument.

Additional audit files are written next to the output dataset:
- <output_stem>_selection_report.csv
- <output_stem>_family_report.csv
- <output_stem>_retained_features_with_family.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler


METADATA_COLUMNS = {"#", "sequence", "label"}

CORRELATION_THRESHOLD = 0.95
CORRELATION_BLOCK_SIZE = 256
INTERVAL_LABELS = (
    "0-2%",
    "2-5%",
    "5%-rounded-window-limit"
)
INTERVAL_WEIGHTS = np.array([0.50, 0.33, 0.17])

N_NEGATIVES_PER_POSITIVE = 5
LENGTH_WINDOW_FRACTION = 0.10
MINIMUM_LENGTH_DIFFERENCE = 1
DISTANCE_CHUNK_SIZE = 1000


# Known iLearn descriptor families used in this project.
# More-specific/longer names are checked first during matching.
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


def compute_maximum_length_difference(sequence_length):
    """
    Calculate the maximum allowed integer length difference.

    The window corresponds to 10% of the positive peptide length,
    using conventional rounding and a minimum difference of one
    residue. This is the same rule used by the minimum-distance
    iLearn method.
    """
    relative_difference = int(
        LENGTH_WINDOW_FRACTION * sequence_length + 0.5
    )

    return max(
        MINIMUM_LENGTH_DIFFERENCE,
        relative_difference
    )


def absolute_length_difference(reference_length, candidate_length):
    return abs(candidate_length - reference_length)


def relative_length_difference(reference_length, candidate_length):
    return (
        absolute_length_difference(reference_length, candidate_length)
        / reference_length
    )


def is_length_compatible(reference_length, candidate_length):
    return (
        absolute_length_difference(reference_length, candidate_length)
        <= compute_maximum_length_difference(reference_length)
    )


def length_interval(relative_difference):
    """
    Assign a compatible candidate to one of three sampling intervals.

    The third interval also contains candidates above 10% that are
    allowed only because of conventional rounding or the minimum
    tolerance of one residue for short peptides.
    """
    if relative_difference <= 0.02:
        return 0
    if relative_difference <= 0.05:
        return 1
    return 2


def get_feature_columns(df):
    return [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
        and not str(column).startswith("Unnamed:")
    ]


def remove_highly_correlated_features(
    X_scaled,
    threshold=CORRELATION_THRESHOLD,
    block_size=CORRELATION_BLOCK_SIZE
):
    """
    Apply the same greedy blockwise Pearson-correlation filter used
    in the minimum-distance iLearn method.

    Features are evaluated in their original order. A feature is
    removed when its absolute correlation with any previously retained
    feature is greater than or equal to ``threshold``.
    """
    n_samples, n_features = X_scaled.shape

    dropped = np.zeros(n_features, dtype=bool)
    kept_indices = []

    for start in range(0, n_features, block_size):
        end = min(start + block_size, n_features)

        current_indices = np.arange(start, end)
        current_indices = current_indices[~dropped[current_indices]]

        if current_indices.size == 0:
            continue

        current_matrix = X_scaled[:, current_indices]
        correlations_within_block = np.abs(
            current_matrix.T @ current_matrix
        ) / n_samples

        accepted_local_positions = []

        for local_position, feature_index in enumerate(current_indices):
            if accepted_local_positions:
                previous_correlations = correlations_within_block[
                    accepted_local_positions,
                    local_position
                ]

                if np.any(previous_correlations >= threshold):
                    dropped[feature_index] = True
                    continue

            accepted_local_positions.append(local_position)

        accepted_indices = current_indices[accepted_local_positions]

        if accepted_indices.size == 0:
            continue

        kept_indices.extend(accepted_indices.tolist())
        accepted_matrix = X_scaled[:, accepted_indices]

        for later_start in range(end, n_features, block_size):
            later_end = min(later_start + block_size, n_features)

            later_indices = np.arange(later_start, later_end)
            later_indices = later_indices[~dropped[later_indices]]

            if later_indices.size == 0:
                continue

            correlations = np.abs(
                accepted_matrix.T @ X_scaled[:, later_indices]
            ) / n_samples

            correlated_columns = np.any(
                correlations >= threshold,
                axis=0
            )

            dropped[later_indices[correlated_columns]] = True

        print(
            f"[CORRELATION] Processed features "
            f"{start + 1}-{end} of {n_features}. "
            f"Retained so far: {len(kept_indices)}"
        )

    return np.asarray(kept_indices, dtype=int)



def normalize_name(text):
    """Normalize a feature/family name for conservative matching."""
    return re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()


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
        key=lambda value: len(normalize_name(value)),
        reverse=True,
    )

    for family in families_sorted:
        family_normalized = normalize_name(family)
        if normalized.startswith(family_normalized):
            return family

    tokens = [
        token
        for token in re.split(r"[\s_:\-|./()\[\]]+", raw)
        if token
    ]
    normalized_tokens = {normalize_name(token) for token in tokens}

    for family in families_sorted:
        if normalize_name(family) in normalized_tokens:
            return family

    return "OTHER"


def build_group_normalization(retained_feature_names):
    """
    Build per-feature weights for group-normalized Euclidean distance.

    Every retained feature in family g, with p_g retained features, is
    multiplied by 1/sqrt(p_g). Ordinary Euclidean distance on the transformed
    matrix is therefore exactly:

        sqrt(sum_g mean_{j in g}((x_j - y_j)^2)).

    Returns
    -------
    sqrt_weights : numpy.ndarray
        Multiplicative factor for each retained feature.
    feature_families : numpy.ndarray
        Descriptor family assigned to each retained feature.
    family_report_df : pandas.DataFrame
        Counts and audit weights for each represented descriptor family.
    """
    feature_families = np.asarray(
        [infer_feature_family(name) for name in retained_feature_names],
        dtype=object,
    )

    other_mask = feature_families == "OTHER"
    if np.any(other_mask):
        unknown_features = np.asarray(retained_feature_names)[other_mask]
        examples = "\n    ".join(str(name) for name in unknown_features[:20])
        raise ValueError(
            "Some retained feature columns could not be assigned to a known "
            "iLearn descriptor family. Group normalization is stopped to "
            "avoid silently assigning an incorrect weight.\n"
            f"Unrecognized features: {len(unknown_features)}\n"
            f"First examples:\n    {examples}\n"
            "Update KNOWN_FAMILIES or infer_feature_family() before rerunning."
        )

    sqrt_weights = np.empty(len(retained_feature_names), dtype=np.float64)
    report_rows = []
    total_features = len(retained_feature_names)
    family_order = list(dict.fromkeys(feature_families.tolist()))

    for family in family_order:
        family_indices = np.flatnonzero(feature_families == family)
        n_family_features = len(family_indices)
        squared_weight = 1.0 / n_family_features
        sqrt_weight = 1.0 / np.sqrt(n_family_features)
        sqrt_weights[family_indices] = sqrt_weight

        report_rows.append({
            "family": family,
            "n_retained_features": n_family_features,
            "feature_share_pct": 100.0 * n_family_features / total_features,
            "squared_distance_weight_per_feature": squared_weight,
            "sqrt_weight_applied_to_feature": sqrt_weight,
            "total_structural_family_weight": 1.0,
        })

    family_report_df = (
        pd.DataFrame(report_rows)
        .sort_values("n_retained_features", ascending=False)
        .reset_index(drop=True)
    )

    return sqrt_weights, feature_families, family_report_df


def load_and_prepare_data(positive_csv, negative_csv):
    """
    Read the CSV files and prepare the common iLearn feature space.

    Base preprocessing is identical to the original Cluster-Euclidean method:

    1. Combine positives and candidate negatives.
    2. Remove constant features.
    3. Standardize the remaining features.
    4. Remove highly correlated features.

    Then, for the new distance only:
    5. Assign every retained feature to its iLearn descriptor family.
    6. Multiply each feature in family g by 1/sqrt(p_g).
    """
    positive_df = pd.read_csv(positive_csv)
    negative_df = pd.read_csv(negative_csv)

    for name, df in (("positive", positive_df), ("negative", negative_df)):
        if "sequence" not in df.columns:
            raise ValueError(
                f"The {name} CSV must contain a 'sequence' column."
            )
        if df["sequence"].isna().any():
            raise ValueError(
                f"The {name} CSV contains missing sequences."
            )

    feature_columns = get_feature_columns(positive_df)
    negative_feature_columns = get_feature_columns(negative_df)

    if not feature_columns:
        raise ValueError("No iLearn feature columns were found.")

    if set(feature_columns) != set(negative_feature_columns):
        raise ValueError(
            "Positive and negative CSV files must contain the same "
            "iLearn feature columns."
        )

    print(f"[INFO] Original number of features: {len(feature_columns)}")

    combined_features = pd.concat(
        [
            positive_df[feature_columns],
            negative_df[feature_columns]
        ],
        ignore_index=True
    )

    X_all = combined_features.to_numpy(
        dtype=np.float64,
        copy=True
    )

    if not np.isfinite(X_all).all():
        raise ValueError(
            "The iLearn feature matrices contain NaN or infinite values."
        )

    variance_filter = VarianceThreshold(threshold=0.0)
    X_nonconstant = variance_filter.fit_transform(X_all)

    nonconstant_feature_names = np.asarray(feature_columns)[
        variance_filter.get_support()
    ]

    n_constant_removed = (
        len(feature_columns) - len(nonconstant_feature_names)
    )

    print(
        f"[INFO] Constant features removed: {n_constant_removed}"
    )
    print(
        "[INFO] Features remaining after variance filter: "
        f"{len(nonconstant_feature_names)}"
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_nonconstant)

    retained_indices = remove_highly_correlated_features(
        X_scaled=X_scaled,
        threshold=CORRELATION_THRESHOLD,
        block_size=CORRELATION_BLOCK_SIZE
    )

    X_final = X_scaled[:, retained_indices]
    retained_feature_names = nonconstant_feature_names[retained_indices]

    sqrt_weights, feature_families, family_report_df = (
        build_group_normalization(retained_feature_names)
    )

    # Transform once so every subsequent Euclidean calculation uses the same
    # group-normalized geometry: cluster assignment, donor-cluster distance,
    # and the distances written to the selection report.
    X_distance = X_final * sqrt_weights[None, :]

    n_correlated_removed = (
        len(nonconstant_feature_names) - len(retained_feature_names)
    )

    print(
        f"[INFO] Correlated features removed: {n_correlated_removed}"
    )
    print(
        "[INFO] Final number of features used for distance: "
        f"{len(retained_feature_names)}"
    )

    n_positive = len(positive_df)

    positive_processed = X_distance[:n_positive].astype(
        np.float32,
        copy=False
    )
    negative_processed = X_distance[n_positive:].astype(
        np.float32,
        copy=False
    )

    positive_sequences = positive_df["sequence"].astype(str).to_numpy()
    negative_sequences = negative_df["sequence"].astype(str).to_numpy()

    positive_lengths = np.fromiter(
        (len(sequence) for sequence in positive_sequences),
        dtype=np.int32,
        count=len(positive_sequences)
    )
    negative_lengths = np.fromiter(
        (len(sequence) for sequence in negative_sequences),
        dtype=np.int32,
        count=len(negative_sequences)
    )

    if np.any(positive_lengths == 0) or np.any(negative_lengths == 0):
        raise ValueError("Empty peptide sequences are not allowed.")

    return (
        positive_df,
        negative_df,
        retained_feature_names,
        positive_sequences,
        negative_sequences,
        positive_lengths,
        negative_lengths,
        positive_processed,
        negative_processed,
        feature_families,
        family_report_df
    )


def assign_negatives_to_clusters(
    positive_scaled,
    negative_scaled,
    positive_lengths,
    negative_lengths,
    chunk_size
):
    """
    Assign each negative to its nearest length-compatible positive using
    group-normalized Euclidean distance.
    """
    assignments = np.full(
        len(negative_scaled),
        -1,
        dtype=np.int32
    )

    maximum_length_differences = np.maximum(
        MINIMUM_LENGTH_DIFFERENCE,
        (
            LENGTH_WINDOW_FRACTION
            * positive_lengths
            + 0.5
        ).astype(np.int32)
    )

    for start in range(0, len(negative_scaled), chunk_size):
        end = min(start + chunk_size, len(negative_scaled))

        distances = euclidean_distances(
            negative_scaled[start:end],
            positive_scaled
        )

        compatible = (
            np.abs(
                negative_lengths[start:end, None]
                - positive_lengths[None, :]
            )
            <= maximum_length_differences[None, :]
        )

        distances[~compatible] = np.inf
        has_candidate = compatible.any(axis=1)

        chunk_assignments = assignments[start:end]
        chunk_assignments[has_candidate] = np.argmin(
            distances[has_candidate],
            axis=1
        )
        assignments[start:end] = chunk_assignments

    clusters = [[] for _ in range(len(positive_scaled))]

    for negative_index, positive_index in enumerate(assignments):
        if positive_index >= 0:
            clusters[positive_index].append(negative_index)

    return assignments, clusters


def sample_by_length_intervals(
    candidate_indices,
    receiver_length,
    negative_lengths,
    number_to_select,
    rng
):
    """
    Sample without replacement using interval weights 0.50, 0.33, 0.17.
    """
    pools = [[], [], []]

    for negative_index in candidate_indices:
        difference = relative_length_difference(
            receiver_length,
            int(negative_lengths[negative_index])
        )

        if not is_length_compatible(
            receiver_length,
            int(negative_lengths[negative_index])
        ):
            continue

        interval_index = length_interval(difference)
        pools[interval_index].append(negative_index)

    selected = []

    while len(selected) < number_to_select:
        available_intervals = [
            interval_index
            for interval_index, pool in enumerate(pools)
            if pool
        ]

        if not available_intervals:
            break

        probabilities = INTERVAL_WEIGHTS[available_intervals].copy()
        probabilities /= probabilities.sum()

        chosen_interval = int(
            rng.choice(available_intervals, p=probabilities)
        )
        chosen_pool = pools[chosen_interval]
        chosen_position = int(rng.integers(len(chosen_pool)))

        selected.append(chosen_pool.pop(chosen_position))

    return selected


def remove_indices(indices, indices_to_remove):
    remove_set = set(indices_to_remove)
    return [index for index in indices if index not in remove_set]


def select_negatives(
    clusters,
    positive_scaled,
    negative_scaled,
    positive_sequences,
    negative_sequences,
    positive_lengths,
    negative_lengths,
    negatives_per_positive,
    seed
):
    rng = np.random.default_rng(seed)

    initial_cluster_sizes = [len(cluster) for cluster in clusters]
    cluster_order = sorted(
        range(len(clusters)),
        key=lambda index: (initial_cluster_sizes[index], index)
    )

    available = [list(cluster) for cluster in clusters]
    processed = np.zeros(len(clusters), dtype=bool)
    positive_distances = euclidean_distances(
        positive_scaled,
        positive_scaled
    )

    selected_by_positive = [[] for _ in clusters]
    records = []

    def register_selection(
        receiver_index,
        negative_index,
        source_index,
        selection_type
    ):
        receiver_length = int(positive_lengths[receiver_index])
        negative_length = int(negative_lengths[negative_index])
        absolute_difference = absolute_length_difference(
            receiver_length,
            negative_length
        )
        maximum_allowed_difference = (
            compute_maximum_length_difference(receiver_length)
        )
        relative_difference = relative_length_difference(
            receiver_length,
            negative_length
        )

        records.append({
            "receiver_positive_index": receiver_index + 1,
            "receiver_positive_sequence":
                positive_sequences[receiver_index],
            "receiver_positive_length": receiver_length,
            "negative_index_in_pool": negative_index + 1,
            "negative_sequence": negative_sequences[negative_index],
            "negative_length": negative_length,
            "selection_type": selection_type,
            "source_cluster_index": source_index + 1,
            "source_positive_sequence":
                positive_sequences[source_index],
            "source_cluster_initial_size":
                initial_cluster_sizes[source_index],
            "absolute_length_difference": absolute_difference,
            "maximum_allowed_length_difference":
                maximum_allowed_difference,
            "relative_length_difference": relative_difference,
            "length_interval":
                INTERVAL_LABELS[length_interval(relative_difference)],
            "distance_to_receiver_positive": float(
                np.linalg.norm(
                    negative_scaled[negative_index]
                    - positive_scaled[receiver_index]
                )
            ),
            "distance_to_source_positive": float(
                np.linalg.norm(
                    negative_scaled[negative_index]
                    - positive_scaled[source_index]
                )
            )
        })

    for receiver_index in cluster_order:
        receiver_length = int(positive_lengths[receiver_index])

        own_selected = sample_by_length_intervals(
            available[receiver_index],
            receiver_length,
            negative_lengths,
            negatives_per_positive,
            rng
        )

        available[receiver_index] = remove_indices(
            available[receiver_index],
            own_selected
        )
        selected_by_positive[receiver_index].extend(own_selected)

        for negative_index in own_selected:
            register_selection(
                receiver_index,
                negative_index,
                receiver_index,
                "own_cluster"
            )

        borrowed_count = 0

        while (
            len(selected_by_positive[receiver_index])
            < negatives_per_positive
        ):
            missing = (
                negatives_per_positive
                - len(selected_by_positive[receiver_index])
            )
            donor_options = []

            for donor_index in range(len(clusters)):
                if donor_index == receiver_index:
                    continue

                if not is_length_compatible(
                    receiver_length,
                    int(positive_lengths[donor_index])
                ):
                    continue

                reserve = (
                    0
                    if processed[donor_index]
                    else negatives_per_positive
                )
                lendable_count = max(
                    0,
                    len(available[donor_index]) - reserve
                )

                if lendable_count == 0:
                    continue

                eligible = [
                    negative_index
                    for negative_index in available[donor_index]
                    if is_length_compatible(
                        receiver_length,
                        int(negative_lengths[negative_index])
                    )
                ]

                if eligible:
                    donor_options.append((
                        float(
                            positive_distances[
                                receiver_index,
                                donor_index
                            ]
                        ),
                        donor_index,
                        lendable_count,
                        eligible
                    ))

            if not donor_options:
                break

            donor_options.sort(key=lambda item: (item[0], item[1]))
            _, donor_index, lendable_count, eligible = donor_options[0]

            number_to_borrow = min(
                missing,
                lendable_count,
                len(eligible)
            )

            borrowed = sample_by_length_intervals(
                eligible,
                receiver_length,
                negative_lengths,
                number_to_borrow,
                rng
            )

            if not borrowed:
                break

            available[donor_index] = remove_indices(
                available[donor_index],
                borrowed
            )
            selected_by_positive[receiver_index].extend(borrowed)
            borrowed_count += len(borrowed)

            for negative_index in borrowed:
                register_selection(
                    receiver_index,
                    negative_index,
                    donor_index,
                    "borrowed"
                )

        processed[receiver_index] = True

        print(
            f"Positive {receiver_index + 1}: "
            f"initial cluster={initial_cluster_sizes[receiver_index]}, "
            f"own={len(own_selected)}, "
            f"borrowed={borrowed_count}, "
            f"total={len(selected_by_positive[receiver_index])}"
        )

    return selected_by_positive, records


def save_results(
    positive_df,
    negative_df,
    selected_by_positive,
    records,
    output_csv,
    report_csv
):
    selected_negative_indices = sorted({
        negative_index
        for positive_selection in selected_by_positive
        for negative_index in positive_selection
    })

    selected_negative_df = negative_df.iloc[
        selected_negative_indices
    ].copy()

    combined_df = pd.concat(
        [positive_df.copy(), selected_negative_df],
        ignore_index=True
    )

    if "#" in combined_df.columns:
        combined_df["#"] = np.arange(1, len(combined_df) + 1)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    report_path = Path(report_csv)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(report_path, index=False)

    return len(selected_negative_indices)


def parse_arguments():
    """
    Read the four positional command-line arguments.

    Order
    -----
    1. Positive CSV file.
    2. Negative CSV file.
    3. Output CSV file.
    4. Random seed.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Cluster-based negative sampling using preprocessed iLearn "
            "features and group-normalized Euclidean distance."
        )
    )

    parser.add_argument(
        "positive_csv",
        help="CSV file containing positive peptides and iLearn features."
    )
    parser.add_argument(
        "negative_csv",
        help=(
            "CSV file containing candidate negative peptides and "
            "iLearn features."
        )
    )
    parser.add_argument(
        "output_csv",
        help="CSV file where the resulting dataset will be saved."
    )
    parser.add_argument(
        "seed",
        type=int,
        help="Random seed used for reproducible sampling."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    (
        positive_df,
        negative_df,
        retained_feature_names,
        positive_sequences,
        negative_sequences,
        positive_lengths,
        negative_lengths,
        positive_scaled,
        negative_scaled,
        feature_families,
        family_report_df
    ) = load_and_prepare_data(
        args.positive_csv,
        args.negative_csv
    )

    print(f"Positive peptides: {len(positive_df)}")
    print(f"Negative peptides in pool: {len(negative_df)}")
    print(
        "iLearn features retained for distance: "
        f"{len(retained_feature_names)}"
    )
    print(
        "Distance method: group-normalized Euclidean "
        "(equal structural weight per descriptor family)"
    )
    print(f"Descriptor families represented: {len(family_report_df)}")
    print("\nFamily normalization summary")
    print("----------------------------")
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
    print(
        "Length window: 10% with conventional rounding "
        "and a minimum difference of one residue"
    )

    assignments, clusters = assign_negatives_to_clusters(
        positive_scaled,
        negative_scaled,
        positive_lengths,
        negative_lengths,
        DISTANCE_CHUNK_SIZE
    )

    print(
        "Negatives without a length-compatible positive: "
        f"{np.sum(assignments == -1)}"
    )

    selected_by_positive, records = select_negatives(
        clusters,
        positive_scaled,
        negative_scaled,
        positive_sequences,
        negative_sequences,
        positive_lengths,
        negative_lengths,
        N_NEGATIVES_PER_POSITIVE,
        args.seed
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = output_path.with_name(
        f"{output_path.stem}_selection_report.csv"
    )
    family_report_path = output_path.with_name(
        f"{output_path.stem}_family_report.csv"
    )
    retained_feature_report_path = output_path.with_name(
        f"{output_path.stem}_retained_features_with_family.csv"
    )

    family_report_df.to_csv(family_report_path, index=False)
    pd.DataFrame({
        "retained_feature": retained_feature_names,
        "family": feature_families,
    }).to_csv(retained_feature_report_path, index=False)

    number_selected = save_results(
        positive_df,
        negative_df,
        selected_by_positive,
        records,
        output_path,
        report_path
    )

    incomplete = [
        index + 1
        for index, selected in enumerate(selected_by_positive)
        if len(selected) < N_NEGATIVES_PER_POSITIVE
    ]
    borrowed_count = sum(
        record["selection_type"] == "borrowed"
        for record in records
    )

    print("\nSelection summary")
    print("-----------------")
    print(f"Unique negatives selected: {number_selected}")
    print(f"Borrowed negatives: {borrowed_count}")
    print(
        "Completed clusters: "
        f"{len(positive_df) - len(incomplete)}/{len(positive_df)}"
    )
    print(f"Output dataset: {output_path}")
    print(f"Selection report: {report_path}")
    print(f"Family report: {family_report_path}")
    print(
        "Retained-feature family mapping: "
        f"{retained_feature_report_path}"
    )

    if incomplete:
        print(
            "Clusters with fewer than "
            f"{N_NEGATIVES_PER_POSITIVE} negatives: {incomplete}"
        )


if __name__ == "__main__":
    main()
