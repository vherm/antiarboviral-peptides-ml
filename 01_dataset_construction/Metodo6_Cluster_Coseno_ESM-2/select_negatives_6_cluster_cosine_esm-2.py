#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METADATA_COLUMNS = {"#", "sequence", "label"}

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


def compute_maximum_length_difference(sequence_length):
    """
    Calculate the maximum allowed integer length difference.

    The window corresponds to 10% of the positive peptide length,
    using conventional rounding and a minimum difference of one
    residue. This is the same rule used by the minimum-cosine-distance
    ESM-2 method.
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
    allowed because of conventional rounding or the minimum tolerance
    of one residue for short peptides.
    """
    if relative_difference <= 0.02:
        return 0
    if relative_difference <= 0.05:
        return 1
    return 2


def normalize_embeddings(X):
    """
    Normalize every ESM-2 embedding to unit L2 norm.

    No dimension-wise standardization, feature filtering, correlation
    filtering or dimensionality reduction is applied.
    """
    norms = np.linalg.norm(
        X,
        axis=1,
        keepdims=True
    )

    if np.any(norms == 0):
        zero_norm_rows = np.flatnonzero(
            norms.ravel() == 0
        )

        raise ValueError(
            "Zero-norm ESM-2 embeddings were found at row "
            f"positions: {zero_norm_rows.tolist()}."
        )

    return (X / norms).astype(
        np.float32,
        copy=False
    )


def load_and_prepare_data(positive_csv, negative_csv):
    """
    Read the CSV files and prepare the ESM-2 embedding space.

    The preprocessing is identical to the minimum-cosine-distance
    ESM-2 method:

    1. Extract the original ESM-2 embedding dimensions.
    2. Do not apply StandardScaler or feature filtering.
    3. Normalize every embedding to unit L2 norm.
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

    embedding_columns = [
        column
        for column in positive_df.columns
        if column not in METADATA_COLUMNS
    ]

    negative_embedding_columns = [
        column
        for column in negative_df.columns
        if column not in METADATA_COLUMNS
    ]

    if not embedding_columns:
        raise ValueError("No ESM-2 embedding columns were found.")

    if embedding_columns != negative_embedding_columns:
        raise ValueError(
            "The positive and negative CSV files must contain the same "
            "ESM-2 embedding columns in the same order."
        )

    positive_embeddings = positive_df[embedding_columns].to_numpy(
        dtype=np.float32,
        copy=True
    )
    negative_embeddings = negative_df[embedding_columns].to_numpy(
        dtype=np.float32,
        copy=True
    )

    if not np.isfinite(positive_embeddings).all():
        raise ValueError(
            "Positive ESM-2 embeddings contain NaN or infinite values."
        )
    if not np.isfinite(negative_embeddings).all():
        raise ValueError(
            "Negative ESM-2 embeddings contain NaN or infinite values."
        )

    positive_normalized = normalize_embeddings(
        positive_embeddings
    )
    negative_normalized = normalize_embeddings(
        negative_embeddings
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

    print(
        f"[INFO] Number of ESM-2 embedding dimensions: "
        f"{len(embedding_columns)}"
    )
    print("[INFO] Dimension-wise standardization: none")
    print("[INFO] Feature filtering: none")
    print("[INFO] Embedding preparation: row-wise L2 normalization")
    print("[INFO] Distance metric: cosine distance")

    return (
        positive_df,
        negative_df,
        embedding_columns,
        positive_sequences,
        negative_sequences,
        positive_lengths,
        negative_lengths,
        positive_normalized,
        negative_normalized
    )


def cosine_similarity(normalized_a, normalized_b):
    """Calculate cosine similarity between two normalized vectors."""
    return float(
        np.clip(
            np.dot(normalized_a, normalized_b),
            -1.0,
            1.0
        )
    )


def cosine_distance(normalized_a, normalized_b):
    """Calculate cosine distance between two normalized vectors."""
    return float(
        np.clip(
            1.0 - np.dot(normalized_a, normalized_b),
            0.0,
            2.0
        )
    )


def assign_negatives_to_clusters(
    positive_normalized,
    negative_normalized,
    positive_lengths,
    negative_lengths,
    chunk_size
):
    """
    Assign each negative to its nearest length-compatible positive
    according to cosine distance.
    """
    assignments = np.full(
        len(negative_normalized),
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

    for start in range(0, len(negative_normalized), chunk_size):
        end = min(start + chunk_size, len(negative_normalized))

        # All rows are unit-normalized, so the dot product is the
        # cosine similarity. Maximizing similarity is equivalent to
        # minimizing cosine distance.
        similarities = (
            negative_normalized[start:end]
            @ positive_normalized.T
        )

        compatible = (
            np.abs(
                negative_lengths[start:end, None]
                - positive_lengths[None, :]
            )
            <= maximum_length_differences[None, :]
        )

        similarities[~compatible] = -np.inf
        has_candidate = compatible.any(axis=1)

        chunk_assignments = assignments[start:end]
        chunk_assignments[has_candidate] = np.argmax(
            similarities[has_candidate],
            axis=1
        )
        assignments[start:end] = chunk_assignments

    clusters = [[] for _ in range(len(positive_normalized))]

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
        negative_length = int(negative_lengths[negative_index])

        if not is_length_compatible(
            receiver_length,
            negative_length
        ):
            continue

        difference = relative_length_difference(
            receiver_length,
            negative_length
        )

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
    positive_normalized,
    negative_normalized,
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

    positive_similarities = (
        positive_normalized
        @ positive_normalized.T
    )
    positive_distances = np.clip(
        1.0 - positive_similarities,
        0.0,
        2.0
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

        receiver_similarity = cosine_similarity(
            negative_normalized[negative_index],
            positive_normalized[receiver_index]
        )
        source_similarity = cosine_similarity(
            negative_normalized[negative_index],
            positive_normalized[source_index]
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
            "cosine_distance_to_receiver_positive": float(
                np.clip(1.0 - receiver_similarity, 0.0, 2.0)
            ),
            "cosine_similarity_to_receiver_positive":
                receiver_similarity,
            "cosine_distance_to_source_positive": float(
                np.clip(1.0 - source_similarity, 0.0, 2.0)
            ),
            "cosine_similarity_to_source_positive":
                source_similarity
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
            "Cluster-based negative sampling using ESM-2 embeddings "
            "and cosine distance."
        )
    )

    parser.add_argument(
        "positive_csv",
        help=(
            "CSV file containing positive peptides and ESM-2 "
            "embeddings."
        )
    )
    parser.add_argument(
        "negative_csv",
        help=(
            "CSV file containing candidate negative peptides and "
            "ESM-2 embeddings."
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
        embedding_columns,
        positive_sequences,
        negative_sequences,
        positive_lengths,
        negative_lengths,
        positive_normalized,
        negative_normalized
    ) = load_and_prepare_data(
        args.positive_csv,
        args.negative_csv
    )

    print(f"Positive peptides: {len(positive_df)}")
    print(f"Negative peptides in pool: {len(negative_df)}")
    print(
        "ESM-2 embedding dimensions used for distance: "
        f"{len(embedding_columns)}"
    )
    print(
        "Length window: 10% with conventional rounding "
        "and a minimum difference of one residue"
    )

    assignments, clusters = assign_negatives_to_clusters(
        positive_normalized,
        negative_normalized,
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
        positive_normalized,
        negative_normalized,
        positive_sequences,
        negative_sequences,
        positive_lengths,
        negative_lengths,
        N_NEGATIVES_PER_POSITIVE,
        args.seed
    )

    output_path = Path(args.output_csv)
    report_path = output_path.with_name(
        f"{output_path.stem}_selection_report.csv"
    )

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

    if incomplete:
        print(
            "Clusters with fewer than "
            f"{N_NEGATIVES_PER_POSITIVE} negatives: {incomplete}"
        )


if __name__ == "__main__":
    main()
