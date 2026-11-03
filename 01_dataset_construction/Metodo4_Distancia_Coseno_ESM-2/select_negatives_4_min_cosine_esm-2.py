#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

N_NEGATIVES_PER_POSITIVE = 5

LENGTH_WINDOW_FRACTION = 0.10
MINIMUM_LENGTH_DIFFERENCE = 1

METADATA_COLUMNS = {"#", "sequence", "label"}


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments():
    """
    Read command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Select negative peptides using cosine distance "
            "over ESM-2 embeddings and a relative sequence-length "
            "window."
        )
    )

    parser.add_argument(
        "positive_csv",
        help=(
            "CSV file containing positive peptides "
            "and ESM-2 embeddings."
        )
    )

    parser.add_argument(
        "negative_csv",
        help=(
            "CSV file containing candidate negative peptides "
            "and ESM-2 embeddings."
        )
    )

    parser.add_argument(
        "output_csv",
        help=(
            "CSV file where the resulting dataset "
            "will be saved."
        )
    )

    return parser.parse_args()


# ============================================================
# Length window
# ============================================================

def compute_maximum_length_difference(sequence_length):
    """
    Calculate the maximum allowed length difference.

    The window corresponds to 10% of the positive peptide length,
    using conventional rounding and a minimum difference of one
    residue.

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
        relative_difference
    )


# ============================================================
# Embedding preparation
# ============================================================

def normalize_embeddings(X):
    """
    Normalize every embedding to unit L2 norm.

    This operation is used only to calculate cosine distance:

        cosine_distance(x, y) =
            1 - dot(x / ||x||, y / ||y||)

    No dimension-wise scaling or standardization is applied.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix containing one embedding per row.

    Returns
    -------
    numpy.ndarray
        Matrix containing L2-normalized embeddings.
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

    return X / norms


def prepare_embedding_space(
    positive_df,
    negative_df
):
    """
    Extract and prepare the ESM-2 embeddings.

    No StandardScaler, feature filtering, correlation filtering
    or dimensionality reduction is applied.

    The embeddings are normalized to unit L2 norm only as part
    of the cosine-distance calculation.

    Returns
    -------
    X_positive_normalized : numpy.ndarray
        L2-normalized positive ESM-2 embeddings.

    X_negative_normalized : numpy.ndarray
        L2-normalized candidate-negative ESM-2 embeddings.

    embedding_columns : list
        Names of the ESM-2 embedding columns.
    """
    embedding_columns = [
        column
        for column in positive_df.columns
        if column not in METADATA_COLUMNS
    ]

    if not embedding_columns:
        raise ValueError(
            "No embedding columns were found after excluding "
            "'#', 'sequence', and 'label'."
        )

    negative_embedding_columns = [
        column
        for column in negative_df.columns
        if column not in METADATA_COLUMNS
    ]

    if embedding_columns != negative_embedding_columns:
        raise ValueError(
            "The positive and negative CSV files do not contain "
            "the same embedding columns in the same order."
        )

    X_positive = positive_df[
        embedding_columns
    ].to_numpy(
        dtype=np.float32,
        copy=True
    )

    X_negative = negative_df[
        embedding_columns
    ].to_numpy(
        dtype=np.float32,
        copy=True
    )

    X_positive_normalized = normalize_embeddings(
        X_positive
    )

    X_negative_normalized = normalize_embeddings(
        X_negative
    )

    print(
        f"[INFO] Number of ESM-2 embedding dimensions: "
        f"{len(embedding_columns)}"
    )

    print(
        "[INFO] Dimension-wise standardization: none"
    )

    print(
        "[INFO] Distance metric: cosine distance"
    )

    return (
        X_positive_normalized,
        X_negative_normalized,
        embedding_columns
    )


# ============================================================
# Negative selection
# ============================================================

def select_negative_peptides(
    positive_df,
    negative_df,
    X_positive_normalized,
    X_negative_normalized,
    n_negatives_per_positive=N_NEGATIVES_PER_POSITIVE
):
    """
    Select the closest negative peptides using cosine distance.

    For each positive peptide:

    1. Calculate a relative length window equal to 10% of the
       positive length, with a minimum of one residue.
    2. Consider only unused negatives inside that length window.
    3. Calculate cosine distances using the original ESM-2
       embedding space.
    4. Select the closest negative peptides.
    5. Remove the selected negatives from the candidate pool.

    The resulting DataFrame contains:

    1. All positive peptides in their original order.
    2. All selected negative peptides in their original relative
       order in the negative CSV file.

    Returns
    -------
    pandas.DataFrame
        Dataset containing positives first and selected negatives
        afterwards.
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
        count=len(negative_sequences)
    )

    available_mask = np.ones(
        len(negative_df),
        dtype=bool
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
            positive_length - maximum_length_difference
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

        # Since all vectors have unit L2 norm:
        #
        # cosine similarity = x dot y
        # cosine distance   = 1 - cosine similarity
        cosine_similarities = (
            X_negative_normalized[candidate_indices]
            @ X_positive_normalized[positive_index]
        )

        cosine_distances = (
            1.0 - cosine_similarities
        )

        # Correct very small floating-point deviations.
        cosine_distances = np.clip(
            cosine_distances,
            0.0,
            2.0
        )

        # Stable sorting resolves exact distance ties according
        # to the original order in the negative CSV.
        ordered_positions = np.argsort(
            cosine_distances,
            kind="stable"
        )

        selected_positions = ordered_positions[
            :n_negatives_per_positive
        ]

        selected_indices = candidate_indices[
            selected_positions
        ]

        selected_distances = cosine_distances[
            selected_positions
        ]

        selected_similarities = cosine_similarities[
            selected_positions
        ]

        # Selected negatives cannot be reused.
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
            similarity
        ) in enumerate(
            zip(
                selected_indices,
                selected_distances,
                selected_similarities
            ),
            start=1
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
                f"cosine distance: {distance:.6f} | "
                f"cosine similarity: {similarity:.6f} | "
                f"length: {negative_length} | "
                f"length difference: "
                f"{abs(positive_length - negative_length)}"
            )

    # Restore the relative order of selected negatives according
    # to their positions in the original negative CSV.
    selected_negative_indices = np.asarray(
        selected_negative_indices,
        dtype=np.int64
    )

    selected_negative_indices.sort()

    selected_negative_df = negative_df.iloc[
        selected_negative_indices
    ]

    # Positives first in their original order, followed by the
    # selected negatives in their original relative order.
    result_df = pd.concat(
        [
            positive_df,
            selected_negative_df
        ],
        ignore_index=True
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
        positive_df.columns
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
        f"[INFO] Relative length window: "
        f"{LENGTH_WINDOW_FRACTION * 100:.0f}%"
    )

    print(
        f"[INFO] Minimum allowed length difference: "
        f"{MINIMUM_LENGTH_DIFFERENCE}"
    )

    (
        X_positive_normalized,
        X_negative_normalized,
        embedding_columns
    ) = prepare_embedding_space(
        positive_df=positive_df,
        negative_df=negative_df
    )

    result_df = select_negative_peptides(
        positive_df=positive_df,
        negative_df=negative_df,
        X_positive_normalized=X_positive_normalized,
        X_negative_normalized=X_negative_normalized,
        n_negatives_per_positive=(
            N_NEGATIVES_PER_POSITIVE
        )
    )

    output_path = Path(
        args.output_csv
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        output_path,
        index=False
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
        f"ESM-2 embedding dimensions used: "
        f"{len(embedding_columns)}"
    )

    print(
        f"Unused negative candidates: "
        f"{len(negative_df) - n_selected_negatives}"
    )

    print(
        f"Result saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
