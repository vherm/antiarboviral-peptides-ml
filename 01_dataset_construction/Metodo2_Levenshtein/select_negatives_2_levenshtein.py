import argparse
import pandas as pd

NEGATIVES_PER_POSITIVE = 5

def levenshtein_distance(sequence_1, sequence_2):
    """
    Calculate the Levenshtein distance between two sequences
    using dynamic programming.
    """
    previous_row = list(range(len(sequence_2) + 1))

    for i, amino_acid_1 in enumerate(sequence_1, start=1):
        current_row = [i]

        for j, amino_acid_2 in enumerate(sequence_2, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1
            substitution = previous_row[j - 1]

            if amino_acid_1 != amino_acid_2:
                substitution += 1

            current_row.append(
                min(insertion, deletion, substitution)
            )

        previous_row = current_row

    return previous_row[-1]


def longest_common_substring_length(sequence_1, sequence_2):
    """
    Calculate the length of the longest common contiguous substring
    between two sequences using dynamic programming.
    """
    previous_row = [0] * (len(sequence_2) + 1)
    maximum_length = 0

    for amino_acid_1 in sequence_1:
        current_row = [0]

        for j, amino_acid_2 in enumerate(sequence_2, start=1):
            if amino_acid_1 == amino_acid_2:
                common_length = previous_row[j - 1] + 1
            else:
                common_length = 0

            current_row.append(common_length)

            if common_length > maximum_length:
                maximum_length = common_length

        previous_row = current_row

    return maximum_length


def select_negatives(positive_df, negative_df):
    """
    Select five unused negative peptides for each positive peptide.

    Selection criteria:
    1. Minimum Levenshtein distance.
    2. Minimum difference in sequence length.
    3. Maximum longest common substring length.
    4. Original order in the negative dataset.
    """
    available_negatives = negative_df.copy()
    selected_groups = []

    total_positives = len(positive_df)

    for position, (_, positive_row) in enumerate(
        positive_df.iterrows(),
        start=1,
    ):
        positive_sequence = positive_row["sequence"]
        positive_length = len(positive_sequence)

        candidates = []

        for index, negative_sequence in enumerate(
            available_negatives["sequence"]
        ):
            distance = levenshtein_distance(
                positive_sequence,
                negative_sequence,
            )

            negative_length = len(negative_sequence)

            length_difference = abs(
                positive_length - negative_length
            )

            common_substring_length = (
                longest_common_substring_length(
                    positive_sequence,
                    negative_sequence,
                )
            )

            candidates.append(
                (
                    distance,
                    length_difference,
                    common_substring_length,
                    index,
                    negative_sequence,
                    negative_length,
                )
            )

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                -item[2],
            )
        )

        selected_candidates = candidates[
            :NEGATIVES_PER_POSITIVE
        ]

        selected_indices = [
            index
            for (
                _,
                _,
                _,
                index,
                _,
                _,
            ) in selected_candidates
        ]

        selected_groups.append(
            available_negatives.iloc[selected_indices]
        )

        print()
        print(
            f"Positive peptide {position}/{total_positives}: "
            f"{positive_sequence}"
        )
        print(f"Positive length: {positive_length}")
        print("Selected negative peptides:")

        for selection_number, candidate in enumerate(
            selected_candidates,
            start=1,
        ):
            (
                distance,
                length_difference,
                common_substring_length,
                _,
                negative_sequence,
                negative_length,
            ) = candidate

            print(
                f"  {selection_number}. "
                f"Sequence: {negative_sequence} | "
                f"Distance: {distance} | "
                f"Length: {negative_length} | "
                f"Length difference: {length_difference} | "
                f"Longest common substring: "
                f"{common_substring_length}"
            )

        available_negatives = (
            available_negatives
            .drop(
                available_negatives.index[
                    selected_indices
                ]
            )
            .reset_index(drop=True)
        )

    return pd.concat(
        selected_groups,
        ignore_index=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select five negative peptides per positive peptide "
            "using Levenshtein distance, length difference and "
            "longest common substring."
        )
    )

    parser.add_argument(
        "positive_csv",
        help="CSV file containing the positive peptides.",
    )

    parser.add_argument(
        "negative_csv",
        help="CSV file containing all possible negative peptides.",
    )

    parser.add_argument(
        "output_csv",
        help="Name of the resulting CSV file.",
    )

    args = parser.parse_args()

    positive_df = pd.read_csv(args.positive_csv)
    negative_df = pd.read_csv(args.negative_csv)

    selected_negatives_df = select_negatives(
        positive_df,
        negative_df,
    )

    result_df = pd.concat(
        [positive_df, selected_negatives_df],
        ignore_index=True,
    )

    result_df.to_csv(
        args.output_csv,
        index=False,
    )

    print()
    print("Selection completed.")
    print(f"Positive peptides: {len(positive_df)}")
    print(
        f"Selected negative peptides: "
        f"{len(selected_negatives_df)}"
    )
    print(f"Total peptides: {len(result_df)}")
    print(f"Output file: {args.output_csv}")


if __name__ == "__main__":
    main()

