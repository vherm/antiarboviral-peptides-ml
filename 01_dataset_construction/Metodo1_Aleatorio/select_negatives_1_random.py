import argparse
import pandas as pd
import random

def read_csv_file(filename):
    """
    Reads a CSV file and returns a pandas DataFrame.
    """
    return pd.read_csv(filename)

def check_required_columns(df, filename):
    """
    Checks that the required columns are present in the DataFrame.
    """
    required_columns = ["#", "sequence", "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"File '{filename}' is missing required columns: {missing_columns}"
        )

def add_sequence_length(df):
    """
    Adds a temporary column with the length of each sequence.
    """
    df["_length"] = df["sequence"].astype(str).str.len()
    return df

def select_random_negatives(pos_df, ntot_df, n_negatives_per_positive=5, seed=123):
    """
    Selects random negative peptides from NTOT.

    For each positive peptide of length l, this function selects
    n_negatives_per_positive negative peptides of the same length.
    Selected negatives are removed from NTOT to avoid repetitions.
    """
    random.seed(seed)
    selected_negative_indices = []

    # Dictionary: sequence length -> list of available NTOT indices
    available_by_length = {}
    for idx, row in ntot_df.iterrows():
        seq_len = row["_length"]

        if seq_len not in available_by_length:
            available_by_length[seq_len] = []

        available_by_length[seq_len].append(idx)

    for _, pos_row in pos_df.iterrows():
        seq_len = pos_row["_length"]

        available_indices = available_by_length[seq_len]

        chosen_indices = random.sample(
            available_indices,
            n_negatives_per_positive
        )

        selected_negative_indices.extend(chosen_indices)

        # Remove selected negatives from the available pool
        available_by_length[seq_len] = [
            idx for idx in available_indices
            if idx not in chosen_indices
        ]

    neg_df = ntot_df.loc[selected_negative_indices].copy()
    return neg_df

def main():
    parser = argparse.ArgumentParser(
        description="Random negative peptide selection method."
    )

    parser.add_argument(
        "positive_csv",
        help="Input CSV file with positive peptides."
    )

    parser.add_argument(
        "negative_total_csv",
        help="Input CSV file with all possible negative peptides."
    )

    parser.add_argument(
        "output_csv",
        help="Output CSV file with positives followed by selected negatives."
    )

    parser.add_argument(
        "seed",
        type=int,
        help="Random seed for negative selection."
    )

    args = parser.parse_args()

    # Read input files
    pos_df = read_csv_file(args.positive_csv)
    ntot_df = read_csv_file(args.negative_total_csv)

    # Check required columns
    check_required_columns(pos_df, args.positive_csv)
    check_required_columns(ntot_df, args.negative_total_csv)

    # Add temporary sequence length column
    pos_df = add_sequence_length(pos_df)
    ntot_df = add_sequence_length(ntot_df)

    # Select negatives
    neg_df = select_random_negatives(
        pos_df=pos_df,
        ntot_df=ntot_df,
        n_negatives_per_positive=5,
        seed=args.seed
    )

    # Remove temporary column before saving
    pos_df = pos_df.drop(columns=["_length"])
    neg_df = neg_df.drop(columns=["_length"])

    # Combine POS followed by NEG
    final_df = pd.concat([pos_df, neg_df], axis=0, ignore_index=True)

    # Save output CSV
    final_df.to_csv(args.output_csv, index=False)

    print("Random negative selection completed.")
    print(f"Positive peptides: {len(pos_df)}")
    print(f"Selected negative peptides: {len(neg_df)}")
    print(f"Total peptides in output file: {len(final_df)}")
    print(f"Output file: {args.output_csv}")

if __name__ == "__main__":
    main()
