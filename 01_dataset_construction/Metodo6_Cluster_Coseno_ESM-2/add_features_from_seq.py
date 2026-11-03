import argparse
import pandas as pd

def find_sequence_column(df, filename):
    """
    Detects the sequence column name.
    Accepted names: sequence, secuencia
    """
    possible_names = ["sequence", "secuencia"]

    for col in possible_names:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find a sequence column in '{filename}'. "
        "Expected 'sequence' or 'secuencia'."
    )

def check_basic_columns(df, filename):
    """
    Checks that the required basic columns are present.
    """
    if "#" not in df.columns:
        raise ValueError(f"File '{filename}' is missing column '#'.")

    if "label" not in df.columns:
        raise ValueError(f"File '{filename}' is missing column 'label'.")

    find_sequence_column(df, filename)

def prepare_feature_table(pos_df, neg_df, pos_seq_col, neg_seq_col):
    """
    Creates a single feature table from POS and NEG.

    Only the sequence column and the feature columns are kept.
    Feature columns are assumed to start at the fourth column.
    """

    pos_feature_cols = list(pos_df.columns[3:])
    neg_feature_cols = list(neg_df.columns[3:])

    if pos_feature_cols != neg_feature_cols:
        raise ValueError(
            "POS and NEG do not have the same feature columns "
            "from the fourth column onward."
        )

    pos_features = pos_df[[pos_seq_col] + pos_feature_cols].copy()
    neg_features = neg_df[[neg_seq_col] + neg_feature_cols].copy()

    pos_features = pos_features.rename(columns={pos_seq_col: "sequence_key"})
    neg_features = neg_features.rename(columns={neg_seq_col: "sequence_key"})

    feature_table = pd.concat(
        [pos_features, neg_features],
        axis=0,
        ignore_index=True
    )

    if feature_table["sequence_key"].duplicated().any():
        duplicated_sequences = (
            feature_table.loc[
                feature_table["sequence_key"].duplicated(),
                "sequence_key"
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Duplicated sequences were found in POS/NEG feature files. "
            f"Examples: {duplicated_sequences}"
        )

    return feature_table

def main():
    parser = argparse.ArgumentParser(
        description="Add peptide features to a SEC file using POS and NEG feature files."
    )

    parser.add_argument(
        "sec_csv",
        help="Input CSV file with peptide sequences: #, sequence/secuencia, label."
    )

    parser.add_argument(
        "pos_csv",
        help="Input CSV file with positive peptide features."
    )

    parser.add_argument(
        "neg_csv",
        help="Input CSV file with negative peptide features."
    )

    parser.add_argument(
        "output_csv",
        help="Output CSV file with sequences, labels and features."
    )

    args = parser.parse_args()

    sec_df = pd.read_csv(args.sec_csv)
    pos_df = pd.read_csv(args.pos_csv)
    neg_df = pd.read_csv(args.neg_csv)

    check_basic_columns(sec_df, args.sec_csv)
    check_basic_columns(pos_df, args.pos_csv)
    check_basic_columns(neg_df, args.neg_csv)

    sec_seq_col = find_sequence_column(sec_df, args.sec_csv)
    pos_seq_col = find_sequence_column(pos_df, args.pos_csv)
    neg_seq_col = find_sequence_column(neg_df, args.neg_csv)

    feature_table = prepare_feature_table(
        pos_df=pos_df,
        neg_df=neg_df,
        pos_seq_col=pos_seq_col,
        neg_seq_col=neg_seq_col
    )

    sec_with_key = sec_df.copy()
    sec_with_key = sec_with_key.rename(columns={sec_seq_col: "sequence_key"})

    result_df = sec_with_key.merge(
        feature_table,
        on="sequence_key",
        how="left"
    )

    missing_features = result_df[result_df.iloc[:, 3:].isna().all(axis=1)]

    if len(missing_features) > 0:
        examples = missing_features["sequence_key"].head(10).tolist()

        raise ValueError(
            f"Features were not found for {len(missing_features)} peptide(s). "
            f"Examples: {examples}"
        )

    result_df = result_df.rename(columns={"sequence_key": sec_seq_col})
    result_df.to_csv(args.output_csv, index=False)

    print("Feature assignment completed.")
    print(f"Peptides in SEC: {len(sec_df)}")
    print(f"Peptides in output file: {len(result_df)}")
    print(f"Feature columns added: {len(feature_table.columns) - 1}")
    print(f"Output file: {args.output_csv}")

if __name__ == "__main__":
    main()
