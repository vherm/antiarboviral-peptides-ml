import argparse
import pandas as pd

def trim_csv_columns(input_file, output_file, num_columns):
    """
    Keep the first N columns from a CSV file and save them to a new CSV file.
    """

    if num_columns <= 0:
        raise ValueError("The number of columns must be greater than 0.")

    df = pd.read_csv(input_file)

    if num_columns > df.shape[1]:
        raise ValueError(
            f"The input CSV has only {df.shape[1]} columns, "
            f"but {num_columns} columns were requested."
        )

    trimmed_df = df.iloc[:, :num_columns]
    trimmed_df.to_csv(output_file, index=False)

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Original dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"New dimensions: {trimmed_df.shape[0]} rows x {trimmed_df.shape[1]} columns")

def main():
    parser = argparse.ArgumentParser(
        description="Keep the first N columns from a CSV file."
    )

    parser.add_argument(
        "input_file",
        help="Input CSV file"
    )

    parser.add_argument(
        "output_file",
        help="Output CSV file"
    )

    parser.add_argument(
        "num_columns",
        type=int,
        help="Number of columns to keep"
    )

    args = parser.parse_args()

    trim_csv_columns(
        args.input_file,
        args.output_file,
        args.num_columns
    )

if __name__ == "__main__":
    main()
