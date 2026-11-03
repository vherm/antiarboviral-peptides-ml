from pathlib import Path
import argparse

import pandas as pd


def build_missing_mask(df):
    """
    Create a Boolean mask identifying missing values.

    Besides values recognized as missing by pandas (NaN, NA, null, etc.),
    empty strings and strings containing only whitespace are also considered
    missing.
    """

    missing_mask = df.isna().copy()

    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        blank_values = (
            df[column]
            .astype("string")
            .str.strip()
            .eq("")
            .fillna(False)
        )
        missing_mask[column] = missing_mask[column] | blank_values

    return missing_mask


def summarize_csv_dimensions(input_folder):
    """
    Check the dimensions and missing values of all CSV files in a folder.

    The generated Excel workbook contains:
    - Summary: dimensions and missing-value counts for each CSV file.
    - Missing Values: exact rows and columns containing missing values.
    """

    folder = Path(input_folder)

    if not folder.exists():
        print(f"Error: The folder does not exist: {folder}")
        return

    if not folder.is_dir():
        print(f"Error: The provided path is not a folder: {folder}")
        return

    csv_files = sorted(folder.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files were found in the folder: {folder}")
        return

    summary = []
    missing_details = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            rows, total_columns = df.shape
            feature_columns = max(total_columns - 3, 0)

            missing_mask = build_missing_mask(df)
            total_missing_values = int(missing_mask.to_numpy().sum())
            rows_with_missing_values = int(missing_mask.any(axis=1).sum())
            columns_with_missing_values = int(missing_mask.any(axis=0).sum())

            summary.append({
                "File": csv_file.name,
                "Rows": rows,
                "Total Columns": total_columns,
                "Feature Columns": feature_columns,
                "Missing Values": total_missing_values,
                "Rows with Missing Values": rows_with_missing_values,
                "Columns with Missing Values": columns_with_missing_values,
                "Status": (
                    "MISSING VALUES FOUND"
                    if total_missing_values > 0
                    else "OK"
                ),
            })

            print(f"File: {csv_file.name}")
            print(f"  Rows: {rows}")
            print(f"  Total columns: {total_columns}")
            print(f"  Feature columns: {feature_columns}")
            print(f"  Missing values: {total_missing_values}")

            if total_missing_values > 0:
                print(
                    f"  Rows with missing values: "
                    f"{rows_with_missing_values}"
                )
                print(
                    f"  Columns with missing values: "
                    f"{columns_with_missing_values}"
                )

                missing_counts_by_column = missing_mask.sum()
                print("  Missing values by column:")
                for column, count in missing_counts_by_column.items():
                    if count > 0:
                        print(f"    - {column}: {int(count)}")

                print("  Missing-value locations:")

                for row_index in df.index[missing_mask.any(axis=1)]:
                    missing_columns = df.columns[
                        missing_mask.loc[row_index]
                    ].tolist()

                    # Data rows start at 1. CSV lines start at 2 because the
                    # first physical line contains the header.
                    data_row = int(row_index) + 1
                    csv_line = data_row + 1

                    print(
                        f"    - Data row {data_row} "
                        f"(CSV line {csv_line}): "
                        f"{', '.join(map(str, missing_columns))}"
                    )

                    for column in missing_columns:
                        column_position = df.columns.get_loc(column) + 1

                        detail = {
                            "File": csv_file.name,
                            "Data Row": data_row,
                            "CSV Line": csv_line,
                            "Column Position": column_position,
                            "Column": column,
                        }

                        # Include common identifiers when available, making it
                        # easier to locate the affected peptide or record.
                        if "#" in df.columns:
                            detail["Record ID"] = df.at[row_index, "#"]

                        if "sequence" in df.columns:
                            detail["Sequence"] = df.at[row_index, "sequence"]

                        missing_details.append(detail)
            else:
                print("  No missing values were found.")

            print("-" * 60)

        except Exception as error:
            summary.append({
                "File": csv_file.name,
                "Rows": "ERROR",
                "Total Columns": "ERROR",
                "Feature Columns": "ERROR",
                "Missing Values": "ERROR",
                "Rows with Missing Values": "ERROR",
                "Columns with Missing Values": "ERROR",
                "Status": f"ERROR: {error}",
            })

            print(f"Error reading file {csv_file.name}: {error}")
            print("-" * 60)

    summary_df = pd.DataFrame(summary)
    missing_details_df = pd.DataFrame(
        missing_details,
        columns=[
            "File",
            "Data Row",
            "CSV Line",
            "Column Position",
            "Column",
            "Record ID",
            "Sequence",
        ],
    )

    output_excel = Path.cwd() / "csv_summary.xlsx"

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        missing_details_df.to_excel(
            writer,
            sheet_name="Missing Values",
            index=False,
        )

    print(f"\nSummary saved to: {output_excel}")



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check the dimensions and missing values of all CSV files "
            "in a given folder."
        )
    )

    parser.add_argument(
        "input_folder",
        help="Path to the folder containing the CSV files.",
    )

    args = parser.parse_args()
    summarize_csv_dimensions(args.input_folder)


if __name__ == "__main__":
    main()
