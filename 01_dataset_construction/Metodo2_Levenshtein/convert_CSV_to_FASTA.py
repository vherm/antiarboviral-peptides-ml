import argparse
import csv
import sys

def csv_to_fasta(input_csv, output_fasta):
    with open(input_csv, "r", encoding="utf-8", newline="") as csv_file, \
         open(output_fasta, "w", encoding="utf-8") as fasta_file:

        reader = csv.DictReader(csv_file)
        required_columns = {"#", "sequence", "label"}

        if reader.fieldnames is None:
            sys.exit("[ERROR] The input CSV file is empty.")

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            sys.exit(
                "[ERROR] Missing required column(s): "
                + ", ".join(sorted(missing_columns))
            )

        count = 0
        for row_number, row in enumerate(reader, start=2):
            name = row["#"].strip()
            sequence = row["sequence"].strip()
            label = row["label"].strip()

            if not name or not sequence or not label:
                print(
                    f"[WARNING] Skipping row {row_number}: "
                    "empty name, sequence, or label.",
                    file=sys.stderr
                )
                continue

            fasta_file.write(f">{name}|{label}\n")
            fasta_file.write(f"{sequence}\n")

            count += 1

    print(f"[OK] FASTA file created: {output_fasta}")
    print(f"[OK] Number of sequences written: {count}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a peptide CSV file to FASTA format."
    )

    parser.add_argument(
        "input_csv",
        help="Input CSV file with columns: #, sequence, label"
    )

    parser.add_argument(
        "output_fasta",
        help="Output FASTA file"
    )

    args = parser.parse_args()
    csv_to_fasta(args.input_csv, args.output_fasta)

if __name__ == "__main__":
    main()
