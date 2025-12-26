import csv
from pathlib import Path

def convert_csv_decimal_and_separator(
    input_csv,
    output_csv,
    in_sep=",",
    out_sep=";",
    in_decimal=".",
    out_decimal=","
):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    with input_csv.open("r", newline="", encoding="utf-8") as fin, \
         output_csv.open("w", newline="", encoding="utf-8") as fout:

        reader = csv.reader(fin, delimiter=in_sep)
        writer = csv.writer(fout, delimiter=out_sep, quoting=csv.QUOTE_MINIMAL)

        for row in reader:
            new_row = []
            for cell in row:
                # Only replace decimal point in numeric-looking fields
                if cell.replace(in_decimal, "", 1).isdigit():
                    cell = cell.replace(in_decimal, out_decimal)
                new_row.append(cell)

            writer.writerow(new_row)

    print(f"Converted file written to: {output_csv}")


def main():
    input_csv = "compression scripts/vit_b16_results_averaged.csv"
    output_csv = "compression scripts/vit_b16_results_averaged_converted.csv"

    convert_csv_decimal_and_separator(
        input_csv,
        output_csv
    )

if __name__ == "__main__":
    main()