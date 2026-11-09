
import sys
import re
import unicodedata
import pandas as pd
from pathlib import Path

INPUT_DIR = Path("/kaggle/input")
OUTPUT_DIR = Path("/kaggle/working/normalized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000


def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).lower()
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))

    x = x.replace("&", " and ")
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    return x


def normalize_series(series):
    return series.fillna("").astype(str).map(normalize_text)


def find_file(filename):
    matches = list(INPUT_DIR.rglob(filename))

    if not matches:
        raise FileNotFoundError(f"Could not find {filename}")

    return matches[0]


def process_file(filename):
    input_path = find_file(filename)

    output_name = filename.replace(".tsv", "_normalized.tsv")
    output_path = OUTPUT_DIR / output_name

    print(f"\nProcessing: {input_path}")
    print(f"Output:    {output_path}")

    first_chunk = True
    total = 0

    for chunk in pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):

        chunk["business_name_normalized"] = normalize_series(
            chunk["business_name"]
        )

        chunk["business_address_normalized"] = normalize_series(
            chunk["business_address"]
        )

        chunk.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk
        )

        first_chunk = False
        total += len(chunk)

        print(f"Processed {total:,} rows")

    print(f"Finished: {total:,} rows")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python normalize_and_save.py train")
        print("python normalize_and_save.py test")
        sys.exit(1)

    split = sys.argv[1]

    if split == "train":
        files = [
            "train_source1.tsv",
            "train_source2.tsv",
            "train_source3.tsv"
        ]

    elif split == "test":
        files = [
            "test_source1.tsv",
            "test_source2.tsv",
            "test_source3.tsv"
        ]

    else:
        raise ValueError("Use train or test")

    for filename in files:
        process_file(filename)

    print("\nAll normalization complete.")
