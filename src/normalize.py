"""Raw TSV -> normalized TSV (adds business_name_normalized / business_address_normalized).

    python src/normalize.py train      # or: test
"""
import argparse

import pandas as pd

from common import CHUNK, DATA_DIR, NORM_DIR, find_data_file, normalize_text


def process(name, data_dir):
    src = find_data_file(name, data_dir)
    dst = NORM_DIR / name.replace(".tsv", "_normalized.tsv")
    first, total = True, 0
    for ch in pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False, chunksize=CHUNK):
        ch["business_name_normalized"] = ch["business_name"].map(normalize_text)
        ch["business_address_normalized"] = ch["business_address"].map(normalize_text)
        ch.to_csv(dst, sep="\t", index=False, mode="w" if first else "a", header=first)
        first, total = False, total + len(ch)
    print(f"{name}: {total:,} rows -> {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", choices=["train", "test"])
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    a = ap.parse_args()
    NORM_DIR.mkdir(parents=True, exist_ok=True)
    for i in (1, 2, 3):
        process(f"{a.split}_source{i}.tsv", a.data_dir)


if __name__ == "__main__":
    main()
