
import sys
import re
import math
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter

BASE = Path("/kaggle/working")
NORM = BASE / "normalized"
OUT = BASE / "candidates"
OUT.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000

# Maximum number of rows allowed behind one blocking key.
MAX_KEY_FREQ = 1200

# Maximum candidates retained for each S1.
MAX_CANDIDATES = 500

# Minimum token length for useful blocking.
MIN_TOKEN_LEN = 3

STOPWORDS = {
    "the", "and", "for", "inc", "llc", "ltd", "limited",
    "company", "corp", "corporation", "co", "corporate",
    "group", "services", "service", "solutions", "solution",
    "international", "india", "usa", "us"
}


def clean(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    if x == "nan":
        return ""
    return x


def tokens(text):
    text = clean(text)
    if not text:
        return []

    return [
        t for t in text.split()
        if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS
    ]


def name_keys(name):
    ts = tokens(name)

    if not ts:
        return []

    keys = set()

    # Exact distinctive tokens.
    for t in ts:
        keys.add(("NT", t))

    # Prefixes of longer tokens.
    for t in ts:
        if len(t) >= 5:
            keys.add(("NP5", t[:5]))

        if len(t) >= 6:
            keys.add(("NP6", t[:6]))

    # Whole normalized name.
    if len(name) >= 5:
        keys.add(("NE", name))

    # Sorted-token representation helps with word-order changes.
    if len(ts) >= 2:
        sorted_name = " ".join(sorted(ts))
        keys.add(("NS", sorted_name))

    return keys


def address_keys(address):
    address = clean(address)

    if not address:
        return []

    ts = tokens(address)

    if not ts:
        return []

    keys = set()

    # Address number.
    m = re.search(r"\b\d{1,6}\b", address)
    number = m.group(0) if m else ""

    # Distinctive address tokens.
    for t in ts:
        if len(t) >= 4:
            keys.add(("AT", t))

        if len(t) >= 6:
            keys.add(("AP6", t[:6]))

    # Number + distinctive token.
    if number:
        for t in ts:
            if len(t) >= 4:
                keys.add(("AN", number + "|" + t))

    return keys


def all_keys(name, address, country):
    country = clean(country)

    keys = set()

    for k in name_keys(name):
        keys.add(("C", country) + k)

    for k in address_keys(address):
        keys.add(("C", country) + k)

    return keys


def read_source(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )


def build_index(source_path):
    print(f"\nBuilding index: {source_path.name}")

    index = defaultdict(list)
    frequency = Counter()

    total = 0

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    ):

        for row in chunk.itertuples(index=False):
            name = clean(row.business_name_normalized)
            address = clean(row.business_address_normalized)
            country = clean(row.country)
            entity_id = row.entity_id

            keys = all_keys(name, address, country)

            for key in keys:
                frequency[key] += 1

            total += 1

        print(f"Scanned {total:,} rows")

    print(f"Unique keys before filtering: {len(frequency):,}")

    usable_keys = {
        key for key, freq in frequency.items()
        if freq <= MAX_KEY_FREQ
    }

    print(f"Usable keys: {len(usable_keys):,}")

    # Second pass: construct compact index.
    total = 0

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    ):

        for row in chunk.itertuples(index=False):
            name = clean(row.business_name_normalized)
            address = clean(row.business_address_normalized)
            country = clean(row.country)
            entity_id = row.entity_id

            for key in all_keys(name, address, country):
                if key in usable_keys:
                    index[key].append(entity_id)

            total += 1

        print(f"Indexed {total:,} rows")

    print(f"Final index keys: {len(index):,}")

    return index


def generate_for_source1(
    s1_path,
    s2_index,
    s3_index,
    output_path,
    sample_limit=None
):

    print(f"\nGenerating candidates from {s1_path.name}")

    total_s1 = 0
    total_candidates = 0

    recall_rows = []

    with open(output_path, "w") as out:

        out.write("source1_entity_id\tcandidate_entity_id\n")

        for chunk in pd.read_csv(
            s1_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE
        ):

            for row in chunk.itertuples(index=False):

                s1_id = row.entity_id

                name = clean(row.business_name_normalized)
                address = clean(row.business_address_normalized)
                country = clean(row.country)

                keys = all_keys(name, address, country)

                # candidate -> number of independent keys shared
                scores = Counter()

                for key in keys:

                    for candidate in s2_index.get(key, ()):
                        scores[candidate] += 1

                    for candidate in s3_index.get(key, ()):
                        scores[candidate] += 1

                if scores:

                    # Rank by:
                    # 1. number of shared blocking keys
                    # 2. deterministic entity ID
                    ranked = sorted(
                        scores.items(),
                        key=lambda x: (-x[1], x[0])
                    )

                    candidates = [
                        candidate
                        for candidate, score in ranked[:MAX_CANDIDATES]
                    ]

                    for candidate in candidates:
                        out.write(f"{s1_id}\t{candidate}\n")

                    total_candidates += len(candidates)

                total_s1 += 1

                if total_s1 % 10_000 == 0:
                    avg = total_candidates / total_s1

                    print(
                        f"S1 processed: {total_s1:,} | "
                        f"avg candidates: {avg:.1f}"
                    )

                if sample_limit and total_s1 >= sample_limit:
                    print(f"\nSample limit reached: {sample_limit:,}")
                    return total_s1, total_candidates

    return total_s1, total_candidates


def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print("python build_candidates_v9.py 50000")
        print("python build_candidates_v9.py full")
        sys.exit(1)

    mode = sys.argv[1]

    s1 = NORM / "train_source1_normalized.tsv"
    s2 = NORM / "train_source2_normalized.tsv"
    s3 = NORM / "train_source3_normalized.tsv"

    print("Building S2 index...")
    s2_index = build_index(s2)

    print("\nBuilding S3 index...")
    s3_index = build_index(s3)

    sample_limit = None

    if mode != "full":
        sample_limit = int(mode)

    output = OUT / f"raw_candidates_v9_{mode}.tsv"

    processed, candidates = generate_for_source1(
        s1,
        s2_index,
        s3_index,
        output,
        sample_limit
    )

    print("\n==============================")
    print("V9 COMPLETE")
    print("==============================")
    print(f"S1 processed: {processed:,}")
    print(f"Candidates:    {candidates:,}")

    if processed:
        print(
            f"Average candidates/S1: "
            f"{candidates / processed:.2f}"
        )

    print(f"Output: {output}")


if __name__ == "__main__":
    main()
