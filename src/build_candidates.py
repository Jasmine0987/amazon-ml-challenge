"""Stage A: multi-key blocking. Index S2/S3 on eight country-scoped key types, then keep the
top-K candidates per S1 ranked by how many independent keys they share.

    python src/build_candidates.py train --limit 50000 --k 500   # training sample
    python src/build_candidates.py test --k 17                   # test run

Output: work/candidates/candidates_<split>.tsv.gz (source1_entity_id, candidate_entity_id),
grouped by S1, best candidate first (row order within an S1 is the candidate rank).
"""
import argparse
import gzip
import re
from collections import Counter, defaultdict

import pandas as pd

from common import CAND_DIR, CHUNK, NORM_DIR

MAX_KEY_FREQ = 1200   # drop keys shared by more records than this
MIN_TOKEN_LEN = 3
STOPWORDS = {"the", "and", "for", "inc", "llc", "ltd", "limited", "company", "corp", "corporation",
             "co", "corporate", "group", "services", "service", "solutions", "solution",
             "international", "india", "usa", "us"}


def clean(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    return "" if x == "nan" else x


def tokens(text):
    text = clean(text)
    return [t for t in text.split() if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS] if text else []


def name_keys(name):
    ts = tokens(name)
    if not ts:
        return set()
    keys = set()
    for t in ts:
        keys.add(("NT", t))                      # distinctive token
        if len(t) >= 5:
            keys.add(("NP5", t[:5]))             # 5-char prefix
        if len(t) >= 6:
            keys.add(("NP6", t[:6]))             # 6-char prefix
    if len(name) >= 5:
        keys.add(("NE", name))                   # whole name
    if len(ts) >= 2:
        keys.add(("NS", " ".join(sorted(ts))))   # sorted tokens
    return keys


def address_keys(address):
    address = clean(address)
    ts = tokens(address)
    if not ts:
        return set()
    m = re.search(r"\b\d{1,6}\b", address)
    number = m.group(0) if m else ""
    keys = set()
    for t in ts:
        if len(t) >= 4:
            keys.add(("AT", t))                  # address token
            if number:
                keys.add(("AN", number + "|" + t))   # number + token
        if len(t) >= 6:
            keys.add(("AP6", t[:6]))             # 6-char prefix
    return keys


def all_keys(name, address, country):
    country = clean(country)
    return {("C", country) + k for k in name_keys(name)} | {("C", country) + k for k in address_keys(address)}


def row_keys(row):
    return all_keys(clean(row.business_name_normalized), clean(row.business_address_normalized), row.country)


def build_index(path):
    """Two passes: count key frequencies, then index only keys that are not too common."""
    print(f"indexing {path.name} ...")
    freq = Counter()
    for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=CHUNK):
        for r in ch.itertuples(index=False):
            freq.update(row_keys(r))
    usable = {k for k, n in freq.items() if n <= MAX_KEY_FREQ}
    print(f"  {len(freq):,} keys, {len(usable):,} usable")
    del freq
    index = defaultdict(list)
    for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=CHUNK):
        for r in ch.itertuples(index=False):
            for k in row_keys(r):
                if k in usable:
                    index[k].append(r.entity_id)
    return index


def generate(s1_path, idx2, idx3, out_path, k, limit=None):
    done = pairs = 0
    with gzip.open(out_path, "wt") as out:
        out.write("source1_entity_id\tcandidate_entity_id\n")
        for ch in pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False, chunksize=CHUNK):
            for r in ch.itertuples(index=False):
                score = Counter()
                for key in row_keys(r):
                    for c in idx2.get(key, ()):
                        score[c] += 1
                    for c in idx3.get(key, ()):
                        score[c] += 1
                for c, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))[:k]:
                    out.write(f"{r.entity_id}\t{c}\n")
                    pairs += 1
                done += 1
                if done % 100_000 == 0:
                    print(f"  S1 {done:,}  pairs {pairs:,}", flush=True)
                if limit and done >= limit:
                    return done, pairs
    return done, pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N S1 entities")
    ap.add_argument("--k", type=int, default=None, help="candidates kept per S1 (default 500 train, 17 test)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    k = a.k or (500 if a.split == "train" else 17)
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    out = a.out or str(CAND_DIR / f"candidates_{a.split}.tsv.gz")
    n = lambda i: NORM_DIR / f"{a.split}_source{i}_normalized.tsv"
    idx2, idx3 = build_index(n(2)), build_index(n(3))
    done, pairs = generate(n(1), idx2, idx3, out, k, a.limit)
    print(f"done: {done:,} S1, {pairs:,} pairs (K={k}) -> {out}")


if __name__ == "__main__":
    main()
