"""Score the test candidates and write the two submission files.

    python src/generate_submission.py --threshold 0.8

Reads work/candidates/candidates_test.tsv.gz (build_candidates.py test --k 17) and models/matcher_lgbm.txt.
Writes output/matching_results.tsv and output/candidate_pairs.tsv in the competition format:
one row per S1 (empty list when none), tab-separated, comma-separated ID lists.
"""
import argparse
import gzip
import time
from itertools import groupby

import lightgbm as lgb
import pandas as pd

from build_features import build_features
from common import CAND_DIR, MODEL_DIR, NORM_DIR, OUT_DIR, iter_s1_chunks, load_lookup


def write_candidate_pairs(cand_path, s1_ids, out_path):
    """Long format (one pair per row, grouped by S1) -> one row per S1 with a comma-separated list."""
    with gzip.open(cand_path, "rt") as f, open(out_path, "w") as o:
        next(f)
        o.write("source1_entity_id\tcandidate_entity_ids\n")
        groups = groupby((l.rstrip("\n").split("\t") for l in f), key=lambda x: x[0])
        pending = next(groups, None)
        for sid in s1_ids:
            if pending is not None and pending[0] == sid:
                o.write(f"{sid}\t{','.join(c for _, c in pending[1])}\n")
                pending = next(groups, None)
            else:
                o.write(f"{sid}\t\n")
    assert pending is None, "candidate file order differs from test_source1 order"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(MODEL_DIR / "matcher_lgbm.txt"))
    ap.add_argument("--candidates", default=str(CAND_DIR / "candidates_test.tsv.gz"))
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--chunk", type=int, default=300_000, help="pairs per scoring chunk")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = lgb.Booster(model_file=a.model)
    feats = model.feature_name()
    house = "num_conflict" in feats
    s1_ids = pd.read_csv(NORM_DIR / "test_source1_normalized.tsv", sep="\t", dtype=str,
                         usecols=["entity_id"])["entity_id"].tolist()

    need, s1need = set(), set()
    for ch in pd.read_csv(a.candidates, sep="\t", dtype=str, chunksize=2_000_000):
        need.update(ch.candidate_entity_id.unique())
        s1need.update(ch.source1_entity_id.unique())
    s1_lk = load_lookup([NORM_DIR / "test_source1_normalized.tsv"], s1need)
    ot_lk = load_lookup([NORM_DIR / "test_source2_normalized.tsv", NORM_DIR / "test_source3_normalized.tsv"], need)
    del need, s1need

    kept, n, t0 = [], 0, time.time()
    for ch in iter_s1_chunks(a.candidates, a.chunk):
        ch["cand_rank"] = ch.groupby("source1_entity_id").cumcount()
        f = build_features(ch, s1_lk, ot_lk, house)
        f["prob"] = model.predict(f[feats])
        kept.append(f.loc[f.prob >= a.threshold, ["source1_entity_id", "candidate_entity_id"]])
        n += len(ch)
        print(f"{n:,} pairs scored ({round(time.time() - t0)}s)", flush=True)

    pred = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    agg = (pred.groupby("source1_entity_id")["candidate_entity_id"]
           .apply(lambda x: ",".join(sorted(set(x)))).rename("matched_entity_ids").reset_index())
    res = pd.DataFrame({"source1_entity_id": s1_ids}).merge(agg, on="source1_entity_id", how="left").fillna("")
    assert len(res) == len(s1_ids) and not res.source1_entity_id.duplicated().any()
    res.to_csv(OUT_DIR / "matching_results.tsv", sep="\t", index=False)
    write_candidate_pairs(a.candidates, s1_ids, OUT_DIR / "candidate_pairs.tsv")
    print(f"{len(res):,} S1 rows, {(res.matched_entity_ids != '').sum():,} with at least one match")
    print("wrote", OUT_DIR / "matching_results.tsv", "and", OUT_DIR / "candidate_pairs.tsv")


if __name__ == "__main__":
    main()
