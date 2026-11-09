"""Label candidates, split by S1 entity, train the LightGBM matcher, sweep thresholds with the
competition's metric (per-S1 F0.5, singletons count 1.0 when you predict nothing).

    python src/train_matcher.py                  # base model, 20 features
    python src/train_matcher.py --house-number   # + 6 house-number features

Reads work/candidates/candidates_train.tsv.gz (build it with build_candidates.py train --k 500).
Writes models/matcher_lgbm.txt, reports/threshold_sweep.csv, reports/recall_at_k.json, reports/summary.json.
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_features import BASE_FEATURES, NUM_FEATURES, build_features
from common import (CAND_DIR, DATA_DIR, MODEL_DIR, NORM_DIR, REPORT_DIR,
                    find_data_file, iter_s1_chunks, load_lookup)

THRESHOLDS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)
PARAMS = dict(objective="binary", metric="average_precision", num_leaves=63, learning_rate=0.05,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5, n_jobs=-1, verbose=-1)


def make_scorer(val_ids, true_map):
    idx = pd.Index(sorted(val_ids))
    nt = np.array([len(true_map[s]) for s in idx], dtype=float)

    def score(sel):
        g = sel.groupby("source1_entity_id")
        tp = g["label"].sum().reindex(idx, fill_value=0).values.astype(float)
        npred = g.size().reindex(idx, fill_value=0).values.astype(float)
        P, R = tp.sum() / max(npred.sum(), 1), tp.sum() / nt.sum()
        pooled = 1.25 * P * R / (0.25 * P + R) if P + R else 0.0
        p = np.where(npred > 0, tp / np.maximum(npred, 1), 0)
        r = np.where(nt > 0, tp / np.maximum(nt, 1), 0)
        f = np.where(p + r > 0, 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12), 0)
        f = np.where(nt == 0, (npred == 0).astype(float), f)   # singleton: empty prediction = 1.0
        return dict(P=P, R=R, pooled_F05=pooled, entity_F05=f.mean())
    return score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default=str(CAND_DIR / "candidates_train.tsv.gz"))
    ap.add_argument("--n-val", type=int, default=3000, help="held-out S1 entities")
    ap.add_argument("--train-k", type=int, default=100, help="candidates per S1 used for training/validation")
    ap.add_argument("--eval-k", type=int, default=17, help="K at which the threshold is chosen")
    ap.add_argument("--house-number", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model-out", default=str(MODEL_DIR / "matcher_lgbm.txt"))
    a = ap.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # S1 entities present in the candidate file, and their true matches
    s1_ids = set()
    for ch in pd.read_csv(a.candidates, sep="\t", dtype=str, usecols=["source1_entity_id"], chunksize=2_000_000):
        s1_ids.update(ch["source1_entity_id"].unique())
    gt = pd.read_csv(find_data_file("train_ground_truth.tsv", DATA_DIR), sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt.source1_entity_id.isin(s1_ids)]
    true_map = {r.source1_entity_id: {x.strip() for x in r.matched_entity_ids.split(",") if x.strip()}
                for r in gt.itertuples(index=False)}
    order = np.array(sorted(true_map))
    np.random.default_rng(42).shuffle(order)                 # split by S1 entity, never by row
    val_s1 = set(order[:min(a.n_val, len(order) // 2)])
    print(f"{len(true_map):,} S1 entities, {len(val_s1):,} held out for validation")

    # label, split, subsample training negatives (all positives + top-10 ranks + 10% of the rest)
    rng = np.random.default_rng(1)
    tr_parts, va_parts, pos_by_rank = [], [], np.zeros(500, dtype=int)
    for ch in iter_s1_chunks(a.candidates):
        ch["cand_rank"] = ch.groupby("source1_entity_id").cumcount()
        ch["label"] = [1 if c in true_map.get(s, ()) else 0
                       for s, c in zip(ch.source1_entity_id, ch.candidate_entity_id)]
        np.add.at(pos_by_rank, ch.loc[ch.label == 1, "cand_rank"].values.clip(max=499), 1)
        ch = ch[ch.cand_rank < a.train_k]
        isv = ch.source1_entity_id.isin(val_s1)
        va_parts.append(ch[isv])
        t = ch[~isv]
        tr_parts.append(t[(t.label == 1) | (t.cand_rank < 10) | (rng.random(len(t)) < 0.10)])
    tr, va = pd.concat(tr_parts, ignore_index=True), pd.concat(va_parts, ignore_index=True)
    total = sum(len(v) for v in true_map.values())
    recall_at_k = {k: round(float(pos_by_rank[:k].sum() / total), 4) for k in (17, 25, 50, 100, 200, 500)}
    print("recall@K:", recall_at_k)
    print(f"train rows {len(tr):,} | validation rows {len(va):,}")

    # features
    s1_lk = load_lookup([NORM_DIR / "train_source1_normalized.tsv"], s1_ids)
    need = set(tr.candidate_entity_id) | set(va.candidate_entity_id)
    ot_lk = load_lookup([NORM_DIR / "train_source2_normalized.tsv", NORM_DIR / "train_source3_normalized.tsv"], need)
    tf = build_features(tr, s1_lk, ot_lk, a.house_number)
    vf = build_features(va, s1_lk, ot_lk, a.house_number)
    cols = BASE_FEATURES + (NUM_FEATURES if a.house_number else [])

    # train
    ds = lgb.Dataset(tf[cols], label=tf.label)
    model = lgb.train(dict(PARAMS, seed=a.seed, bagging_seed=a.seed, feature_fraction_seed=a.seed), ds, 500,
                      valid_sets=[lgb.Dataset(vf[cols], label=vf.label, reference=ds)],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(20)])
    model.save_model(a.model_out)
    print("saved", a.model_out)

    # threshold sweep with the competition metric
    vf["prob"] = model.predict(vf[cols])
    score = make_scorer(val_s1, true_map)
    rows = [dict(K=k, t=t, **score(vf[(vf.cand_rank < k) & (vf.prob >= t)]))
            for k in sorted({a.eval_k, 25, 50, 100} & set(range(1, a.train_k + 1))) for t in THRESHOLDS]
    sweep = pd.DataFrame(rows)
    sweep.to_csv(REPORT_DIR / "threshold_sweep.csv", index=False)
    json.dump(recall_at_k, open(REPORT_DIR / "recall_at_k.json", "w"))
    at_k = sweep[sweep.K == a.eval_k]
    print(at_k.round(4).to_string(index=False))
    best = at_k.loc[at_k.entity_F05.idxmax()]
    json.dump(dict(eval_k=a.eval_k, best_threshold=float(best.t), entity_F05=float(best.entity_F05),
                   house_number=a.house_number, n_val_s1=len(val_s1)),
              open(REPORT_DIR / "summary.json", "w"), indent=2)
    print(f"best threshold at K={a.eval_k}: {best.t} (entity F0.5 {best.entity_F05:.4f})")


if __name__ == "__main__":
    main()
