"""Stage B features for (S1, candidate) pairs. Library module used by train_matcher.py and
generate_submission.py.

Base model: 20 features. With house_numbers=True, six house-number features are added
(the post-deadline experiment that lifted validation entity F0.5 from 0.856 to 0.884).
"""
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import WRatio, ratio, token_set_ratio, token_sort_ratio

from common import TEXT_COLS

BASE_FEATURES = [
    "cand_rank", "name_wratio", "name_ratio", "name_tsort", "name_tset", "name_lev",
    "addr_wratio", "addr_ratio", "addr_tsort", "addr_tset", "addr_lev",
    "name_exact", "addr_exact", "name_missing", "addr_missing", "name_len_diff", "addr_len_diff",
    "country_match", "shared_name_tokens", "shared_addr_tokens",
]
NUM_FEATURES = ["num_both", "num_overlap", "num_a_only", "num_b_only", "num_jacc", "num_conflict"]


def _num_set(s):
    """Numeric tokens of an address, leading zeros stripped ('0016613' -> '16613')."""
    return {t.lstrip("0") or "0" for t in s.split() if t.isdigit()}


def build_features(df, s1_lookup, other_lookup, house_numbers=False):
    """df needs source1_entity_id, candidate_entity_id, cand_rank (and optionally label)."""
    c = TEXT_COLS
    other_lookup = other_lookup[~other_lookup.index.duplicated()]
    m = (df.merge(s1_lookup[c], left_on="source1_entity_id", right_index=True)
           .rename(columns={c[0]: "n1", c[1]: "d1", "country": "ca"})
           .merge(other_lookup[c], left_on="candidate_entity_id", right_index=True)
           .rename(columns={c[0]: "n2", c[1]: "d2", "country": "cb"}))
    n1, n2, d1, d2 = m.n1.tolist(), m.n2.tolist(), m.d1.tolist(), m.d2.tolist()
    for pre, A, B in (("name", n1, n2), ("addr", d1, d2)):
        m[pre + "_wratio"] = [WRatio(a, b) if a and b else 0 for a, b in zip(A, B)]
        m[pre + "_ratio"] = [ratio(a, b) if a and b else 0 for a, b in zip(A, B)]
        m[pre + "_tsort"] = [token_sort_ratio(a, b) if a and b else 0 for a, b in zip(A, B)]
        m[pre + "_tset"] = [token_set_ratio(a, b) if a and b else 0 for a, b in zip(A, B)]
        m[pre + "_lev"] = [Levenshtein.normalized_similarity(a, b) * 100 if a and b else 0 for a, b in zip(A, B)]
    nm, am = (m.n1 == "") | (m.n2 == ""), (m.d1 == "") | (m.d2 == "")
    m["name_exact"] = ((m.n1 == m.n2) & ~nm).astype(int)
    m["addr_exact"] = ((m.d1 == m.d2) & ~am).astype(int)
    m["name_missing"], m["addr_missing"] = nm.astype(int), am.astype(int)
    m["name_len_diff"] = (m.n1.str.len() - m.n2.str.len()).abs()
    m["addr_len_diff"] = (m.d1.str.len() - m.d2.str.len()).abs()
    m["country_match"] = (m.ca == m.cb).astype(int)
    m["shared_name_tokens"] = [len(set(a.split()) & set(b.split())) for a, b in zip(n1, n2)]
    m["shared_addr_tokens"] = [len(set(a.split()) & set(b.split())) for a, b in zip(d1, d2)]
    feats = list(BASE_FEATURES)
    if house_numbers:
        sa, sb = [_num_set(x) for x in d1], [_num_set(x) for x in d2]
        m["num_both"] = [int(bool(x and y)) for x, y in zip(sa, sb)]
        m["num_overlap"] = [len(x & y) for x, y in zip(sa, sb)]
        m["num_a_only"] = [len(x - y) for x, y in zip(sa, sb)]
        m["num_b_only"] = [len(y - x) for x, y in zip(sa, sb)]
        m["num_jacc"] = [len(x & y) / len(x | y) if (x | y) else 0 for x, y in zip(sa, sb)]
        m["num_conflict"] = [int(bool(x and y) and bool(x - y) and bool(y - x)) for x, y in zip(sa, sb)]
        feats += NUM_FEATURES
    keep = ["source1_entity_id", "candidate_entity_id"] + (["label"] if "label" in m else [])
    out = m[keep + feats].copy()
    out[feats] = out[feats].astype("float32")
    return out
