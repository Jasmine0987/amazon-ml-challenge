"""Shared paths and I/O helpers. Set ER_ROOT to relocate data/, work/, models/, output/."""
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("ER_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "data"
NORM_DIR = ROOT / "work" / "normalized"
CAND_DIR = ROOT / "work" / "candidates"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "output"
REPORT_DIR = ROOT / "reports"

TEXT_COLS = ["business_name_normalized", "business_address_normalized", "country"]
CHUNK = 100_000


def find_data_file(name, data_dir=DATA_DIR):
    """Locate a competition file anywhere below data_dir (flat or train/test sub-folders)."""
    hits = sorted(Path(data_dir).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {data_dir}")
    return hits[0]


def normalize_text(x):
    """lowercase, NFKD, strip accents, '&' -> 'and', drop punctuation, collapse spaces."""
    if pd.isna(x):
        return ""
    x = unicodedata.normalize("NFKD", str(x).lower())
    x = "".join(c for c in x if not unicodedata.combining(c)).replace("&", " and ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", x)).strip()


def iter_s1_chunks(path, chunksize=500_000):
    """Yield chunks of a file grouped by source1_entity_id without splitting one S1 across chunks."""
    carry = None
    for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize):
        if carry is not None:
            ch = pd.concat([carry, ch], ignore_index=True)
        last = ch["source1_entity_id"].iloc[-1]
        m = (ch["source1_entity_id"] == last).values
        carry, ch = ch[m], ch[~m]
        if len(ch):
            yield ch
    if carry is not None and len(carry):
        yield carry


def load_lookup(paths, wanted):
    """entity_id -> normalized name/address/country, for the wanted ids only."""
    parts = []
    for p in paths:
        for ch in pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False,
                              usecols=["entity_id"] + TEXT_COLS, chunksize=500_000):
            x = ch[ch.entity_id.isin(wanted)]
            if len(x):
                parts.append(x)
    return pd.concat(parts).set_index("entity_id")
