"""Generates the README diagrams as standalone SVG (editorial style: paper/ink/one accent,
1px hairlines, no shadows, layout on a 4px grid). Run: python make_diagrams.py"""
PAPER, INK, MUTED, HAIR, ACC, TINT = "#F5F5F5", "#1A1A1A", "#5B6B7B", "#C9CCD0", "#C2410C", "#FBE9DC"
SERIF = "Georgia, 'Times New Roman', serif"
SANS = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

def svg(w, h, title, desc, body, name):
    s = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-labelledby="t d">
<title id="t">{title}</title><desc id="d">{desc}</desc>
<defs><marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto"><path d="M1 1 L7 4 L1 7" fill="none" stroke="{MUTED}" stroke-width="1.2"/></marker></defs>
<rect width="{w}" height="{h}" fill="{PAPER}"/>
{body}
</svg>'''
    open(name, "w", encoding="utf-8").write(s)

def head(title, sub):
    return (f'<text x="36" y="48" font-family="{SERIF}" font-style="italic" font-size="24" fill="{INK}">{title}</text>\n'
            f'<text x="36" y="72" font-family="{MONO}" font-size="11" fill="{MUTED}">{sub}</text>\n')

def T(x, y, s, size=12, fam=None, fill=INK, anchor="start", weight="400", style="normal"):
    return (f'<text x="{x}" y="{y}" font-family="{fam or SANS}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" font-style="{style}">{s}</text>\n')

# ---------- 1. pipeline ----------
b = head("Two-stage entity resolution", "TEST SET  S1 1,732,544  ·  S2 4,887,273  ·  S3 5,082,316 records")
stages = [("01", "Normalize", ["lowercase, NFKD", "strip punctuation"], False),
          ("02", "Blocking", ["8 key types", "country-scoped"], True),
          ("03", "Top-K pairs", ["rank by shared", "keys, K = 17"], False),
          ("04", "Features", ["20 name, address", "and structure"], False),
          ("05", "LightGBM", ["P(match) per pair", "split by S1"], True),
          ("06", "Threshold", ["t = 0.8", "empty list ok"], False)]
for i, (n, name, subs, acc) in enumerate(stages):
    x = 36 + 152 * i
    b += f'<rect x="{x}" y="120" width="128" height="104" rx="6" fill="{TINT if acc else "#FFFFFF"}" stroke="{ACC if acc else HAIR}" stroke-width="1"/>\n'
    b += T(x + 16, 144, n, 10, MONO, MUTED)
    b += T(x + 16, 168, name, 15, SANS, INK, weight="600")
    b += T(x + 16, 192, subs[0], 10, MONO, MUTED) + T(x + 16, 208, subs[1], 10, MONO, MUTED)
    if i < 5:
        b += f'<line x1="{x+130}" y1="172" x2="{x+150}" y2="172" stroke="{MUTED}" stroke-width="1" marker-end="url(#ar)"/>\n'
b += f'<path d="M188 240 V248 H468 V240" fill="none" stroke="{MUTED}" stroke-width="1"/>\n'
b += f'<path d="M492 240 V248 H924 V240" fill="none" stroke="{MUTED}" stroke-width="1"/>\n'
b += T(328, 272, "Stage A · retrieval sets the recall ceiling", 14, SERIF, INK, "middle", style="italic")
b += T(708, 272, "Stage B · matching decides precision", 14, SERIF, INK, "middle", style="italic")
b += T(36, 316, "RESULT  29,434,096 candidate pairs scored → 1,558,568 of 1,732,544 S1 entities with at least one match (90.0%)", 11, MONO, MUTED)
svg(960, 336, "Pipeline overview", "Normalize, multi-key blocking, top-K candidate pairs, feature building, LightGBM matcher, probability threshold.", b, "pipeline.svg")

# ---------- 2. recall@K ----------
b = head("Candidate recall vs. K", "50k TRAIN S1 SAMPLE  ·  173,213 TRUE LINKS  ·  AXIS STARTS AT 0.80")
data = [(17, 0.8456), (25, 0.8573), (50, 0.8739), (100, 0.8889), (200, 0.9033), (500, 0.9235)]
X = lambda v: 180 + (v - 0.80) * 4000
for tv in (0.80, 0.85, 0.90, 0.95):
    b += f'<line x1="{X(tv):.0f}" y1="112" x2="{X(tv):.0f}" y2="344" stroke="{HAIR}" stroke-width="1"/>\n'
    b += T(f"{X(tv):.0f}", 364, f"{tv:.2f}", 11, MONO, MUTED, "middle")
for i, (k, v) in enumerate(data):
    y = 136 + 40 * i; x = X(v); acc = (k == 17)
    b += T(36, y + 4, f"K = {k}", 12, MONO, ACC if acc else INK, weight="600" if acc else "400")
    b += f'<line x1="180" y1="{y}" x2="{x:.0f}" y2="{y}" stroke="{ACC if acc else INK}" stroke-width="1"/>\n'
    b += f'<circle cx="{x:.0f}" cy="{y}" r="5" fill="{ACC if acc else INK}"/>\n'
    b += T(f"{x+14:.0f}", y + 4, f"{v:.4f}", 12, MONO, INK)
b += T(448, 140, "used for the test submission", 14, SERIF, ACC, style="italic")
b += T(760, 336, "full blocking ceiling", 14, SERIF, MUTED, style="italic")
svg(960, 392, "Candidate recall versus K", "Share of true matches inside the top-K candidates per S1, rising from 0.846 at K=17 to 0.924 at K=500.", b, "recall_at_k.svg")

# ---------- 3. loss breakdown ----------
b = head("Where the 10,429 true matches go", "3,000 HELD-OUT S1  ·  BASELINE MODEL  ·  K = 17  ·  t = 0.7")
segs = [(635, MUTED, "75.5%"), (123, ACC, "14.7%"), (82, INK, "9.8%")]
x = 60
for w, c, lab in segs:
    b += f'<rect x="{x}" y="152" width="{w}" height="44" fill="{c}"/>\n'
    b += T(x + w / 2, 180, lab, 14, MONO, "#FFFFFF", "middle", "600")
    x += w
b += T(60, 244, "7,871 predicted correctly", 13, SANS, INK, weight="600")
b += f'<line x1="756" y1="196" x2="756" y2="284" stroke="{ACC}" stroke-width="1"/>\n'
b += T(748, 292, "1,535 never became candidates (blocking miss at K = 17)", 13, SANS, INK, "end")
b += f'<line x1="860" y1="196" x2="860" y2="332" stroke="{INK}" stroke-width="1"/>\n'
b += T(852, 340, "1,023 were candidates but scored below the threshold", 13, SANS, INK, "end")
b += T(60, 376, "Not shown: 433 false positives, a precision cost on top of these misses.", 14, SERIF, MUTED, style="italic")
svg(960, 400, "Where true matches are lost", "75.5 percent found, 14.7 percent missed by blocking, 9.8 percent rejected by the threshold.", b, "loss_breakdown.svg")

# ---------- 4. results ----------
b = head("Entity-level F0.5 on held-out validation", "MACRO PER S1, SINGLETONS COUNTED  ·  3,000 S1  ·  K = 17  ·  BEST THRESHOLD PER VARIANT")
rows = [("name features only", 0.6946, "0.695", False), ("address features only", 0.7174, "0.717", False),
        ("name + address (baseline)", 0.8552, "0.855", False),
        ("+ house-number features", 0.8839, "0.884 ± 0.001 (3 seeds)", True)]
S = 560
for tv in (0.0, 0.5, 1.0):
    xx = 260 + tv * S
    b += f'<line x1="{xx:.0f}" y1="108" x2="{xx:.0f}" y2="336" stroke="{HAIR}" stroke-width="1"/>\n'
    b += T(f"{xx:.0f}", 356, f"{tv:.1f}", 11, MONO, MUTED, "middle")
for i, (lab, v, txt, acc) in enumerate(rows):
    y = 124 + 56 * i
    b += T(36, y + 17, lab, 13, SANS, ACC if acc else INK, weight="600" if acc else "400")
    b += f'<rect x="260" y="{y}" width="{v*S:.0f}" height="24" fill="{ACC if acc else MUTED}"/>\n'
    b += T(f"{260+v*S+12:.0f}", y + 17, txt, 12, MONO, INK)
b += T(36, 388, "Last bar: post-deadline experiment on validation only; it is not part of the submitted file.", 14, SERIF, MUTED, style="italic")
svg(960, 400, "Entity-level F0.5 results", "Name only 0.695, address only 0.717, both 0.855, with house-number features 0.884.", b, "results.svg")
print("ok")
