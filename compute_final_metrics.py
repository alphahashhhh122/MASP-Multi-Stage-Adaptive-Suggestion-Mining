"""
MASP EMNLP 2026 — FINAL METRICS COMPUTATION
Run ONLY on test_FINAL_v5 results (post all pipeline fixes).
All numbers in this file are the DEFINITIVE paper numbers.
"""

import csv
import math
import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def stem(word):
    word = word.lower()
    for suffix in [
        "ing",
        "tion",
        "ment",
        "ness",
        "able",
        "ible",
        "ful",
        "less",
        "ous",
        "ive",
        "ly",
        "ed",
        "er",
        "es",
        "s",
    ]:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def stemmed_overlap(pred_text, gold_text):
    pred_stems = set(
        stem(w) for w in re.findall(r"\w+", pred_text.lower()) if len(w) > 2
    )
    gold_stems = set(
        stem(w) for w in re.findall(r"\w+", gold_text.lower()) if len(w) > 2
    )
    if not pred_stems or not gold_stems:
        return 0
    return len(pred_stems & gold_stems) / min(len(pred_stems), len(gold_stems))


def compute_metrics(subset):
    tp = sum(1 for p in subset if p["pred"] == 1 and p["gold"] == 1)
    fp = sum(1 for p in subset if p["pred"] == 1 and p["gold"] == 0)
    fn = sum(1 for p in subset if p["pred"] == 0 and p["gold"] == 1)
    tn = sum(1 for p in subset if p["pred"] == 0 and p["gold"] == 0)
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    F1 = 2 * P * R / (P + R) if P + R else 0
    mcc_num = tp * tn - fp * fn
    mcc_den = (
        math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0
        else 1
    )
    mcc = mcc_num / mcc_den
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "P": P,
        "R": R,
        "F1": F1,
        "MCC": mcc,
        "n": len(subset),
    }


# Load results
with open("results/test_FINAL_v5/pipeline_results.csv") as f:
    results = list(csv.DictReader(f))
with open("data/test.csv") as f:
    gold = {r["entry_id"]: r for r in csv.DictReader(f)}

preds = []
for r in results:
    g = gold.get(r["entry_id"], {})
    preds.append(
        {
            "entry_id": r["entry_id"],
            "pred": 1 if int(r.get("num_suggestions", 0)) > 0 else 0,
            "gold": 1 if g.get("is_suggestion") == "True" else 0,
            "domain": r.get("domain", ""),
            "path": g.get("extraction_path", ""),
            "combo": r.get("modality_combo", ""),
            "sug_type": g.get("suggestion_type", ""),
            "top_suggestion": r.get("top_suggestion", ""),
            "gold_suggestion": g.get("suggestion_text", ""),
            "top_score": float(r.get("top_score", 0) or 0),
        }
    )

print("=" * 70)
print("MASP FINAL RESULTS — test_FINAL_v5")
print("(All pipeline fixes applied: audio ctx + SPECIFIC override + tokenizer)")
print("=" * 70)

# OVERALL
m = compute_metrics(preds)
print(f"\nOVERALL: P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} MCC={m['MCC']:.3f}")
print(f"  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']} N={m['n']}")

# BOOTSTRAP CI
random.seed(42)
n = len(preds)
f1s = []
for _ in range(1000):
    sample = [preds[random.randint(0, n - 1)] for _ in range(n)]
    f1s.append(compute_metrics(sample)["F1"])
f1s.sort()
print(f"\nBOOTSTRAP 95% CI: F1={f1s[500]:.3f} [{f1s[25]:.3f}, {f1s[974]:.3f}]")

# PER PATH
print("\nPER-PATH:")
for path in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "HN"]:
    s = [p for p in preds if p["path"] == path]
    if s:
        m2 = compute_metrics(s)
        print(
            f"  {path}: n={m2['n']:>3} TP={m2['tp']:>3} FP={m2['fp']:>3} FN={m2['fn']:>3} R={m2['R']:.3f} F1={m2['F1']:.3f}"
        )

# PER MODALITY
print("\nPER-MODALITY:")
for c in ["T", "T+I", "T+A", "T+I+A"]:
    s = [p for p in preds if p["combo"] == c]
    if s:
        m2 = compute_metrics(s)
        print(
            f"  {c}: F1={m2['F1']:.3f} P={m2['P']:.3f} R={m2['R']:.3f} MCC={m2['MCC']:.3f}"
        )

# PER TYPE
print("\nPER-TYPE:")
for stype in ["explicit", "implicit"]:
    s = [p for p in preds if p["sug_type"] == stype]
    if s:
        tp = sum(1 for p in s if p["pred"] == 1)
        fn = sum(1 for p in s if p["pred"] == 0)
        R = tp / (tp + fn) if tp + fn else 0
        print(f"  {stype}: {tp}/{tp + fn} R={R:.3f}")

# EXTRACTION QUALITY
overlaps = []
overlaps_p578 = []
for p in preds:
    if (
        p["pred"] == 1
        and p["gold"] == 1
        and p["top_suggestion"]
        and p["gold_suggestion"]
    ):
        ov = stemmed_overlap(p["top_suggestion"], p["gold_suggestion"])
        overlaps.append(ov)
        if p["path"] in ["P5", "P7", "P8"]:
            overlaps_p578.append(ov)
overlaps.sort()
print("\nEXTRACTION QUALITY:")
print(
    f"  All TP: mean={sum(overlaps) / len(overlaps):.3f} median={overlaps[len(overlaps) // 2]:.3f} n={len(overlaps)}"
)
if overlaps_p578:
    print(
        f"  P5/P7/P8: mean={sum(overlaps_p578) / len(overlaps_p578):.3f} n={len(overlaps_p578)}"
    )

# MINING F1
det_f1 = compute_metrics(preds)["F1"]
ext_quality = sum(overlaps) / len(overlaps)
mining_f1 = (
    2 * det_f1 * ext_quality / (det_f1 + ext_quality) if det_f1 + ext_quality else 0
)
print(f"\nMINING F1: Det={det_f1:.3f} Ext={ext_quality:.3f} Mining={mining_f1:.3f}")

# BASELINES COMPARISON (recompute on same splits)
with open("data/train.csv") as f:
    train = list(csv.DictReader(f))
with open("data/test.csv") as f:
    test_data = list(csv.DictReader(f))

train_texts = [r["raw_text"] for r in train]
train_labels = [1 if r["is_suggestion"] == "True" else 0 for r in train]
test_texts = [r["raw_text"] for r in test_data]
test_labels = [1 if r["is_suggestion"] == "True" else 0 for r in test_data]

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = tfidf.fit_transform(train_texts)
X_test = tfidf.transform(test_texts)

# B3: SVM
svm = LinearSVC(max_iter=5000, C=1.0)
svm.fit(X_train, train_labels)
svm_preds = svm.predict(X_test).tolist()
svm_tp = sum(1 for p, g in zip(svm_preds, test_labels) if p == 1 and g == 1)
svm_fp = sum(1 for p, g in zip(svm_preds, test_labels) if p == 1 and g == 0)
svm_fn = sum(1 for p, g in zip(svm_preds, test_labels) if p == 0 and g == 1)
svm_P = svm_tp / (svm_tp + svm_fp) if svm_tp + svm_fp else 0
svm_R = svm_tp / (svm_tp + svm_fn) if svm_tp + svm_fn else 0
svm_F1 = 2 * svm_P * svm_R / (svm_P + svm_R) if svm_P + svm_R else 0

# Heuristic extraction for baselines
KEYWORDS = [
    "should",
    "could",
    "need",
    "wish",
    "would",
    "please",
    "hope",
    "suggest",
    "add",
    "fix",
    "improve",
    "better",
    "more",
]
baseline_overlaps = []
for r in test_data:
    if r["is_suggestion"] != "True":
        continue
    text = r["raw_text"]
    gold_sug = r.get("suggestion_text", "")
    if not gold_sug:
        continue
    sentences = re.split(r"[.!?]+", text)
    best_sent = ""
    best_score = -1
    for s in sentences:
        score = sum(1 for k in KEYWORDS if k in s.lower())
        if score > best_score:
            best_score = score
            best_sent = s.strip()
    if best_sent:
        baseline_overlaps.append(stemmed_overlap(best_sent, gold_sug))
baseline_ext = (
    sum(baseline_overlaps) / len(baseline_overlaps) if baseline_overlaps else 0
)

print(f"\n{'=' * 70}")
print("MAIN RESULTS TABLE")
print(f"{'=' * 70}")
print(f"{'System':<25} {'Det F1':>8} {'Ext':>8} {'Mining F1':>10}")
print("-" * 55)
print(
    f"{'B3 TF-IDF+SVM':<25} {svm_F1:>8.3f} {baseline_ext:>8.3f} {2 * svm_F1 * baseline_ext / (svm_F1 + baseline_ext):>10.3f}"
)
print(f"{'MASP (ours)':<25} {det_f1:>8.3f} {ext_quality:>8.3f} {mining_f1:>10.3f}")

print(f"\n{'=' * 70}")
print("SAVE THIS FILE AS THE DEFINITIVE PAPER NUMBERS")
