#!/usr/bin/env python3
"""Compare MASP and SemEval suggestion-mining baselines.

Proves two things:
  1. MASP benchmark difficulty is comparable to SemEval-2019 Task 9
     (same classical methods → similar F1 on both datasets)
  2. Prior methods don't trivially solve our dataset
     (justifies building MASP pipeline)

Baselines are sourced ENTIRELY from published literature:
  - B1: Rule-based cues from Negi & Buitelaar (2015), Table 2
  - B2: TF-IDF + LogReg — standard text classification baseline
         (SemEval-2019 Task 9 organiser baseline)
  - B7: Keyword+Sentiment — cues from Negi & Buitelaar (2015),
         sentiment lexicon from Hu & Liu (2004), classification
         rule following Ramanand et al. (2010)

SemEval-2019 Task 9 winner: BERT fine-tuned, F1=0.8579 (Subtask A)
  Source: Negi et al. (2019) "SemEval-2019 Task 9"

Data format:
  SemEval: CSV with columns [id, text, label], no header row
  MASP:    CSV with columns [entry_id, raw_text, is_suggestion, ...]

Usage:
    python3 eval_semeval_cross.py

Output:
    Cross-benchmark comparison table (stdout + saved to results/)

Author: Anonymous
"""

import pandas as pd
import math
import re
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════


def evaluate(y_true, y_pred):
    """Compute P, R, F1, MCC, BalAcc."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(den) if den > 0 else 0
    bal = 0.5 * (tp / (tp + fn) if tp + fn else 0) + 0.5 * (
        tn / (tn + fp) if tn + fp else 0
    )
    return {
        "P": round(p, 4),
        "R": round(r, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
        "BalAcc": round(bal, 4),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "n": len(y_true),
        "pos": sum(y_true),
        "neg": len(y_true) - sum(y_true),
    }


# ════════════════════════════════════════════════════════════════
# BASELINES — all keyword lists from published sources
# ════════════════════════════════════════════════════════════════


def baseline_b1_rules(texts):
    """B1: Rule-based suggestion cues.
    Source: Negi & Buitelaar (2015) Table 2 — suggestion speech act cues.
           Ramanand et al. (2010) — modal verb indicators.
    """
    NEGI_CUES = [
        "suggest",
        "recommend",
        "advise",
        "propose",
        "should",
        "could",
        "would",
        "might",
        "ought",
        "why not",
        "how about",
        "what about",
        "better if",
        "would be nice",
        "wish",
        "hope",
        "need to",
        "have to",
        "must",
        "please",
        "try to",
        "consider",
        "it would help",
        "you may want",
    ]
    IMPLICIT = [
        r"(crash|freeze|bug|error|slow|broken|fail|stuck)",
        r"(too small|too big|too slow|too fast)",
        r"(doesn't work|won't load|can't find|not working)",
    ]
    preds = []
    for text in texts:
        tl = text.lower()
        has_cue = any(kw in tl for kw in NEGI_CUES)
        has_implicit = any(re.search(p, tl) for p in IMPLICIT)
        preds.append(1 if has_cue or has_implicit else 0)
    return preds


def baseline_b2_tfidf(train_texts, train_labels, test_texts):
    """B2: TF-IDF + Logistic Regression.
    Standard text classification baseline. Used as organiser baseline
    in SemEval-2019 Task 9 (Negi et al. 2019, Section 4).
    """
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000, ngram_range=(1, 2), sublinear_tf=True
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
        ]
    )
    pipe.fit(train_texts, train_labels)
    return pipe.predict(test_texts).tolist()


def baseline_b7_keyword_sentiment(texts):
    """B7: Keyword + Sentiment heuristic.
    Suggestion cues: Negi & Buitelaar (2015) Table 2.
    Sentiment lexicon: Hu & Liu (2004) opinion lexicon (negative subset).
    Classification rule: Ramanand et al. (2010) Section 3.
    Rule: suggestion if (has_cue OR has_negative_sentiment).
    """
    NEGI_CUES = [
        "suggest",
        "recommend",
        "advise",
        "propose",
        "should",
        "could",
        "would",
        "might",
        "ought",
        "why not",
        "how about",
        "what about",
        "better if",
        "would be nice",
        "wish",
        "hope",
        "need to",
        "have to",
        "must",
        "please",
        "try to",
        "consider",
        "it would help",
        "you may want",
    ]
    HU_LIU_NEG = [
        "bad",
        "terrible",
        "awful",
        "horrible",
        "poor",
        "worst",
        "annoying",
        "frustrating",
        "disappointed",
        "disappointing",
        "useless",
        "waste",
        "mediocre",
        "inferior",
        "unacceptable",
        "defective",
        "faulty",
        "unreliable",
        "inconvenient",
    ]
    preds = []
    for text in texts:
        tl = text.lower()
        has_cue = any(kw in tl for kw in NEGI_CUES)
        has_neg = any(kw in tl for kw in HU_LIU_NEG)
        preds.append(1 if has_cue or has_neg else 0)
    return preds


# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════


def load_semeval(base_dir):
    """Load SemEval-2019 Task 9 Subtask A data."""
    base = Path(base_dir) / "Subtask-A-master"
    if not base.exists():
        base = Path(base_dir)

    train_path = base / "V1.4_Training.csv"
    test_path = base / "SubtaskA_EvaluationData_labeled.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"SemEval data not found at {base}. "
            f"Expected V1.4_Training.csv and SubtaskA_EvaluationData_labeled.csv"
        )

    # SemEval CSVs: columns are [id, text, label], no header
    train = pd.read_csv(train_path, header=None, names=["id", "text", "label"])
    test = pd.read_csv(test_path, header=None, names=["id", "text", "label"])

    # Clean
    train = train.dropna(subset=["text", "label"])
    test = test.dropna(subset=["text", "label"])
    train["label"] = train["label"].astype(int)
    test["label"] = test["label"].astype(int)

    return train, test


def load_masp():
    """Load MASP dataset."""
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    return train, test


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════


def main():
    print("=" * 74)
    print("CROSS-BENCHMARK VALIDATION: SemEval-2019 Task 9 vs MASP")
    print("=" * 74)

    # ── Load SemEval ──
    semeval_train, semeval_test = load_semeval("data/semeval")
    log.info(
        f"SemEval train: {len(semeval_train)} samples "
        f"({semeval_train['label'].sum()} pos, "
        f"{len(semeval_train) - semeval_train['label'].sum()} neg)"
    )
    log.info(
        f"SemEval test:  {len(semeval_test)} samples "
        f"({semeval_test['label'].sum()} pos, "
        f"{len(semeval_test) - semeval_test['label'].sum()} neg)"
    )

    # ── Load MASP ──
    masp_train, masp_test = load_masp()
    log.info(
        f"MASP train: {len(masp_train)} samples "
        f"({masp_train['is_suggestion'].sum()} pos)"
    )
    log.info(
        f"MASP test:  {len(masp_test)} samples ({masp_test['is_suggestion'].sum()} pos)"
    )

    # ── Run baselines on BOTH datasets ──
    results = {}

    for dataset_name, train_df, test_df, text_col, label_col in [
        ("SemEval-2019", semeval_train, semeval_test, "text", "label"),
        ("MASP (ours)", masp_train, masp_test, "raw_text", "is_suggestion"),
    ]:
        print(f"\n{'─' * 74}")
        print(f"Dataset: {dataset_name} (test n={len(test_df)})")
        print(f"{'─' * 74}")

        texts = test_df[text_col].astype(str).tolist()
        labels = test_df[label_col].astype(int).tolist()
        train_texts = train_df[text_col].astype(str).tolist()
        train_labels = train_df[label_col].astype(int).tolist()

        for bname, fn in [
            ("B1: Rule-based", lambda: baseline_b1_rules(texts)),
            (
                "B2: TF-IDF+LogReg",
                lambda: baseline_b2_tfidf(train_texts, train_labels, texts),
            ),
            ("B7: Keyword+Sent", lambda: baseline_b7_keyword_sentiment(texts)),
        ]:
            preds = fn()
            r = evaluate(labels, preds)
            results[(dataset_name, bname)] = r
            print(
                f"  {bname:<22} P={r['P']:.3f} R={r['R']:.3f} "
                f"F1={r['F1']:.3f} MCC={r['MCC']:.3f}"
            )

    # ── Cross-benchmark comparison table ──
    print(f"\n{'=' * 74}")
    print("CROSS-BENCHMARK TABLE (for paper)")
    print(f"{'=' * 74}")
    print(
        f"{'Method':<22} | {'SemEval F1':>10} {'MCC':>6} | "
        f"{'MASP F1':>8} {'MCC':>6} | {'F1 diff':>7}"
    )
    print("-" * 74)

    for bname in ["B1: Rule-based", "B2: TF-IDF+LogReg", "B7: Keyword+Sent"]:
        se = results.get(("SemEval-2019", bname), {})
        ma = results.get(("MASP (ours)", bname), {})
        diff = ma.get("F1", 0) - se.get("F1", 0)
        print(
            f"{bname:<22} | {se.get('F1', 0):>10.3f} {se.get('MCC', 0):>6.3f} | "
            f"{ma.get('F1', 0):>8.3f} {ma.get('MCC', 0):>6.3f} | {diff:>+7.3f}"
        )

    # Published reference
    print(
        f"{'SemEval-2019 Winner':<22} | {'0.858':>10} {'---':>6} | "
        f"{'---':>8} {'---':>6} |"
    )

    print("\nInterpretation:")
    print("  If |F1 diff| < 0.15 → datasets have comparable difficulty")
    print("  If MASP B1 F1 in [0.3, 0.6] → dataset valid, not trivially solvable")

    # ── Save results ──
    import json

    out = {}
    for (ds, bn), r in results.items():
        out[f"{ds}__{bn}"] = r
    Path("results").mkdir(exist_ok=True)
    with open("results/cross_benchmark_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/cross_benchmark_results.json")


if __name__ == "__main__":
    main()
