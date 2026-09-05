#!/usr/bin/env python3
"""Evaluate the rule-based, classical ML, and model-assisted baselines.

B1: Rule-based keyword matching (SemEval-2019 style)
B2: TF-IDF + LogReg (pre-LLM classical ML)
B3: TF-IDF + SVM (pre-LLM classical ML)
B4: Zero-shot LLM prompt (single Gemma3 call, no pipeline)
B5: Text-only MASP (pipeline but no multimodal)

The classical baselines provide reference points for judging how much signal is
available without the multi-stage pipeline.

Usage:
    python baselines_comprehensive.py --dataset data/test.csv --output results/
"""

import pandas as pd
import json
import re
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# B1: Rule-based keyword matching (SemEval-2019 style)
# ════════════════════════════════════════════════════════════════
def baseline_rules(df):
    """Rule-based suggestion detection using published cue lists.

    Keywords sourced from:
      - Suggestion speech act cues: Negi & Buitelaar (2015), Table 2
      - Modal verb indicators: Ramanand et al. (2010)
      - Implicit complaint patterns: standard NLP patterns

    NO dataset-specific keywords. All cues from published literature.
    """
    # From Negi & Buitelaar (2015) Table 2 — suggestion speech act cues
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
    # Standard implicit complaint patterns (not dataset-specific)
    IMPLICIT_PATTERNS = [
        r"(crash|freeze|bug|error|slow|broken|fail|stuck)",
        r"(too small|too big|too slow|too fast)",
        r"(doesn\'t work|won\'t load|can\'t find|not working)",
    ]
    preds = []
    for _, row in df.iterrows():
        text = str(row["raw_text"]).lower()
        has_cue = any(kw in text for kw in NEGI_CUES)
        has_implicit = any(re.search(p, text) for p in IMPLICIT_PATTERNS)
        pred = 1 if (has_cue or has_implicit) else 0
        preds.append(pred)
    return preds


def baseline_tfidf_logreg(train_df, test_df):
    """Classical ML baseline: TF-IDF features + LogReg classifier."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000, ngram_range=(1, 2), sublinear_tf=True
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, C=1.0, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )

    pipe.fit(train_df["raw_text"].astype(str), train_df["is_suggestion"])
    preds = pipe.predict(test_df["raw_text"].astype(str))
    return preds.tolist()


# ════════════════════════════════════════════════════════════════
# B3: TF-IDF + SVM
# ════════════════════════════════════════════════════════════════
def baseline_tfidf_svm(train_df, test_df):
    """Classical ML baseline: TF-IDF + SVM."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000, ngram_range=(1, 2), sublinear_tf=True
                ),
            ),
            (
                "clf",
                LinearSVC(
                    max_iter=5000, C=1.0, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )

    pipe.fit(train_df["raw_text"].astype(str), train_df["is_suggestion"])
    preds = pipe.predict(test_df["raw_text"].astype(str))
    return preds.tolist()


# ════════════════════════════════════════════════════════════════
# B4: Zero-shot LLM (single prompt, no pipeline)
# ════════════════════════════════════════════════════════════════
def baseline_zeroshot_llm(df):
    """Single LLM call per sample — no pipeline, no multi-agent."""
    from llm_backend import call_llm

    preds = []
    for i, (_, row) in enumerate(df.iterrows()):
        text = str(row["raw_text"])
        try:
            result, _ = call_llm(
                "You are a suggestion detector. Respond ONLY with JSON.",
                f'Does this review contain an actionable suggestion? Review: "{text}"\nReturn: {{"is_suggestion": 0 or 1, "suggestion": "the suggestion or null"}}',
            )
            pred = result.get("is_suggestion", 0)
            preds.append(int(pred))
        except Exception as exc:
            log.warning("Rule baseline failed for one record: %s", exc)
            preds.append(0)

        if (i + 1) % 20 == 0:
            log.info(f"  B4 zero-shot: {i + 1}/{len(df)}")

    return preds


# ════════════════════════════════════════════════════════════════
# B5: Text-only MASP (pipeline without multimodal)
# ════════════════════════════════════════════════════════════════
def baseline_text_only_masp(df):
    """Full MASP pipeline but with multimodal_context stripped."""
    from main import run_pipeline

    preds = []
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            result = run_pipeline(
                text=str(row["raw_text"]),
                sample_id=f"B5_{row['entry_id']}",
                metadata={"domain": row.get("domain", "general")},
            )
            ranked = result.get("ranked_suggestions", [])
            preds.append(1 if ranked else 0)
        except Exception as exc:
            log.warning("Pipeline baseline failed for one record: %s", exc)
            preds.append(0)

        if (i + 1) % 20 == 0:
            log.info(f"  B5 text-only: {i + 1}/{len(df)}")

    return preds


# ════════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════════
def evaluate(y_true, y_pred, name):
    import math

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    acc = (tp + tn) / len(y_true) if y_true else 0
    mcc_num = tp * tn - fp * fn
    mcc_den = (
        math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0
        else 1
    )
    mcc = mcc_num / mcc_den
    bal_acc = 0.5 * (tp / (tp + fn) if tp + fn else 0) + 0.5 * (
        tn / (tn + fp) if tn + fp else 0
    )
    return {
        "name": name,
        "P": round(p, 4),
        "R": round(r, 4),
        "F1": round(f1, 4),
        "Acc": round(acc, 4),
        "MCC": round(mcc, 4),
        "BalAcc": round(bal_acc, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/test.csv")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--output", default="results")
    parser.add_argument(
        "--baseline",
        default="all",
        choices=["all", "classical", "llm", "B1", "B2", "B3", "B4", "B5"],
    )
    args = parser.parse_args()

    test = pd.read_csv(args.dataset)
    train = pd.read_csv(args.train)
    y_true = test["is_suggestion"].tolist()

    results = {}

    # Classical baselines (fast, no GPU)
    if args.baseline in ["all", "classical", "B1"]:
        log.info("Running B1: Rule-based keywords...")
        preds = baseline_rules(test)
        results["B1_rules"] = evaluate(y_true, preds, "Rule-based keywords")
        log.info(f"  B1: F1={results['B1_rules']['F1']:.3f}")

    if args.baseline in ["all", "classical", "B2"]:
        log.info("Running B2: TF-IDF + LogReg...")
        preds = baseline_tfidf_logreg(train, test)
        results["B2_logreg"] = evaluate(y_true, preds, "TF-IDF + LogReg")
        log.info(f"  B2: F1={results['B2_logreg']['F1']:.3f}")

    if args.baseline in ["all", "classical", "B3"]:
        log.info("Running B3: TF-IDF + SVM...")
        preds = baseline_tfidf_svm(train, test)
        results["B3_svm"] = evaluate(y_true, preds, "TF-IDF + SVM")
        log.info(f"  B3: F1={results['B3_svm']['F1']:.3f}")

    # LLM baselines (need GPU)
    if args.baseline in ["all", "llm", "B4"]:
        log.info("Running B4: Zero-shot LLM...")
        preds = baseline_zeroshot_llm(test)
        results["B4_zeroshot"] = evaluate(y_true, preds, "Zero-shot Gemma3-27B")
        log.info(f"  B4: F1={results['B4_zeroshot']['F1']:.3f}")

    if args.baseline in ["all", "llm", "B5"]:
        log.info("Running B5: Text-only MASP...")
        preds = baseline_text_only_masp(test)
        results["B5_textonly"] = evaluate(y_true, preds, "Text-only MASP")
        log.info(f"  B5: F1={results['B5_textonly']['F1']:.3f}")

    # Save
    Path(args.output).mkdir(exist_ok=True)
    with open(f"{args.output}/table2_main.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print table
    print(f"\n{'=' * 60}")
    print("BASELINE COMPARISON (Table 2)")
    print(f"{'=' * 60}")
    print(f"{'System':<25} {'P':>6} {'R':>6} {'F1':>6} {'MCC':>6} {'BalAcc':>6}")
    print("-" * 70)
    for name, m in results.items():
        print(
            f"{m['name']:<25} {m['P']:>6.3f} {m['R']:>6.3f} {m['F1']:>6.3f} {m['MCC']:>6.3f} {m['BalAcc']:>6.3f}"
        )

    print("\nDataset validation: If B1-B3 get F1 in [0.3, 0.6], dataset has")
    print("signal but isn't trivially solvable by classical methods.")


if __name__ == "__main__":
    main()
