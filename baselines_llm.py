#!/usr/bin/env python3
"""Evaluate the model-based MASP baselines.

B4a: Minimal zero-shot (single short prompt)
B4b: Detailed zero-shot (definition + CoT + multimodal context)
B6:  BERT-style NLI zero-shot ("Does this entail a suggestion?")
B7:  Keyword + Sentiment (Negi 2015 + Hu & Liu 2004)

All prompts are documented with published sources.
No dataset-specific tuning.

Usage:
    python baselines_llm.py --dataset data/test.csv --baseline B4a
    python baselines_llm.py --dataset data/test.csv --baseline B4b
    python baselines_llm.py --dataset data/test.csv --baseline B7
    python baselines_llm.py --dataset data/test.csv --baseline all

Author: Anonymous
"""

import pandas as pd
import math
import time
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def evaluate(y_true, y_pred, name):
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
    print(f"\n{name}:")
    print(f"  P={p:.3f} R={r:.3f} F1={f1:.3f} MCC={mcc:.3f} BalAcc={bal:.3f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(
        f"  Predicted positive: {sum(y_pred)}/{len(y_pred)} ({sum(y_pred) / len(y_pred) * 100:.0f}%)"
    )
    return {"P": p, "R": r, "F1": f1, "MCC": mcc}


# ════════════════════════════════════════════════════════════════
# B4a: Minimal zero-shot
# ════════════════════════════════════════════════════════════════

B4A_SYSTEM = "You are a suggestion detector. Respond ONLY with JSON."
B4A_USER = 'Does this review contain an actionable suggestion? Review: "{text}"\nReturn: {{"is_suggestion": 0 or 1, "suggestion": "the suggestion or null"}}'


def baseline_b4a(df):
    """B4a: Minimal zero-shot — single short prompt, no examples."""
    from llm_backend import call_llm

    preds = []
    errors = 0
    for i, (_, row) in enumerate(df.iterrows()):
        text = str(row["raw_text"])
        try:
            result, _ = call_llm(B4A_SYSTEM, B4A_USER.format(text=text))
            preds.append(int(result.get("is_suggestion", 0)))
        except Exception as e:
            errors += 1
            preds.append(0)
            if errors <= 3:
                log.warning(f"  B4a ERROR: {type(e).__name__}: {str(e)[:80]}")
        if (i + 1) % 20 == 0:
            log.info(f"  B4a: {i + 1}/{len(df)} done | {errors} errors")
    return preds


# ════════════════════════════════════════════════════════════════
# B4b: Detailed zero-shot with CoT + multimodal context
# ════════════════════════════════════════════════════════════════

B4B_SYSTEM = """You are an expert suggestion mining system. Your task is to determine whether a customer review contains an actionable suggestion — something the business could implement to improve their product or service.

A suggestion can be:
- EXPLICIT: directly stated using words like "should", "could", "recommend", "please add", "it would be better if"
- IMPLICIT: implied through complaints, problems, or unmet needs that point to a specific improvement

NOT a suggestion:
- Pure praise ("This app is amazing!")
- Factual descriptions ("The restaurant is on Main Street")
- Vague complaints with no actionable direction ("This sucks")
- Sarcasm or ironic praise ("Oh great, another crash")
- Resolved issues ("They used to be slow but fixed it")

Think step by step:
1. Read the review carefully
2. Identify if there is a specific, actionable improvement implied or stated
3. If yes, extract the suggestion
4. If no, classify as not a suggestion

Respond ONLY with valid JSON: {"is_suggestion": 0 or 1, "reasoning": "brief explanation", "suggestion": "the suggestion text or null"}"""


def baseline_b4b(df):
    """B4b: Detailed zero-shot — full definition, CoT, multimodal context."""
    from llm_backend import call_llm

    preds = []
    errors = 0
    start = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        text = str(row["raw_text"])
        mc = str(row.get("multimodal_context", ""))
        if mc in ["", "nan", "—", "None"]:
            user_msg = f'Review: "{text}"\n\nDoes this review contain an actionable suggestion?'
        else:
            user_msg = (
                f'Review: "{text}"\n\nAdditional context '
                f"(image/audio description): {mc}\n\n"
                f"Does this review contain an actionable suggestion?"
            )
        try:
            result, _ = call_llm(B4B_SYSTEM, user_msg)
            preds.append(int(result.get("is_suggestion", 0)))
        except Exception as e:
            errors += 1
            preds.append(0)
            if errors <= 3:
                log.warning(f"  B4b ERROR: {type(e).__name__}: {str(e)[:80]}")
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            eta = (len(df) - i - 1) / ((i + 1) / elapsed) / 60
            log.info(
                f"  B4b: {i + 1}/{len(df)} done | {errors} errors | ETA: {eta:.1f} min"
            )
    return preds


# ════════════════════════════════════════════════════════════════
# B6: BERT-style NLI zero-shot
# ════════════════════════════════════════════════════════════════

B6_SYSTEM = """You are a text classifier. Your task is to determine if the given text entails a suggestion. Respond ONLY with valid JSON: {"label": 0 or 1}

Label 1 = the text contains or implies a suggestion for improvement.
Label 0 = the text does not contain a suggestion."""


def baseline_b6(df):
    """B6: BERT-style NLI — "Does this text entail a suggestion?" """
    from llm_backend import call_llm

    preds = []
    errors = 0
    for i, (_, row) in enumerate(df.iterrows()):
        text = str(row["raw_text"])
        try:
            result, _ = call_llm(B6_SYSTEM, f'Text: "{text}"')
            preds.append(int(result.get("label", 0)))
        except Exception as e:
            errors += 1
            preds.append(0)
            if errors <= 3:
                log.warning(f"  B6 ERROR: {type(e).__name__}: {str(e)[:80]}")
        if (i + 1) % 20 == 0:
            log.info(f"  B6: {i + 1}/{len(df)} done | {errors} errors")
    return preds


# ════════════════════════════════════════════════════════════════
# B7: Keyword + Sentiment (published sources only)
# ════════════════════════════════════════════════════════════════


def baseline_b7(df):
    """B7: Keyword + Sentiment heuristic.

    Sources:
      - Suggestion cues: Negi & Buitelaar (2015), Table 2
      - Sentiment lexicon: Hu & Liu (2004), opinion lexicon (negative subset)
      - Classification rule: Ramanand et al. (2010), Section 3

    Rule: suggestion if (has_cue OR has_negative_sentiment).
    NO dataset-specific keywords.
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
    for _, row in df.iterrows():
        tl = str(row["raw_text"]).lower()
        has_cue = any(kw in tl for kw in NEGI_CUES)
        has_neg = any(kw in tl for kw in HU_LIU_NEG)
        preds.append(1 if has_cue or has_neg else 0)
    return preds


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="LLM baselines for MASP")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--baseline", required=True, choices=["B4a", "B4b", "B6", "B7", "all"]
    )
    args = parser.parse_args()

    test = pd.read_csv(args.dataset)
    y_true = test["is_suggestion"].tolist()

    baselines = {
        "B4a": ("B4a: Zero-shot minimal", baseline_b4a),
        "B4b": ("B4b: Zero-shot detailed+CoT", baseline_b4b),
        "B6": ("B6: BERT-style NLI", baseline_b6),
        "B7": ("B7: Keyword+Sentiment (Negi 2015 + Hu&Liu 2004)", baseline_b7),
    }

    to_run = baselines.keys() if args.baseline == "all" else [args.baseline]

    for key in to_run:
        name, fn = baselines[key]
        log.info(f"Running {name}...")
        preds = fn(test)
        evaluate(y_true, preds, name)
        # Save predictions
        pd.DataFrame({"entry_id": test["entry_id"], f"{key}_pred": preds}).to_csv(
            f"results/{key.lower()}_preds.csv", index=False
        )
        log.info(f"Saved: results/{key.lower()}_preds.csv")


if __name__ == "__main__":
    main()
