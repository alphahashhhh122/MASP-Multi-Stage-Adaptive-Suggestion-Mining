#!/usr/bin/env python3
"""B6: Single-prompt multimodal baseline using same Gemma3 model.
Tests whether the multi-stage pipeline architecture is justified versus one call."""

import csv
import json
import time
import re
import argparse
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma3:27b-it-qat"

PROMPT = """You are analyzing a customer review that may contain an actionable suggestion.

REVIEW TEXT:
{text}

{image_section}
{audio_section}

TASK: Determine if this review contains an actionable suggestion.
A suggestion must identify a specific problem AND imply or state a concrete improvement.
A pure complaint without constructive direction is NOT a suggestion.

Respond ONLY in this JSON format, nothing else:
{{"is_suggestion": true or false, "confidence": 0.0 to 1.0, "suggestion_text": "extracted suggestion or empty string"}}"""


def run_one(text, img="", aud=""):
    prompt = PROMPT.format(
        text=text,
        image_section=f"IMAGE DESCRIPTION:\n{img}" if img.strip() else "",
        audio_section=f"AUDIO PROSODY:\n{aud}" if aud.strip() else "",
    )
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 200},
            },
            timeout=180,
        )
        raw = resp.json().get("response", "")
        match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if match:
            d = json.loads(match.group())
            v = d.get("is_suggestion", False)
            if isinstance(v, str):
                v = v.lower() == "true"
            return v, d.get("suggestion_text", "")
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        pass
    return False, ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/test.csv")
    p.add_argument("--output", default="results/baseline_b6/")
    a = p.parse_args()
    Path(a.output).mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(a.csv)))
    tp = fp = fn = tn = 0
    results = []

    print(f"B6 single-prompt baseline on {len(rows)} samples...")
    for i, r in enumerate(rows):
        gold = r["is_suggestion"].strip().lower() == "true"
        pred, sug = run_one(
            r["raw_text"],
            r.get("image_description", ""),
            r.get("audio_description", ""),
        )

        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1
        results.append(
            {"entry_id": r["entry_id"], "gold": gold, "pred": pred, "suggestion": sug}
        )

        if (i + 1) % 50 == 0:
            pr = tp / (tp + fp) if tp + fp else 0
            rc = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
            print(f"  [{i + 1}/{len(rows)}] P={pr:.3f} R={rc:.3f} F1={f1:.3f}")
        time.sleep(0.5)

    pr = tp / (tp + fp) if tp + fp else 0
    rc = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
    print(f"\nB6 RESULTS: P={pr:.3f} R={rc:.3f} F1={f1:.3f}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")

    with open(f"{a.output}/b6_results.json", "w") as f:
        json.dump(
            {
                "precision": pr,
                "recall": rc,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved to {a.output}/b6_results.json")


if __name__ == "__main__":
    main()
