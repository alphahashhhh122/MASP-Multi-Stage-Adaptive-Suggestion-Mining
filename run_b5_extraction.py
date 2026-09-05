#!/usr/bin/env python3
"""Run the B5 text-only MASP ablation with extraction logging.

This script is optional. It is provided because the paper reports B5 as a
detection ablation and explains that extraction strings were not logged in the
original run. If time permits, run this script on the same server used for MASP
to obtain a direct B5 extraction estimate.

Usage:
    python run_b5_extraction.py --dataset data/test.csv --output results/b5_extraction.csv
"""

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

from main import run_pipeline


def to_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def toks(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def overlap(pred, gold):
    gold_toks = toks(gold)
    if not gold_toks:
        return 0.0
    pred_toks = toks(pred)
    return len(pred_toks & gold_toks) / len(gold_toks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/test.csv")
    parser.add_argument("--output", default="results/b5_extraction.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in df.iterrows():
        result = run_pipeline(
            text=str(row["raw_text"]),
            sample_id=f"B5EXT_{row['entry_id']}",
            metadata={"domain": row.get("domain", "general")},
        )
        ranked = result.get("ranked_suggestions", []) or []
        pred_text = ranked[0].get("text", "") if ranked else ""
        pred = bool(pred_text)
        gold = to_bool(row.get("is_suggestion", False))
        ext = overlap(pred_text, row.get("suggestion_text", ""))
        rows.append(
            {
                "entry_id": row["entry_id"],
                "extraction_path": row.get("extraction_path", ""),
                "gold": int(gold),
                "pred": int(pred),
                "predicted_suggestion": pred_text,
                "gold_suggestion": row.get("suggestion_text", ""),
                "extraction_overlap": ext,
            }
        )
        if (i + 1) % 20 == 0:
            print(f"B5 extraction: {i + 1}/{len(df)}")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tp = sum(1 for r in rows if r["gold"] == 1 and r["pred"] == 1)
    fp = sum(1 for r in rows if r["gold"] == 0 and r["pred"] == 1)
    fn = sum(1 for r in rows if r["gold"] == 1 and r["pred"] == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    det_f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    ext_values = [
        r["extraction_overlap"] for r in rows if r["gold"] == 1 and r["pred"] == 1
    ]
    ext = sum(ext_values) / len(ext_values) if ext_values else 0.0
    mining_f1 = 2 * det_f1 * ext / (det_f1 + ext) if det_f1 + ext else 0.0
    print(
        f"P={precision:.3f} R={recall:.3f} DetectionF1={det_f1:.3f} Ext={ext:.3f} MiningF1={mining_f1:.3f}"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
