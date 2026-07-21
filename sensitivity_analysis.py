"""
sensitivity_analysis.py — Sweep tau threshold, generate Figure 4 data.

Usage:
    python sensitivity_analysis.py --tau_values 0.3 0.4 0.5 0.6 0.7 0.8 0.9
"""
import json, argparse, logging, time
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

def run_with_tau(tau: float, test_df: pd.DataFrame) -> dict:
    """Run full pipeline with specific tau value."""
    from llm_backend import set_config
    from graph.pipeline import build_graph
    from main import run_pipeline
    
    # Override tau in the pipeline
    # Note: you'll need to wire config.yaml's tau into cross_modal_align_node
    set_config(tau=tau)
    
        # Pipeline built inside run_pipeline()
    # v5: description-first, no media files
    
    preds = []
    for _, row in test_df.iterrows():
        # v5: description-first — build metadata from CSV columns
        mc_parts = []
        if row.get("has_image") == True and row.get("image_description"):
            mc_parts.append(f"Image: {row['image_description']}")
        if row.get("has_audio") == True and row.get("audio_description"):
            mc_parts.append(f"Audio: {row['audio_description']}")
        metadata = {
            "domain": row.get("domain", "general"),
            "image_description": row.get("image_description", ""),
            "audio_description": row.get("audio_description", ""),
            "multimodal_context": " | ".join(mc_parts),
        }
        result = run_pipeline(
            text=str(row["raw_text"]),
            sample_id=row["entry_id"],
            metadata=metadata,
        )
        ranked = result.get("ranked_suggestions", [])
        cma = result.get("cross_modal_alignment", {})
        preds.append({
            "entry_id": row["entry_id"],
            "pred": 1 if ranked else 0,
            "gold": row["is_suggestion"],
            "path": row["extraction_path"],
            "alignment": cma.get("overall_alignment", 1.0),
            "switch_mode": "COMMON" if cma.get("overall_alignment", 1.0) >= tau else "SPECIFIC",
        })
    
    tp = sum(1 for p in preds if p["pred"]==1 and p["gold"]==1)
    fp = sum(1 for p in preds if p["pred"]==1 and p["gold"]==0)
    fn = sum(1 for p in preds if p["pred"]==0 and p["gold"]==1)
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    
    n_common = sum(1 for p in preds if p["switch_mode"]=="COMMON")
    n_specific = sum(1 for p in preds if p["switch_mode"]=="SPECIFIC")
    
    return {"tau": tau, "P": round(p,4), "R": round(r,4), "F1": round(f1,4),
            "n_common": n_common, "n_specific": n_specific, "n_total": len(preds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau_values", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--test_data", default="../data/test.csv")
    parser.add_argument("--output", default="../results/sensitivity.json")
    args = parser.parse_args()
    
    df = pd.read_csv(args.test_data)
    logger.info(f"Loaded {len(df)} test samples")
    
    results = []
    for tau in args.tau_values:
        logger.info(f"\nRunning with tau={tau}...")
        t0 = time.time()
        r = run_with_tau(tau, df)
        r["time_seconds"] = round(time.time()-t0, 1)
        results.append(r)
        logger.info(f"  tau={tau}: F1={r['F1']:.3f} COMMON={r['n_common']} SPECIFIC={r['n_specific']}")
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print table for paper
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS — Figure 4 data")
    print("="*70)
    print(f"{'tau':>6} {'F1':>8} {'P':>8} {'R':>8} {'COMMON':>8} {'SPECIFIC':>10}")
    print("-"*50)
    for r in results:
        print(f"{r['tau']:>6.1f} {r['F1']:>8.3f} {r['P']:>8.3f} {r['R']:>8.3f} {r['n_common']:>8} {r['n_specific']:>10}")
    
    best = max(results, key=lambda x: x["F1"])
    print(f"\nOptimal tau={best['tau']} with F1={best['F1']:.3f}")

if __name__ == "__main__":
    main()
