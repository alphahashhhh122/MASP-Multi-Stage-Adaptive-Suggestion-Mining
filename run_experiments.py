"""Run the main MASP experiment and supporting analyses.

Usage:
    # Run everything (takes several hours)
    python run_experiments.py --all

    # Run specific experiment
    python run_experiments.py --main          # Table 2: MASP vs baselines
    python run_experiments.py --switch        # Table 4: switch analysis
"""

import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
TEST_DATA = Path("data/test.csv")
TRAIN_DATA = Path("data/train.csv")


def load_test_data():
    df = pd.read_csv(TEST_DATA)
    logger.info(f"Loaded {len(df)} test samples")
    return df


def compute_metrics(predictions: list) -> dict:
    tp = sum(1 for p in predictions if p["pred"] == 1 and p["gold"] == 1)
    fp = sum(1 for p in predictions if p["pred"] == 1 and p["gold"] == 0)
    fn = sum(1 for p in predictions if p["pred"] == 0 and p["gold"] == 1)
    tn = sum(1 for p in predictions if p["pred"] == 0 and p["gold"] == 0)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {
        "P": round(p, 4),
        "R": round(r, 4),
        "F1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": len(predictions),
    }


def bootstrap_ci(predictions: list, n_bootstrap=1000, alpha=0.05) -> tuple:
    """Compute bootstrap 95% CI for F1."""
    np.random.seed(42)
    f1s = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(predictions, size=len(predictions), replace=True)
        m = compute_metrics(list(sample))
        f1s.append(m["F1"])
    lower = np.percentile(f1s, 100 * alpha / 2)
    upper = np.percentile(f1s, 100 * (1 - alpha / 2))
    return round(lower, 4), round(upper, 4)


def paired_bootstrap_test(preds_a: list, preds_b: list, n_bootstrap=1000) -> float:
    """Paired bootstrap test: is system A better than system B?"""
    np.random.seed(42)
    wins = 0
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(preds_a), size=len(preds_a), replace=True)
        sample_a = [preds_a[i] for i in idx]
        sample_b = [preds_b[i] for i in idx]
        f1_a = compute_metrics(sample_a)["F1"]
        f1_b = compute_metrics(sample_b)["F1"]
        if f1_a > f1_b:
            wins += 1
    return round(1 - wins / n_bootstrap, 4)  # p-value


def run_main_experiment():
    """Table 2: MASP vs B1-B4."""
    logger.info("\n" + "=" * 60 + "\nRUNNING MAIN EXPERIMENT (Table 2)\n" + "=" * 60)
    df = load_test_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    systems = {}

    # Run MASP full pipeline
    logger.info("Running MASP (full)...")
    from main import run_pipeline
    # v5: description-first, no media files

    masp_preds = []
    for _, row in df.iterrows():
        mc_parts = []
        if row.get("has_image") and row.get("image_description"):
            mc_parts.append(f"Image: {row['image_description']}")
        if row.get("has_audio") and row.get("audio_description"):
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
        masp_preds.append(
            {
                "entry_id": row["entry_id"],
                "pred": 1 if ranked else 0,
                "gold": row["is_suggestion"],
                "path": row["extraction_path"],
            }
        )
    systems["MASP"] = masp_preds

    # Run baselines
    from baselines_comprehensive import baseline_rules, baseline_tfidf_logreg

    train_df = pd.read_csv(TRAIN_DATA)
    baseline_predictions = {
        "B1": baseline_rules(df),
        "B2": baseline_tfidf_logreg(train_df, df),
    }
    for name, raw_predictions in baseline_predictions.items():
        logger.info(f"Running {name}...")
        preds = [
            {
                "entry_id": row["entry_id"],
                "pred": int(prediction),
                "gold": row["is_suggestion"],
                "path": row["extraction_path"],
            }
            for (_, row), prediction in zip(df.iterrows(), raw_predictions)
        ]
        systems[name] = preds

    # B3: text-only (run MASP with no images/audio)
    logger.info("Running B3 (text-only MASP)...")
    b3_preds = []
    for _, row in df.iterrows():
        result = run_pipeline(text=row["raw_text"], sample_id=f"B3_{row['entry_id']}")
        ranked = result.get("ranked_suggestions", [])
        b3_preds.append(
            {
                "entry_id": row["entry_id"],
                "pred": 1 if ranked else 0,
                "gold": row["is_suggestion"],
                "path": row["extraction_path"],
            }
        )
    systems["B3"] = b3_preds

    # Print Table 2
    print("\n" + "=" * 60)
    print("TABLE 2: Main Results")
    print("=" * 60)
    print(f"{'System':<25} {'P':>6} {'R':>6} {'F1':>6} {'95% CI':>14}")
    print("-" * 60)
    for name, preds in systems.items():
        m = compute_metrics(preds)
        ci = bootstrap_ci(preds)
        print(
            f"{name:<25} {m['P']:>6.3f} {m['R']:>6.3f} {m['F1']:>6.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        )

    # Save
    with open(RESULTS_DIR / "table2_main.json", "w") as f:
        json.dump(
            {name: compute_metrics(preds) for name, preds in systems.items()},
            f,
            indent=2,
        )


def run_switch_analysis():
    """Compare COMMON and SPECIFIC routing modes."""
    logger.info("\n" + "=" * 60 + "\nRUNNING SWITCH ANALYSIS (Table 4)\n" + "=" * 60)
    # This needs actual pipeline outputs with alignment scores
    # Load from results if available
    results_file = RESULTS_DIR / "masp_full_outputs.json"
    if not results_file.exists():
        logger.error("Run main experiment first to generate pipeline outputs")
        return

    with open(results_file) as f:
        outputs = json.load(f)

    common_preds, specific_preds = [], []
    for output in outputs:
        cma = output.get("cross_modal_alignment", {})
        alignment = cma.get("overall_alignment", 1.0)
        ranked = output.get("ranked_suggestions", [])
        pred = 1 if ranked else 0
        gold = output.get("gold_is_suggestion", 0)
        entry = {"pred": pred, "gold": gold, "alignment": alignment}

        if alignment >= 0.6:
            common_preds.append(entry)
        else:
            specific_preds.append(entry)

    print("\n" + "=" * 60)
    print("TABLE 4: Switch Analysis")
    print("=" * 60)
    print(f"{'Mode':<20} {'#':>4} {'F1':>8}")
    print("-" * 36)

    for name, preds in [
        ("COMMON (a≥0.6)", common_preds),
        ("SPECIFIC (a<0.6)", specific_preds),
    ]:
        m = compute_metrics(preds)
        print(f"{name:<20} {m['n']:>4} {m['F1']:>8.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--main", action="store_true")
    parser.add_argument("--switch", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all or args.main:
        run_main_experiment()
    if args.all or args.switch:
        run_switch_analysis()


if __name__ == "__main__":
    main()
