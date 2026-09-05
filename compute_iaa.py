#!/usr/bin/env python3
"""IAA: python compute_iaa.py --export  |  python compute_iaa.py --compute --annotator2 data/annotator2.csv"""

import argparse
import pandas as pd


def export_for_annotation():
    df = pd.read_csv("data/MASP_Dataset_v5_2500_FINAL.csv")
    seed = df[~df["entry_id"].str.startswith("GEN")]
    samples = seed[seed["extraction_path"].isin(["P3", "P4", "P5", "P6", "P7", "P8"])]
    if len(samples) < 40:
        extra = seed[seed["extraction_path"].isin(["HN", "P2"])].sample(
            min(40 - len(samples), 15), random_state=42
        )
        samples = pd.concat([samples, extra])
    export = samples[
        [
            "entry_id",
            "extraction_path",
            "domain",
            "raw_text",
            "has_image",
            "image_description",
            "has_audio",
            "audio_description",
        ]
    ].copy()
    export["annotator2_is_suggestion"] = ""
    export["annotator2_suggestion_type"] = ""
    export.to_csv("data/iaa_annotation_sheet.csv", index=False)
    print(f"Exported {len(export)} samples to data/iaa_annotation_sheet.csv")


def compute(a2_path):
    df = pd.read_csv("data/MASP_Dataset_v5_2500_FINAL.csv")
    a2 = pd.read_csv(a2_path)
    m = df.merge(
        a2[["entry_id", "annotator2_is_suggestion"]], on="entry_id", how="inner"
    )
    m = m[m["annotator2_is_suggestion"].notna()]
    m["annotator2_is_suggestion"] = m["annotator2_is_suggestion"].astype(int)
    from sklearn.metrics import cohen_kappa_score

    k = cohen_kappa_score(
        m["is_suggestion"].values, m["annotator2_is_suggestion"].values
    )
    agree = (m["is_suggestion"] == m["annotator2_is_suggestion"]).sum()
    print(f"Cohen's kappa: {k:.3f} ({agree}/{len(m)} agree)")
    if k >= 0.70:
        print("✓ Substantial agreement")
    elif k >= 0.60:
        print("⚠ Moderate (borderline)")
    else:
        print("❌ Weak — revise guidelines")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--export", action="store_true")
    p.add_argument("--compute", action="store_true")
    p.add_argument("--annotator2", default="data/annotator2.csv")
    a = p.parse_args()
    if a.export:
        export_for_annotation()
    elif a.compute:
        compute(a.annotator2)
    else:
        export_for_annotation()
