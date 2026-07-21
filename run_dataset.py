#!/usr/bin/env python3
"""
run_dataset.py — Run MASP pipeline on the v5 dataset.

Description-first approach: no .jpg or .wav files needed.
Image and audio information comes from dataset columns:
  - image_description: VLM-quality caption (pipeline input)
  - audio_description: structured speech analysis (pipeline input)

Usage:
    # Run full dataset
    python run_dataset.py --csv ../data/MASP_Dataset_v5.csv --output ../results/

    # Run with sample limit
    python run_dataset.py --csv ../data/MASP_Dataset_v5.csv --max-samples 50

    # Run single review
    python run_dataset.py --text "The app crashes frequently" --domain tech_software
"""

import os
import sys
import json
import time
import argparse
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

from main import run_pipeline, print_results


def build_metadata_from_row(row):
    """Build pipeline metadata dict from a v5 CSV row.
    
    Constructs multimodal_context from image_description and audio_description
    columns, matching the format expected by run_pipeline() in main.py.
    """
    domain = row.get("domain", "general")
    img_desc = str(row.get("image_description", "")).strip()
    aud_desc = str(row.get("audio_description", "")).strip()
    
    if img_desc in ("", "nan", "None", "—"):
        img_desc = ""
    if aud_desc in ("", "nan", "None", "—"):
        aud_desc = ""
    
    mc_parts = []
    if img_desc:
        mc_parts.append(f"Image: {img_desc}")
    if aud_desc:
        mc_parts.append(f"Audio: {aud_desc}")
    
    return {
        "domain": domain,
        "entry_id": row.get("entry_id", ""),
        "extraction_path": row.get("extraction_path", ""),
        "image_description": img_desc,
        "audio_description": aud_desc,
        "multimodal_context": " | ".join(mc_parts) if mc_parts else "",
        "modality_combo": row.get("modality_combo", "T"),
    }


def run_single(text, domain="tech_software", image_description="",
               audio_description="", sample_id=None, debug=False):
    """Run pipeline on a single review."""
    metadata = {
        "domain": domain,
        "image_description": image_description,
        "audio_description": audio_description,
    }
    mc_parts = []
    if image_description:
        mc_parts.append(f"Image: {image_description}")
    if audio_description:
        mc_parts.append(f"Audio: {audio_description}")
    metadata["multimodal_context"] = " | ".join(mc_parts) if mc_parts else ""
    
    return run_pipeline(
        text=text,
        sample_id=sample_id or "single_001",
        metadata=metadata,
        debug=debug,
    )


def _save_state(state, path):
    """Save pipeline state to JSON, handling non-serializable fields."""
    skip = {"messages", "raw_images", "raw_audio", "image_base64"}
    cleaned = {}
    for k, v in state.items():
        if k in skip:
            cleaned[k] = f"<{len(v or [])} items>" if isinstance(v, (list, type(None))) else "<binary>"
        else:
            try:
                json.dumps(v)
                cleaned[k] = v
            except (TypeError, ValueError):
                cleaned[k] = str(v)
    with open(path, "w") as f:
        json.dump(cleaned, f, indent=2, default=str)


def run_dataset(dataset_path, output_dir="results", split=None,
                max_samples=None, debug=False):
    """Run pipeline on every sample in the v5 dataset."""
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} samples from {dataset_path}")
    
    # Validate required columns
    required = ["entry_id", "domain", "raw_text", "has_image", "image_description",
                "has_audio", "audio_description", "modality_combo", "is_suggestion",
                "extraction_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    if max_samples:
        df = df.head(max_samples)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    total_time = 0
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        eid = row["entry_id"]
        combo = row.get("modality_combo", "T")
        domain = row.get("domain", "general")
        
        if domain not in ("tech_software", "restaurant"):
            logger.warning(f"{eid}: domain='{domain}' invalid, defaulting to 'tech_software'")
            domain = "tech_software"
        
        print(f"\n{'─'*50}\n[{i}/{len(df)}] {eid} | {combo} | {domain}\n{'─'*50}")
        
        metadata = build_metadata_from_row(row)
        
        has_img = str(row.get("has_image", "False")) == "True"
        has_aud = str(row.get("has_audio", "False")) == "True"
        
        if has_img:
            print(f"  IMG: description ({len(metadata['image_description'])} chars)")
        if has_aud:
            print(f"  AUD: analysis ({len(metadata['audio_description'])} chars)")
        
        try:
            t0 = time.time()
            result = run_pipeline(
                text=str(row["raw_text"]),
                sample_id=eid,
                metadata=metadata,
                debug=debug,
            )
            elapsed = time.time() - t0
            total_time += elapsed
            
            ranked = result.get("ranked_suggestions", [])
            cma = result.get("cross_modal_alignment") or {}
            
            out = {
                "entry_id": eid,
                "extraction_path": row.get("extraction_path"),
                "domain": domain,
                "modality_combo": combo,
                "has_image_desc": has_img,
                "has_audio_desc": has_aud,
                "num_suggestions": len(ranked),
                "top_suggestion": ranked[0]["text"] if ranked else None,
                "top_score": ranked[0].get("score") if ranked else None,
                "alignment": cma.get("overall_alignment"),
                "has_contradiction": cma.get("has_contradiction", False),
                "dominant_modality": cma.get("dominant_modality"),
                "switch_mode": "SPECIFIC" if (cma.get("overall_alignment") or 1) < 0.6 else "COMMON",
                "error": result.get("error"),
                "time_s": round(elapsed, 1),
                "gold_is_suggestion": row.get("is_suggestion"),
                "gold_suggestion": row.get("suggestion_text"),
                "gold_type": row.get("suggestion_type"),
            }
            results.append(out)
            
            # Save incrementally
            pd.DataFrame(results).to_csv(
                os.path.join(output_dir, "pipeline_results.csv"), index=False)
            _save_state(result, os.path.join(output_dir, f"{eid}_full.json"))
            
            if ranked:
                print(f"  → {ranked[0]['text'][:60]} (score={ranked[0].get('score','?')})")
            else:
                print(f"  → No suggestions")
            print(f"  ({elapsed:.1f}s)")
        
        except Exception as e:
            logger.error(f"FAILED {eid}: {e}")
            results.append({"entry_id": eid, "error": str(e), "num_suggestions": 0})
            pd.DataFrame(results).to_csv(
                os.path.join(output_dir, "pipeline_results.csv"), index=False)
    
    # Final summary
    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(output_dir, "pipeline_results.csv"), index=False)
    
    print(f"\n{'='*50}")
    print(f"Done: {len(rdf)} samples → {output_dir}/pipeline_results.csv")
    print(f"  Suggestions found: {(rdf['num_suggestions']>0).sum()}/{len(rdf)}")
    print(f"  With image desc: {rdf.get('has_image_desc', pd.Series(dtype=bool)).sum()}")
    print(f"  With audio desc: {rdf.get('has_audio_desc', pd.Series(dtype=bool)).sum()}")
    print(f"  Errors: {rdf['error'].notna().sum()}")
    print(f"  Total time: {total_time/60:.1f} min ({total_time/len(rdf):.1f}s/sample avg)")
    
    return rdf


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MASP Pipeline Runner (v5 description-first)")
    p.add_argument("--text", help="Review text (single mode)")
    p.add_argument("--domain", default="tech_software", choices=["tech_software", "restaurant"])
    p.add_argument("--image-desc", default="", help="Image description (single mode)")
    p.add_argument("--audio-desc", default="", help="Audio description (single mode)")
    p.add_argument("--sample-id", help="Sample ID")
    p.add_argument("--csv", help="Dataset CSV path")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--output", default="../results")
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    
    if a.text:
        r = run_single(a.text, a.domain, a.image_desc, a.audio_desc, a.sample_id, a.debug)
        print_results(r)
        os.makedirs(a.output, exist_ok=True)
        _save_state(r, os.path.join(a.output, f"{r['sample_id']}_full.json"))
    elif a.csv:
        run_dataset(a.csv, a.output, max_samples=a.max_samples, debug=a.debug)
    else:
        p.print_help()
