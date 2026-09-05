#!/usr/bin/env python3
"""
run_ablations.py — component-ablation runner for MASP.

Each ablation removes ONE component by replacing its node with a
pass-through function in the LangGraph pipeline. This guarantees
the component is actually disabled — unlike environment variables
which the pipeline never reads.

Ablations:
  A1: −Switch       Force COMMON mode (alignment=1.0 always)
                     Tests: View-Weighting Switch (Contribution #1)
  A2: −Memory       Skip memory node (pass-through)
                     Tests: Memory system
  A3: −Dual_label   Skip conservative labeller (liberal only)
                     Tests: Dual labelling (Contribution #4)
  A4: −Arbitration  Accept ALL labels (no filtering)
                     Tests: Arbitration + HN filter
  A5: −Cross_modal  Skip cross-modal alignment (pass-through)
                     Tests: Input signal to Switch
  A6: −Type_switch  Skip suggestion_switch (pass-through)
                     Tests: Type-differentiated extraction (Contribution #2)
  A7: −Evidence     Skip evidence provenance (pass-through)
                     Tests: Grounding verification

Usage:
    python run_ablations.py --dataset data/test.csv --ablation A1
    python run_ablations.py --dataset data/test.csv --ablation all
    python run_ablations.py --dataset data/test.csv --ablation A1 --quick 10

Author: Anonymous
"""

# Reproducibility note for reviewers:
# The final paper's Table 5 uses paper-level labels:
#   A1 = no-image input, A2 = no-audio input, B5 = text-only,
#   A4 = switch-off / no SPECIFIC mode.
# This script is a lower-level component-ablation runner. The legacy
# ``--ablation A1`` option forces COMMON mode and corresponds to the
# paper's A4 switch-off condition. Other legacy A2-A7 options are
# exploratory and are not reported in the final paper table.

import time
import math
import logging
import argparse
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# PASS-THROUGH NODE FACTORIES
# Each returns a function that takes state and returns it unchanged
# (or with minimal modification to keep the graph valid)
# ════════════════════════════════════════════════════════════════


def make_passthrough(name):
    """Generic pass-through: returns empty dict (no state changes)."""

    def _passthrough(state):
        log.debug(f"[ABLATION] {name} skipped (pass-through)")
        return {}

    return _passthrough


def make_force_common_align(original_node):
    """A1: Runs cross-modal alignment but forces alignment=1.0
    so the Switch always uses COMMON mode (no adaptive weighting)."""

    def _forced_common(state):
        result = original_node(state)
        # Override alignment to 1.0 → always COMMON mode
        if "cross_modal_alignment" in result and result["cross_modal_alignment"]:
            result["cross_modal_alignment"]["overall_alignment"] = 1.0
            result["cross_modal_alignment"]["_ablation"] = "A1_forced_common"
        return result

    return _forced_common


def make_accept_all_arbitration(original_node):
    """A4: Runs arbitration but accepts ALL labels (no HN filtering)."""

    def _accept_all(state):
        # Accept everything from merged labels
        all_labels = state.get("all_labels", [])
        return {
            "accepted_suggestions": all_labels,
            "rejected_suggestions": [],
        }

    return _accept_all


def make_liberal_only_merge():
    """A3: Merge node that only uses liberal labels (skips conservative)."""

    def _liberal_only(state):
        liberal = state.get("_liberal_labels", [])
        return {"all_labels": liberal}

    return _liberal_only


# ════════════════════════════════════════════════════════════════
# ABLATION GRAPH BUILDERS
# ════════════════════════════════════════════════════════════════


def build_ablation_graph(ablation_id):
    """Build a LangGraph pipeline with one component ablated."""
    from langgraph.graph import StateGraph, END, START
    from graph.state import PipelineState
    from agents.nodes import (
        preprocess_node,
        text_view_builder_node,
        image_view_builder_node,
        audio_view_builder_node,
        cross_modal_align_node,
        domain_router_node,
        conservative_labeller_node,
        liberal_labeller_node,
        merge_labels_node,
        arbitration_node,
        canonicaliser_node,
        memory_node,
        reranker_node,
        human_review_gate_node,
    )
    from agents.evidence_provenance import evidence_provenance_node
    from agents.suggestion_switch import suggestion_switch_node
    from agents.cluster_agent import cluster_node

    def _route_after_arbitration(state):
        if state.get("error"):
            return "end"
        if not state.get("accepted_suggestions"):
            return "end"
        return "evidence_provenance"

    # Start with default node assignments
    nodes = {
        "preprocess": preprocess_node,
        "text_views": text_view_builder_node,
        "image_views": image_view_builder_node,
        "audio_views": audio_view_builder_node,
        "cross_modal_align": cross_modal_align_node,
        "domain_router": domain_router_node,
        "conservative_labeller": conservative_labeller_node,
        "liberal_labeller": liberal_labeller_node,
        "merge_labels": merge_labels_node,
        "arbitration": arbitration_node,
        "evidence_provenance": evidence_provenance_node,
        "suggestion_switch": suggestion_switch_node,
        "canonicaliser": canonicaliser_node,
        "cluster_agent": cluster_node,  # M1 FIX: use real cluster node
        "memory_agent": memory_node,
        "reranker": reranker_node,
        "human_review_gate": human_review_gate_node,
    }

    # Apply ablation-specific modifications
    if ablation_id == "A1":
        # Force COMMON mode: alignment always 1.0
        nodes["cross_modal_align"] = make_force_common_align(cross_modal_align_node)
        log.info("[A1] View-Weighting Switch DISABLED — forced COMMON mode")

    elif ablation_id == "A2":
        # Skip memory node
        nodes["memory_agent"] = make_passthrough("memory_agent")
        log.info("[A2] Memory system DISABLED — pass-through")

    elif ablation_id == "A3":
        # Liberal only — skip conservative, modify merge
        nodes["conservative_labeller"] = make_passthrough("conservative_labeller")
        nodes["merge_labels"] = make_liberal_only_merge()
        log.info("[A3] Dual labeller DISABLED — liberal only")

    elif ablation_id == "A4":
        # Accept all — no HN filtering
        nodes["arbitration"] = make_accept_all_arbitration(arbitration_node)
        log.info("[A4] Arbitration DISABLED — accept all labels")

    elif ablation_id == "A5":
        # Skip cross-modal alignment
        nodes["cross_modal_align"] = make_passthrough("cross_modal_align")
        log.info("[A5] Cross-modal alignment DISABLED — pass-through")

    elif ablation_id == "A6":
        # Skip suggestion switch (no explicit/implicit differentiation)
        nodes["suggestion_switch"] = make_passthrough("suggestion_switch")
        log.info("[A6] Type-differentiated switch DISABLED — pass-through")

    elif ablation_id == "A7":
        # Skip evidence provenance
        nodes["evidence_provenance"] = make_passthrough("evidence_provenance")
        log.info("[A7] Evidence provenance DISABLED — pass-through")

    else:
        raise ValueError(f"Unknown ablation: {ablation_id}")

    # Build graph with modified nodes
    graph = StateGraph(PipelineState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    # Edges — identical to original pipeline
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "text_views")
    graph.add_edge("preprocess", "image_views")
    graph.add_edge("preprocess", "audio_views")
    graph.add_edge("text_views", "cross_modal_align")
    graph.add_edge("image_views", "cross_modal_align")
    graph.add_edge("audio_views", "cross_modal_align")
    graph.add_edge("cross_modal_align", "domain_router")
    graph.add_edge("domain_router", "conservative_labeller")
    graph.add_edge("domain_router", "liberal_labeller")
    graph.add_edge("conservative_labeller", "merge_labels")
    graph.add_edge("liberal_labeller", "merge_labels")
    graph.add_edge("merge_labels", "arbitration")
    graph.add_conditional_edges(
        "arbitration",
        _route_after_arbitration,
        {"evidence_provenance": "evidence_provenance", "end": END},
    )
    graph.add_edge("evidence_provenance", "suggestion_switch")
    graph.add_edge("suggestion_switch", "canonicaliser")
    graph.add_edge("canonicaliser", "cluster_agent")
    graph.add_edge("cluster_agent", "memory_agent")
    graph.add_edge("memory_agent", "reranker")
    graph.add_edge("reranker", "human_review_gate")
    graph.add_edge("human_review_gate", END)

    return graph.compile()


# ════════════════════════════════════════════════════════════════
# ABLATION RUNNER
# ════════════════════════════════════════════════════════════════


def run_ablation_on_dataset(ablation_id, test_csv, quick=None):
    """Run one ablation on the test set."""

    # Build ablated pipeline
    app = build_ablation_graph(ablation_id)

    # Load test data
    test = pd.read_csv(test_csv)
    if quick:
        test = test.head(quick)
        log.info(f"Quick mode: {quick} samples only")

    log.info(f"Running {ablation_id} on {len(test)} samples...")

    # Empty state template (same as main.py)
    _EMPTY = dict(
        raw_images=[],
        raw_audio=None,
        source_metadata={},
        clean_text="",
        sentences=[],
        language="en",
        sentiment="neutral",
        has_angry_markers=False,
        word_count=0,
        image_captions=[],
        image_base64=[],
        has_images=False,
        audio_transcript=None,
        has_audio=False,
        text_semantic_view=None,
        text_syntactic_view=None,
        text_pragmatic_view=None,
        text_view_confidence=0.0,
        image_semantic_view=None,
        image_syntactic_view=None,
        image_pragmatic_view=None,
        image_view_confidence=0.0,
        audio_semantic_view=None,
        audio_pragmatic_view=None,
        audio_view_confidence=0.0,
        cross_modal_alignment=None,
        domain="general",
        domain_jargon=[],
        all_labels=[],
        _conservative_labels=[],
        _liberal_labels=[],
        accepted_suggestions=[],
        rejected_suggestions=[],
        canonical_suggestions=[],
        memory_hits={},
        memory_context={},
        ranked_suggestions=[],
        uncertain_cases=[],
        needs_human_review=False,
        messages=[],
        error=None,
        processing_time=0.0,
        multimodal_context="",
        acoustic_features=None,
        acoustic_description=None,
        cluster_stats={},
        evidence_maps=[],
        switch_stats={},
    )

    results = []
    errors = 0
    start_all = time.time()

    for i, (_, row) in enumerate(test.iterrows()):
        entry_id = str(row.get("entry_id", f"unknown_{i}"))
        text = str(row.get("raw_text", ""))
        mc = str(row.get("multimodal_context", ""))
        if mc in ["", "nan", "None", "—"]:
            mc = ""

        # Extract audio info from multimodal_context
        audio_desc = None
        has_audio = False
        audio_transcript = None
        if mc:
            mc_lower = mc.lower()
            if any(
                kw in mc_lower
                for kw in [
                    "audio:",
                    "tone",
                    "voice",
                    "frustrated",
                    "monotone",
                    "sarcastic",
                    "emphatic",
                    "resigned",
                    "agitated",
                    "disappointed",
                ]
            ):
                if "Audio:" in mc:
                    audio_desc = mc.split("Audio:")[1].strip()
                elif "audio:" in mc:
                    audio_desc = mc.split("audio:")[1].strip()
                else:
                    audio_desc = mc
                has_audio = True
                audio_transcript = text  # The review text IS the spoken content

        # M2 FIX: Read domain from dataset row
        domain = str(row.get("domain", "tech_software"))
        if domain not in ("tech_software", "restaurant"):
            domain = "tech_software"

        # C4 FIX: Pass image/audio descriptions separately
        img_desc = str(row.get("image_description", ""))
        aud_desc = str(row.get("audio_description", ""))
        if img_desc in ("", "nan", "None", "—"):
            img_desc = ""
        if aud_desc in ("", "nan", "None", "—"):
            aud_desc = ""

        state = {
            **_EMPTY,
            "sample_id": entry_id,
            "raw_text": text,
            "domain": domain,
            "multimodal_context": mc,
            "source_metadata": {
                "multimodal_context": mc,
                "domain": domain,
                "image_description": img_desc,
                "audio_description": aud_desc,
            }
            if mc
            else {
                "domain": domain,
                "image_description": img_desc,
                "audio_description": aud_desc,
            },
            "acoustic_description": audio_desc,
            "has_audio": has_audio,
            "audio_transcript": audio_transcript,
        }

        try:
            result = app.invoke(state)
            n_sugg = len(
                result.get("ranked_suggestions", [])
                or result.get("canonical_suggestions", [])
                or result.get("accepted_suggestions", [])
            )
            results.append(
                {
                    "entry_id": entry_id,
                    "num_suggestions": n_sugg,
                    "ablation": ablation_id,
                }
            )
        except Exception as e:
            errors += 1
            results.append(
                {
                    "entry_id": entry_id,
                    "num_suggestions": 0,
                    "ablation": ablation_id,
                }
            )
            if errors <= 5:
                log.warning(f"  ERROR {entry_id}: {type(e).__name__}: {str(e)[:100]}")

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_all
            rate = (i + 1) / elapsed
            eta = (len(test) - i - 1) / rate / 60
            log.info(
                f"  {ablation_id}: {i + 1}/{len(test)} done | "
                f"{errors} errors | ETA: {eta:.1f} min"
            )

    elapsed = time.time() - start_all
    log.info(
        f"{ablation_id} complete: {len(test)} samples in "
        f"{elapsed / 60:.1f} min | {errors} errors"
    )

    # Save results
    results_df = pd.DataFrame(results)
    out_path = f"results/ablation_{ablation_id}.csv"
    results_df.to_csv(out_path, index=False)

    # Compute metrics
    merged = test.merge(results_df, on="entry_id", how="left")
    merged["num_suggestions"] = merged["num_suggestions"].fillna(0)
    y_true = merged["is_suggestion"].tolist()
    y_pred = [1 if n > 0 else 0 for n in merged["num_suggestions"]]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(den) if den > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"ABLATION {ablation_id} RESULTS")
    print(f"{'=' * 60}")
    print(f"  P={p:.3f} R={r:.3f} F1={f1:.3f} MCC={mcc:.3f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print("  (Compare against MASP full-pipeline run on same test set)")

    # Per-path breakdown
    print("\n  Per-path F1:")
    for path in sorted(merged["extraction_path"].unique()):
        pm = merged[merged["extraction_path"] == path]
        yt = pm["is_suggestion"].tolist()
        yp = [1 if n > 0 else 0 for n in pm["num_suggestions"]]
        tp2 = sum(1 for t, p2 in zip(yt, yp) if t == 1 and p2 == 1)
        fp2 = sum(1 for t, p2 in zip(yt, yp) if t == 0 and p2 == 1)
        fn2 = sum(1 for t, p2 in zip(yt, yp) if t == 1 and p2 == 0)
        pp = tp2 / (tp2 + fp2) if tp2 + fp2 else 0
        rr = tp2 / (tp2 + fn2) if tp2 + fn2 else 0
        ff = 2 * pp * rr / (pp + rr) if pp + rr else 0
        print(f"    {path:<6} F1={ff:.3f} (tp={tp2} fp={fp2} fn={fn2})")

    print(f"\n  Saved: {out_path}")
    return {"ablation": ablation_id, "F1": f1, "MCC": mcc, "P": p, "R": r}


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="MASP ablation experiments")
    parser.add_argument("--dataset", required=True, help="Path to test.csv")
    parser.add_argument(
        "--ablation",
        required=True,
        choices=["A1", "A2", "A3", "A4", "A5", "A6", "A7", "all"],
    )
    parser.add_argument(
        "--quick",
        type=int,
        default=None,
        help="Run on first N samples only (for testing)",
    )
    args = parser.parse_args()

    ablations = (
        ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
        if args.ablation == "all"
        else [args.ablation]
    )

    all_results = []
    for abl in ablations:
        r = run_ablation_on_dataset(abl, args.dataset, args.quick)
        all_results.append(r)

    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("COMPONENT ABLATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Ablation':<12} {'F1':>6} {'MCC':>6}")
        print("-" * 30)
        for r in all_results:
            print(f"{r['ablation']:<12} {r['F1']:>6.3f} {r['MCC']:>6.3f}")


if __name__ == "__main__":
    main()
