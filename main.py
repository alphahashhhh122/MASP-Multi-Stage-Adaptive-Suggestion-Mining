"""
main.py — Entry point for the MASP Multimodal Suggestion Mining System

Run:   python main.py
"""

import uuid
import logging

from graph.pipeline import build_graph, build_graph_with_memory
from memory.store import get_memory_manager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

_EMPTY_STATE = dict(
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
    # NEW: evidence provenance + suggestion switch
    evidence_maps=[],
    switch_stats={},
)


def run_pipeline(
    text, images=None, audio_transcript=None, sample_id=None, metadata=None, debug=False
):
    app = build_graph_with_memory() if debug else build_graph()
    # Extract audio description from metadata if present
    mc = (metadata or {}).get("multimodal_context", "")
    audio_desc = None
    has_audio_flag = audio_transcript is not None
    audio_tx = audio_transcript
    if mc and mc != "—":
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
            has_audio_flag = True
            if not audio_tx:
                audio_tx = text  # The review text IS the spoken content

    state = {
        **_EMPTY_STATE,
        "sample_id": sample_id or f"s_{uuid.uuid4().hex[:8]}",
        "raw_text": text,
        "raw_images": images or [],
        "audio_transcript": audio_tx,
        "has_audio": has_audio_flag,
        "acoustic_description": audio_desc,
        "multimodal_context": mc,
        "domain": (metadata or {}).get("domain", "general"),  # validated below
        "source_metadata": metadata or {},
    }
    config = {"configurable": {"thread_id": state["sample_id"]}} if debug else {}
    # C2 FIX: Validate domain
    VALID_DOMAINS = {"tech_software", "restaurant"}
    if state["domain"] not in VALID_DOMAINS:
        logger.warning(
            f"Domain '{state['domain']}' not in {VALID_DOMAINS} — defaulting to 'tech_software'"
        )
        state["domain"] = "tech_software"

    logger.info(f"Pipeline start: {state['sample_id']}")
    result = app.invoke(state, config=config)
    logger.info(f"Pipeline done:  {state['sample_id']}")
    return result


def print_results(state: dict):
    print("\n" + "=" * 70)
    print(f"  SAMPLE : {state.get('sample_id')}")
    print(
        f"  DOMAIN : {state.get('domain')}  |  LANG: {state.get('language')}  |  SENTIMENT: {state.get('sentiment')}"
    )
    print(
        f"  MODALITIES: text={'Y' if state.get('clean_text') else 'N'}  "
        f"image={'Y' if state.get('has_images') else 'N'}  "
        f"audio={'Y' if state.get('has_audio') else 'N'}"
    )

    cma = state.get("cross_modal_alignment") or {}
    if cma:
        alignment = cma.get("overall_alignment", 0)
        mode = "COMMON" if alignment >= 0.6 else "SPECIFIC"
        print(
            f"\n  CROSS-MODAL ALIGNMENT : {alignment:.2f} -> {mode} view-weighting mode"
        )
        print(f"  DOMINANT MODALITY     : {cma.get('dominant_modality', 'text')}")
        if cma.get("image_unique_signals"):
            print(f"  IMAGE UNIQUE SIGNALS  : {cma['image_unique_signals']}")
        if cma.get("has_contradiction"):
            print(f"  CONTRADICTIONS        : {cma.get('contradictions')}")

    # Labelling stats
    all_labels = state.get("all_labels", [])
    c_labels = [
        label for label in all_labels if label.get("labeller") == "conservative"
    ]
    l_labels = [label for label in all_labels if label.get("labeller") == "liberal"]
    accepted = state.get("accepted_suggestions", [])
    print(
        f"\n  LABELS: conservative={len(c_labels)}  liberal={len(l_labels)}  "
        f"accepted={len(accepted)}  rejected={len(state.get('rejected_suggestions', []))}"
    )

    # Evidence Provenance
    evidence_maps = state.get("evidence_maps", [])
    if evidence_maps:
        print(f"\n  EVIDENCE PROVENANCE ({len(evidence_maps)} suggestions grounded):")
        for emap in evidence_maps:
            grounding = emap.get("overall_grounding_score", 0)
            mods = emap.get("modalities_involved", [])
            n_text = len(emap.get("text_evidence", []))
            n_img = len(emap.get("image_evidence", []))
            n_audio = len(emap.get("audio_evidence", []))
            n_cross = len(emap.get("cross_modal_evidence", []))
            print(f"    [{grounding:.2f}] {emap['suggestion_text'][:60]}")
            print(
                f"         pieces={emap.get('total_evidence_pieces', 0)}  modalities={mods}"
            )
            print(
                f"         text={n_text}  image={n_img}  audio={n_audio}  cross_modal={n_cross}"
            )

    # Suggestion Switch
    switch_stats = state.get("switch_stats", {})
    if switch_stats:
        print("\n  SUGGESTION SWITCH:")
        print(f"    Input: {switch_stats.get('total_input', 0)}")
        print(
            f"    Explicit: {switch_stats.get('explicit_count', 0)}  "
            f"Implicit: {switch_stats.get('implicit_count', 0)}"
        )
        print(
            f"    Passed: {switch_stats.get('passed_count', 0)}  "
            f"Failed: {switch_stats.get('failed_count', 0)}"
        )

    # Ranked suggestions
    ranked = state.get("ranked_suggestions", [])
    if ranked:
        print(f"\n  RANKED SUGGESTIONS ({len(ranked)}):")
        tier_icon = {"CRITICAL": "[!]", "HIGH": "[H]", "MEDIUM": "[M]", "LOW": "[L]"}
        for r in ranked:
            feats = r.get("features", {})
            icon = tier_icon.get(feats.get("priority_tier", "LOW"), "[ ]")
            print(f"  {r['rank']:>2}. [{r['score']:.1f}] {icon} {r['text']}")
            print(
                f"       mode={r.get('view_weighting_mode')}  "
                f"type={feats.get('suggestion_type', '?')}  "
                f"grounding={feats.get('grounding_score', '?')}  "
                f"mem_hit={feats.get('memory_hit', '?')}"
            )
    else:
        print("\n  No suggestions extracted.")

    # Uncertain cases
    uncertain = state.get("uncertain_cases", [])
    if uncertain:
        print(f"\n  HUMAN REVIEW: {len(uncertain)} case(s):")
        for u in uncertain:
            print(f"     [{u.get('score', '?')}] {u['text']}  reasons={u['reasons']}")

    if state.get("error"):
        print(f"\n  ERROR: {state['error']}")
    print()


def submit_human_feedback(text, decision, edited=None):
    mem = get_memory_manager()
    mem.record_decision(edited or text, decision, human=True)
    logger.info(f"Human feedback: {decision!r} -> '{edited or text}'")


def main():
    print("\n" + "=" * 70)
    print("   MASP - Multimodal Suggestion Mining Pipeline")
    print("=" * 70)

    print("\n[Example 1] Text-only review")
    state1 = run_pipeline(
        text="The app is so slow when uploading photos! Why can't it compress images before uploading like Instagram does?",
        metadata={
            "source": "app_store",
            "platform": "iOS",
            "user_segment": "power_user",
        },
        sample_id="eg_text_001",
    )
    print_results(state1)

    print("\n[Example 2] Text + audio transcript")
    state2 = run_pipeline(
        text="The checkout process takes too long.",
        audio_transcript="Um, yeah so like... the checkout is REALLY slow, I mean REALLY slow. "
        "It took me like five clicks just to buy one item. Amazon does it in ONE click. Why can't you?",
        metadata={"source": "voice_review"},
        sample_id="eg_audio_002",
    )
    print_results(state2)

    submit_human_feedback("Add image compression before upload", "accept")
    print(f"\nMemory: {get_memory_manager().stats()}")


if __name__ == "__main__":
    main()
