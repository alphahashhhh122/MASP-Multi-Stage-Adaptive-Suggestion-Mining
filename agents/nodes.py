"""
agents/nodes.py

Every agent is a pure function: (PipelineState) -> dict (partial state update)

16-node pipeline with integrated research grounding:
  preprocess_node         — text + image + audio preprocessing
  text_view_builder_node  — 3 views of TEXT
  image_view_builder_node — 3 views of IMAGE (Gemma3/Ollama vision)
  audio_view_builder_node — 2 views of AUDIO transcript
  cross_modal_align_node  — fuses all modalities, computes alignment → drives switch
  domain_router_node      — domain classification (agentic RAG framing)
  conservative_labeller_node — explicit only (MVP mitigation via dual labelling)
  liberal_labeller_node      — explicit + implied
  merge_labels_node
  arbitration_node        — cross-modal boosting
  [evidence_provenance_node — in evidence_provenance.py]
  [suggestion_switch_node — in suggestion_switch.py]
  canonicaliser_node
  memory_node             — collective cognition (5-type memory)
  reranker_node           — VIEW-WEIGHTING SWITCH + 15-feature scoring (PATCHED)
  human_review_gate_node  — flags uncertain cases (PATCHED)
"""

import json
import base64
import time
import logging

from llm_backend import call_llm, call_llm_vision
from graph.state import PipelineState
from memory.store import get_memory_manager
from prompts.prompts import (
    TEXT_PREPROCESS_SYSTEM,
    TEXT_PREPROCESS_USER,
    IMAGE_CAPTION_SYSTEM,
    IMAGE_CAPTION_USER,
    TEXT_VIEW_BUILDER_SYSTEM,
    TEXT_VIEW_BUILDER_USER,
    IMAGE_VIEW_BUILDER_SYSTEM,
    IMAGE_VIEW_BUILDER_USER,
    AUDIO_VIEW_BUILDER_SYSTEM,
    AUDIO_VIEW_BUILDER_USER,
    CROSS_MODAL_ALIGNMENT_SYSTEM,
    CROSS_MODAL_ALIGNMENT_USER,
    CONSERVATIVE_LABELLER_SYSTEM,
    CONSERVATIVE_LABELLER_USER,
    LIBERAL_LABELLER_SYSTEM,
    LIBERAL_LABELLER_USER,
    ARBITRATION_SYSTEM,
    ARBITRATION_USER,
    CANONICALISER_SYSTEM,
    CANONICALISER_USER,
)

# UPDATED: reranker prompts in separate file (15-feature set)
from prompts.reranker_prompts import RERANKER_SYSTEM, RERANKER_USER

logger = logging.getLogger(__name__)


def _call_llm(system: str, user: str, fast: bool = False) -> tuple[dict, list]:
    """Route all text calls through the shared Ollama/Gemma3 backend."""
    return call_llm(system, user, fast=fast)


def _call_llm_vision(
    system: str, user_text: str, image_b64: str, mime: str = "image/jpeg"
) -> tuple[dict, list]:
    """Route all vision calls through the shared Ollama/Gemma3 backend."""
    return call_llm_vision(system, user_text, image_b64, mime=mime)


# ═══ PREPROCESSING HELPERS ════════════════════════════════════════════════════


def _detect_discourse_markers(text: str) -> bool:
    markers = [
        "also,",
        "also ",
        "but ",
        "but,",
        "additionally",
        "on the other hand",
        "another thing",
        "furthermore",
        "moreover",
        "however",
        "besides",
        "in addition",
        "what's more",
        "not only that",
        "and also",
    ]
    t = text.lower().strip()
    return any(t.startswith(m) for m in markers)


def _compute_span_similarities(sentences: list[dict]) -> list[dict]:
    if len(sentences) <= 1:
        for s in sentences:
            s["starts_new_topic"] = False
            s["shared_entities"] = []
        return sentences

    def keywords(text):
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "it",
            "i",
            "my",
            "to",
            "of",
            "in",
            "on",
            "for",
            "and",
            "or",
            "but",
            "this",
        }
        return {
            w.lower().strip(".,!?") for w in text.split() if w.lower() not in stopwords
        }

    kw_sets = [keywords(s["text"]) for s in sentences]
    for i, sent in enumerate(sentences):
        shared = set()
        for j, other_kw in enumerate(kw_sets):
            if i != j:
                shared |= kw_sets[i] & other_kw
        sent["shared_entities"] = list(shared)
        sent["starts_new_topic"] = _detect_discourse_markers(sent["text"])
    return sentences


# ═══ NODE 1: PREPROCESS ══════════════════════════════════════════════════════


def preprocess_node(state: PipelineState) -> dict:
    logger.info(f"[preprocess] {state['sample_id']}")
    t0 = time.time()
    updates, all_msgs = {}, []
    try:
        result, msgs = _call_llm(
            TEXT_PREPROCESS_SYSTEM, TEXT_PREPROCESS_USER.format(text=state["raw_text"])
        )
        enriched = _compute_span_similarities(result.get("sentences", []))
        updates.update(
            {
                "clean_text": state["raw_text"].strip(),
                "sentences": enriched,
                "language": result.get("language", "en"),
                "sentiment": result.get("sentiment", "neutral"),
                "has_angry_markers": result.get("has_angry_markers", False),
                "word_count": len(state["raw_text"].split()),
            }
        )
        all_msgs.extend(msgs)
    except Exception as e:
        return {"error": f"preprocess text: {e}", "messages": []}

    image_captions, image_b64_list = [], []
    for raw_bytes in state.get("raw_images", []):
        try:
            b64 = base64.b64encode(raw_bytes).decode("utf-8")
            image_b64_list.append(b64)
            caption_user = IMAGE_CAPTION_USER.format(review_text=state["raw_text"])
            caption_result, msgs = _call_llm_vision(
                IMAGE_CAPTION_SYSTEM, caption_user, b64
            )
            image_captions.append(caption_result)
            all_msgs.extend(msgs)
        except Exception as e:
            logger.warning(f"Image caption failed: {e}")
            image_captions.append(
                {
                    "caption": "Image could not be processed",
                    "ui_state": "normal",
                    "visible_problem": None,
                    "missing_elements": [],
                    "ui_elements": [],
                    "emotional_context": "neutral",
                    "suggestion_implied": None,
                }
            )
    updates.update(
        {
            "image_captions": image_captions,
            "image_base64": image_b64_list,
            "has_images": len(image_b64_list) > 0,
        }
    )

    # ── Audio preprocessing ──
    # If raw_audio bytes provided but no transcript, transcribe with Whisper
    audio_transcript = state.get("audio_transcript") or None
    acoustic_features = state.get("acoustic_features") or None
    acoustic_description = state.get("acoustic_description") or None

    if not audio_transcript and state.get("raw_audio"):
        try:
            from media.processor import AudioProcessor
            import tempfile

            proc = AudioProcessor(whisper_model="base")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(state["raw_audio"])
                tmp_path = tmp.name
            features = proc.process(tmp_path)
            audio_transcript = features.transcript
            acoustic_features = features.to_dict()
            acoustic_description = features.to_description()
            import os

            os.unlink(tmp_path)
            logger.info(
                f"  Audio transcribed: {len(audio_transcript)} chars, tone={features.tone_classification}"
            )
        except Exception as e:
            logger.warning(f"Audio transcription failed: {e}")

    updates.update(
        {
            "audio_transcript": audio_transcript,
            "has_audio": audio_transcript is not None,
            "acoustic_features": acoustic_features,
            "acoustic_description": acoustic_description,
        }
    )
    updates["messages"] = all_msgs
    updates["processing_time"] = time.time() - t0
    return updates


# ═══ NODE 2: TEXT VIEWS ══════════════════════════════════════════════════════


def text_view_builder_node(state: PipelineState) -> dict:
    logger.info(f"[text_views] {state['sample_id']}")
    user = TEXT_VIEW_BUILDER_USER.format(
        text=state["clean_text"],
        sentences=json.dumps(state.get("sentences", []), indent=2),
        sentiment=state["sentiment"],
        angry=state["has_angry_markers"],
    )
    try:
        result, msgs = _call_llm(TEXT_VIEW_BUILDER_SYSTEM, user)
    except Exception as e:
        return {"error": f"text_view_builder: {e}", "messages": []}
    return {
        "text_semantic_view": result.get("semantic"),
        "text_syntactic_view": result.get("syntactic"),
        "text_pragmatic_view": result.get("pragmatic"),
        "text_view_confidence": result.get("text_view_confidence", 0.5),
        "messages": msgs,
    }


# ═══ NODE 3: IMAGE VIEWS ═════════════════════════════════════════════════════


def image_view_builder_node(state: PipelineState) -> dict:
    logger.info(f"[image_views] {state['sample_id']}")
    if not state.get("has_images"):
        return {
            "image_semantic_view": None,
            "image_syntactic_view": None,
            "image_pragmatic_view": None,
            "image_view_confidence": 0.0,
            "messages": [],
        }
    all_msgs, best_result, best_confidence = [], None, 0.0
    for i, (b64, cap) in enumerate(
        zip(state.get("image_base64", []), state.get("image_captions", []))
    ):
        cap_str = cap.get("caption", "") if isinstance(cap, dict) else str(cap)
        try:
            user = IMAGE_VIEW_BUILDER_USER.format(
                text_context=state["clean_text"][:300], caption=cap_str
            )
            result, msgs = _call_llm_vision(IMAGE_VIEW_BUILDER_SYSTEM, user, b64)
            all_msgs.extend(msgs)
            conf = result.get("image_view_confidence", 0.0)
            if conf > best_confidence:
                best_confidence, best_result = conf, result
        except Exception as e:
            logger.warning(f"Image view build failed for image {i}: {e}")
    if best_result is None:
        return {
            "image_semantic_view": None,
            "image_syntactic_view": None,
            "image_pragmatic_view": None,
            "image_view_confidence": 0.0,
            "messages": all_msgs,
        }
    return {
        "image_semantic_view": best_result.get("semantic"),
        "image_syntactic_view": best_result.get("syntactic"),
        "image_pragmatic_view": best_result.get("pragmatic"),
        "image_view_confidence": best_confidence,
        "messages": all_msgs,
    }


# ═══ NODE 4: AUDIO VIEWS ═════════════════════════════════════════════════════


def audio_view_builder_node(state: PipelineState) -> dict:
    logger.info(f"[audio_views] {state['sample_id']}")
    if not state.get("has_audio") or not state.get("audio_transcript"):
        return {
            "audio_semantic_view": None,
            "audio_pragmatic_view": None,
            "audio_view_confidence": 0.0,
            "messages": [],
        }

    # Use real acoustic features if available, otherwise fall back to description or placeholder
    acoustic_str = "Not available — using transcript only"
    if state.get("acoustic_features"):
        af = state["acoustic_features"]
        acoustic_str = (
            f"tone={af.get('tone_classification', 'unknown')} "
            f"(confidence={af.get('tone_confidence', 0):.2f}), "
            f"speaking_rate={af.get('speaking_rate_wpm', 0):.0f} wpm, "
            f"pitch_avg={af.get('avg_pitch_hz', 0):.0f}Hz "
            f"(variability={af.get('pitch_variability', 0):.1f}Hz), "
            f"pauses={af.get('pause_count', 0)} "
            f"(longest={af.get('longest_pause_seconds', 0):.1f}s), "
            f"emphasis_words=[{', '.join(e.get('word', '') for e in af.get('emphasis_segments', [])[:5])}], "
            f"energy_variability={af.get('energy_variability', 0):.3f}"
        )
    elif state.get("acoustic_description"):
        acoustic_str = state["acoustic_description"]

    user = AUDIO_VIEW_BUILDER_USER.format(
        transcript=state["audio_transcript"], acoustic_features=acoustic_str
    )
    try:
        result, msgs = _call_llm(AUDIO_VIEW_BUILDER_SYSTEM, user, fast=True)
    except Exception as e:
        return {"error": f"audio_view_builder: {e}", "messages": []}
    return {
        "audio_semantic_view": result.get("semantic"),
        "audio_pragmatic_view": result.get("pragmatic"),
        "audio_view_confidence": result.get("audio_view_confidence", 0.0),
        "messages": msgs,
    }


# ═══ NODE 5: CROSS-MODAL ALIGNMENT ═══════════════════════════════════════════


def cross_modal_align_node(state: PipelineState) -> dict:
    logger.info(f"[cross_modal_align] {state['sample_id']}")
    text_sem = state.get("text_semantic_view") or {}
    text_prag = state.get("text_pragmatic_view") or {}
    img_sem = state.get("image_semantic_view") or {}
    audio_prag = state.get("audio_pragmatic_view") or {}
    captions = state.get("image_captions") or []
    first_cap = captions[0] if captions else {}
    cap_str = (
        first_cap.get("caption", "No image")
        if isinstance(first_cap, dict)
        else str(first_cap)
    )
    user = CROSS_MODAL_ALIGNMENT_USER.format(
        raw_text=state["clean_text"],
        true_intent=text_sem.get("true_intent", "unknown"),
        complaint_frame=text_sem.get("complaint_frame", "none"),
        comparison_frame=text_sem.get("comparison_frame", "none"),
        urgency_score=text_prag.get("urgency_score", 0.5),
        image_caption=cap_str,
        image_implied_problem=img_sem.get("implied_problem", "No image"),
        image_implied_suggestion=img_sem.get("implied_suggestion", "No image"),
        image_shows_error=(state.get("image_pragmatic_view") or {}).get(
            "shows_error", False
        ),
        audio_transcript=state.get("audio_transcript") or "No audio",
        audio_tone=audio_prag.get("tone", "No audio"),
        audio_urgency=audio_prag.get("urgency_score", 0.5),
    )
    try:
        result, msgs = _call_llm(CROSS_MODAL_ALIGNMENT_SYSTEM, user)
    except Exception as e:
        return {"error": f"cross_modal_align: {e}", "messages": []}
    return {"cross_modal_alignment": result, "messages": msgs}


# ═══ NODE 6: DOMAIN ROUTER ═══════════════════════════════════════════════════


def domain_router_node(state: PipelineState) -> dict:
    logger.info(f"[domain_router] passthrough for {state['sample_id']}")
    return {
        "domain": state.get("domain", "tech_software"),
        "domain_jargon": [],
        "messages": [],
    }


# ═══ NODE 7a: CONSERVATIVE LABELLER ══════════════════════════════════════════


def conservative_labeller_node(state: PipelineState) -> dict:
    logger.info(f"[conservative_labeller] {state['sample_id']}")
    cma = state.get("cross_modal_alignment") or {}
    captions = state.get("image_captions") or []
    first_cap = captions[0] if captions else {}
    user = CONSERVATIVE_LABELLER_USER.format(
        domain=state.get("domain", "general"),
        text=state["clean_text"],
        text_semantic=json.dumps(state.get("text_semantic_view"), indent=2),
        text_syntactic=json.dumps(state.get("text_syntactic_view"), indent=2),
        image_semantic=json.dumps(state.get("image_semantic_view"), indent=2),
        image_syntactic=json.dumps(state.get("image_syntactic_view"), indent=2),
        image_caption=json.dumps(first_cap, indent=2),
        audio_semantic=json.dumps(state.get("audio_semantic_view"), indent=2),
        cross_modal_alignment=cma.get("overall_alignment", 0.5),
        cross_modal_suggestion=cma.get("cross_modal_suggestion", "none"),
    )
    try:
        result, msgs = _call_llm(CONSERVATIVE_LABELLER_SYSTEM, user)
    except Exception as e:
        return {"error": f"conservative_labeller: {e}", "messages": []}
    suggestions = result.get("suggestions", [])
    for s in suggestions:
        s["labeller"] = "conservative"
    return {"_conservative_labels": suggestions, "messages": msgs}


# ═══ NODE 7b: LIBERAL LABELLER ═══════════════════════════════════════════════


def liberal_labeller_node(state: PipelineState) -> dict:
    logger.info(f"[liberal_labeller] {state['sample_id']}")
    text_sem = state.get("text_semantic_view") or {}
    cma = state.get("cross_modal_alignment") or {}
    captions = state.get("image_captions") or []
    user = LIBERAL_LABELLER_USER.format(
        domain=state.get("domain", "general"),
        text=state["clean_text"],
        text_semantic=json.dumps(state.get("text_semantic_view"), indent=2),
        text_syntactic=json.dumps(state.get("text_syntactic_view"), indent=2),
        text_pragmatic=json.dumps(state.get("text_pragmatic_view"), indent=2),
        true_intent=text_sem.get("true_intent", "unknown"),
        image_semantic=json.dumps(state.get("image_semantic_view"), indent=2),
        image_syntactic=json.dumps(state.get("image_syntactic_view"), indent=2),
        image_pragmatic=json.dumps(state.get("image_pragmatic_view"), indent=2),
        image_captions=json.dumps(captions, indent=2),
        audio_semantic=json.dumps(state.get("audio_semantic_view"), indent=2),
        audio_pragmatic=json.dumps(state.get("audio_pragmatic_view"), indent=2),
        overall_alignment=cma.get("overall_alignment", 0.5),
        image_unique=json.dumps(cma.get("image_unique_signals", [])),
        audio_unique=json.dumps(cma.get("audio_unique_signals", [])),
        cross_modal_suggestion=cma.get("cross_modal_suggestion", "none"),
        dominant_modality=cma.get("dominant_modality", "text"),
    )
    try:
        result, msgs = _call_llm(LIBERAL_LABELLER_SYSTEM, user)
    except Exception as e:
        return {"error": f"liberal_labeller: {e}", "messages": []}
    suggestions = result.get("suggestions", [])
    for s in suggestions:
        s["labeller"] = "liberal"
    return {"_liberal_labels": suggestions, "messages": msgs}


# ═══ NODE 8: MERGE LABELS ════════════════════════════════════════════════════


def merge_labels_node(state: PipelineState) -> dict:
    combined = state.get("_conservative_labels", []) + state.get("_liberal_labels", [])
    return {"all_labels": combined}


# ═══ NODE 9: ARBITRATION ═════════════════════════════════════════════════════


def arbitration_node(state: PipelineState) -> dict:
    logger.info(f"[arbitration] {state['sample_id']}")
    conservative = [
        s for s in state.get("all_labels", []) if s.get("labeller") == "conservative"
    ]
    liberal = [s for s in state.get("all_labels", []) if s.get("labeller") == "liberal"]
    text_prag = state.get("text_pragmatic_view") or {}
    cma = state.get("cross_modal_alignment") or {}
    user = ARBITRATION_USER.format(
        conservative_labels=json.dumps(conservative, indent=2),
        liberal_labels=json.dumps(liberal, indent=2),
        domain=state.get("domain", "general"),
        overall_alignment=cma.get("overall_alignment", 0.5),
        dominant_modality=cma.get("dominant_modality", "text"),
        urgency_score=text_prag.get("urgency_score", 0.5),
    )
    try:
        result, msgs = _call_llm(ARBITRATION_SYSTEM, user)
    except Exception as e:
        return {"error": f"arbitration: {e}", "messages": []}
    return {
        "accepted_suggestions": result.get("accepted", []),
        "rejected_suggestions": result.get("rejected", []),
        "messages": msgs,
    }


# ═══ NODE 10: CANONICALISER ══════════════════════════════════════════════════


def canonicaliser_node(state: PipelineState) -> dict:
    logger.info(f"[canonicaliser] {state['sample_id']}")
    accepted = state.get("accepted_suggestions", [])
    if not accepted:
        return {"canonical_suggestions": []}
    user = CANONICALISER_USER.format(
        accepted_suggestions=json.dumps(accepted, indent=2, default=str),
        domain=state.get("domain", "general"),
    )
    try:
        result, msgs = _call_llm(CANONICALISER_SYSTEM, user)
    except Exception as e:
        return {"error": f"canonicaliser: {e}", "messages": []}
    canonical = result.get("canonical", [])

    # FIX: Re-attach enrichment fields from evidence_provenance + suggestion_switch.
    # The LLM canonicaliser returns new JSON that drops these fields.
    # Match each canonical suggestion back to its source via text overlap.
    ENRICH_FIELDS = [
        "evidence_grounding_score",
        "total_evidence_pieces",
        "evidence_modalities",
        "suggestion_type",
        "faithfulness_score",
        "actionability_score",
        "feasibility_score",
        "specificity_score",
        "inference_chain",
        "type_switch_passed",
        "span_verified",
        "action_verb",
        "action_target",
    ]
    for canon in canonical:
        canon_text = (canon.get("canonical_text") or canon.get("text", "")).lower()
        original_forms = [f.lower() for f in canon.get("original_forms", [])]
        # Find best matching accepted suggestion
        best_match, best_score = None, 0
        for acc in accepted:
            acc_text = acc.get("text", "").lower()
            # Match by: canonical_text == original text, or original_forms overlap
            if acc_text == canon_text or acc_text in original_forms:
                best_match = acc
                break
            # Fallback: token overlap
            score = _token_overlap(acc_text, canon_text)
            if score > best_score:
                best_score, best_match = score, acc
        if best_match:
            for field in ENRICH_FIELDS:
                if field in best_match and field not in canon:
                    canon[field] = best_match[field]

    get_memory_manager().store_canonical(canonical)
    return {"canonical_suggestions": canonical, "messages": msgs}


def _token_overlap(a: str, b: str) -> float:
    """Quick token overlap for matching canonical back to original."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


# ═══ NODE 11: MEMORY (Collective Cognition) ══════════════════════════════════


def memory_node(state: PipelineState) -> dict:
    logger.info(f"[memory_node] {state['sample_id']}")
    mem = get_memory_manager()
    cma = state.get("cross_modal_alignment") or {}
    enriched = mem.apply_memory_to_suggestions(
        suggestions=state.get("canonical_suggestions", []),
        alignment_score=cma.get("overall_alignment", 0.5),
        dominant_modality=cma.get("dominant_modality", "text"),
        domain=state.get("domain", "general"),
    )
    return {
        "canonical_suggestions": enriched,
        "memory_hits": mem.lookup(
            state.get("canonical_suggestions", []),
            cma.get("overall_alignment", 0.5),
            cma.get("dominant_modality", "text"),
            state.get("domain", "general"),
        ),
        "memory_context": mem.stats(),
    }


# ═══ NODE 12: RERANKER (PATCHED — 15-feature + evidence + switch) ════════════


def reranker_node(state: PipelineState) -> dict:
    """
    Layer 11 — Reranker with View-Weighting Switch.
    PATCHED: consumes grounding_score + type-specific features from
    evidence_provenance and suggestion_switch nodes.
    """
    logger.info(f"[reranker] {state['sample_id']}")
    text_prag = state.get("text_pragmatic_view") or {}
    cma = state.get("cross_modal_alignment") or {}
    metadata = state.get("source_metadata") or {}
    segment_weights = {"power_user": 0.9, "regular": 0.7, "casual": 0.5}
    user_segment = metadata.get("user_segment", "regular")
    user_segment_importance = segment_weights.get(user_segment, 0.7)

    TOTAL_VIEWS = 8
    enriched = []
    switch_stats = state.get("switch_stats", {})

    for s in state.get("canonical_suggestions", []):
        supporting_views = s.get("supporting_views", [])
        view_count = min(len(supporting_views), TOTAL_VIEWS)
        enriched.append(
            {
                **s,
                "view_agreement_count": view_count,
                "view_agreement_ratio": round(view_count / TOTAL_VIEWS, 3),
                "user_segment_importance": user_segment_importance,
                "modality_alignment": cma.get("overall_alignment", 0.5),
                # Evidence provenance features
                "grounding_score": s.get("evidence_grounding_score", 0.5),
                "total_evidence_pieces": s.get("total_evidence_pieces", 0),
                "evidence_modalities": s.get("evidence_modalities", []),
                # Suggestion switch features
                "suggestion_type": s.get("suggestion_type", "explicit"),
                "is_actionable": s.get("is_actionable", True),
                "is_feasible": s.get("is_feasible", True),
                "faithfulness_score": s.get("faithfulness_score", 0.5),
                "actionability_score": s.get("actionability_score", 0.0),
                "feasibility_score": s.get("feasibility_score", 0.0),
                "specificity_score": s.get("specificity_score", 0.0),
                "type_adjusted_score": s.get("type_adjusted_score", 0.0),
                "inference_chain": s.get("inference_chain", ""),
            }
        )

    evidence_summary = (
        f"Avg grounding: {sum(e.get('grounding_score', 0.5) for e in enriched) / max(len(enriched), 1):.2f}, "
        f"Avg evidence pieces: {sum(e.get('total_evidence_pieces', 0) for e in enriched) / max(len(enriched), 1):.1f}"
    )
    type_summary = (
        f"explicit={switch_stats.get('explicit_count', '?')}, "
        f"implicit={switch_stats.get('implicit_count', '?')}, "
        f"passed={switch_stats.get('passed_count', '?')}/{switch_stats.get('total_input', '?')}"
    )

    user = RERANKER_USER.format(
        canonical_suggestions=json.dumps(enriched, indent=2, default=str),
        overall_alignment=cma.get("overall_alignment", 0.5),
        dominant_modality=cma.get("dominant_modality", "text"),
        memory_context=json.dumps(state.get("memory_context", {}), indent=2),
        domain=state.get("domain", "general"),
        urgency_score=text_prag.get("urgency_score", 0.5),
        user_segment=f"{user_segment} (importance={user_segment_importance})",
        evidence_summary=evidence_summary,
        type_switch_summary=type_summary,
    )
    try:
        result, msgs = _call_llm(RERANKER_SYSTEM, user)
    except Exception as e:
        return {"error": f"reranker: {e}", "messages": []}

    ranked = result.get("ranked", [])
    mem = get_memory_manager()
    for item in ranked:
        mem.record_decision(
            item["text"],
            "accept",
            human=False,
            alignment_score=cma.get("overall_alignment", 0.5),
            dominant_modality=cma.get("dominant_modality", "text"),
            domain=state.get("domain", "general"),
        )
    return {"ranked_suggestions": ranked, "messages": msgs}


# ═══ NODE 13: HUMAN REVIEW GATE (PATCHED — grounding + actionability flags) ═


def human_review_gate_node(state: PipelineState) -> dict:
    """
    Layer 12 — Human-in-the-Loop Gate.
    PATCHED: also flags low grounding_score and implicit low actionability.
    """
    logger.info(f"[human_gate] {state['sample_id']}")
    cma = state.get("cross_modal_alignment") or {}
    has_contradiction = cma.get("has_contradiction", False)

    uncertain = []
    for item in state.get("ranked_suggestions", []):
        reasons = []
        feats = item.get("features", {})
        if item.get("score", 10) < 6.0:
            reasons.append("low_score")
        if not feats.get("memory_hit", True):
            reasons.append("novel_suggestion")
        if feats.get("dominant_modality") == "image" and not feats.get(
            "text_corroborated"
        ):
            reasons.append("image_only_no_text_corroboration")
        if has_contradiction:
            reasons.append("cross_modal_contradiction")
        grounding = feats.get(
            "grounding_score", feats.get("evidence_grounding_score", 1.0)
        )
        if grounding < 0.40:
            reasons.append("low_grounding_faithfulness")
        if (
            feats.get("suggestion_type") == "implicit"
            and feats.get("actionability_score", 5) < 3.0
        ):
            reasons.append("implicit_low_actionability")

        if reasons:
            uncertain.append(
                {
                    "text": item["text"],
                    "score": item.get("score"),
                    "reasons": reasons,
                    "dominant_modality": item.get("dominant_modality"),
                    "grounding_score": grounding,
                    "suggestion_type": feats.get("suggestion_type"),
                }
            )

    return {"uncertain_cases": uncertain, "needs_human_review": len(uncertain) > 0}
