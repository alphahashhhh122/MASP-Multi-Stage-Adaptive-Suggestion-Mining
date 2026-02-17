"""
agents/nodes.py

Every agent is a pure function: (PipelineState) -> dict (partial state update)

NEW NODES vs old version:
  preprocess_node         — now handles TEXT + IMAGE + AUDIO preprocessing
  text_view_builder_node  — 3 views of TEXT only
  image_view_builder_node — 3 views of IMAGE(s) using Claude Vision
  audio_view_builder_node — 2 views of AUDIO transcript + acoustics
  cross_modal_align_node  — THE KEY NODE: fuses all modalities, computes alignment
                             drives the view-weighting switch in reranker
  domain_router_node      — uses all modalities for classification
  conservative_labeller_node — labels from ALL modalities (explicit only)
  liberal_labeller_node      — labels from ALL modalities (explicit + implied)
  merge_labels_node
  arbitration_node        — cross-modal boosting of confidence scores
  canonicaliser_node
  memory_node
  reranker_node           — VIEW-WEIGHTING SWITCH based on cross_modal alignment
  human_review_gate_node
"""

import json
import re
import base64
import time
import logging
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from ..graph.state import PipelineState
from ..memory.store import get_memory_manager
from ..agents.scorer import compute_final_score, get_learned_scorer
from ..prompts.prompts import (
    TEXT_PREPROCESS_SYSTEM, TEXT_PREPROCESS_USER,
    IMAGE_CAPTION_SYSTEM, IMAGE_CAPTION_USER,
    TEXT_VIEW_BUILDER_SYSTEM, TEXT_VIEW_BUILDER_USER,
    IMAGE_VIEW_BUILDER_SYSTEM, IMAGE_VIEW_BUILDER_USER,
    AUDIO_VIEW_BUILDER_SYSTEM, AUDIO_VIEW_BUILDER_USER,
    CROSS_MODAL_ALIGNMENT_SYSTEM, CROSS_MODAL_ALIGNMENT_USER,
    DOMAIN_ROUTER_SYSTEM, DOMAIN_ROUTER_USER,
    CONSERVATIVE_LABELLER_SYSTEM, CONSERVATIVE_LABELLER_USER,
    LIBERAL_LABELLER_SYSTEM, LIBERAL_LABELLER_USER,
    ARBITRATION_SYSTEM, ARBITRATION_USER,
    CANONICALISER_SYSTEM, CANONICALISER_USER,
    RERANKER_SYSTEM, RERANKER_USER,
)

logger = logging.getLogger(__name__)

# ─── Shared LLM client ───────────────────────────────────────────────────────
_llm      = ChatAnthropic(model="claude-opus-4-5-20251101", temperature=0)
_llm_fast = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)  # for cheap tasks


def _call_llm(system: str, user: str, fast: bool = False) -> tuple[dict, list]:
    """Call Claude, strip markdown fences, parse JSON. Returns (dict, messages)."""
    model = _llm_fast if fast else _llm
    msgs = [SystemMessage(content=system), HumanMessage(content=user)]
    response = model.invoke(msgs)
    raw = response.content.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw), [*msgs, response]
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}\nRaw: {raw[:300]}")
        raise ValueError(f"LLM returned invalid JSON: {e}")


def _call_llm_vision(system: str, user_text: str, image_b64: str, mime: str = "image/jpeg") -> tuple[dict, list]:
    """
    Call Claude with an image (base64) + text prompt.
    Uses the multimodal content format required by Claude Vision.
    """
    user_content = [
        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": image_b64}},
        {"type": "text", "text": user_text},
    ]
    msgs = [SystemMessage(content=system), HumanMessage(content=user_content)]
    response = _llm.invoke(msgs)
    raw = response.content.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw), [*msgs, response]
    except json.JSONDecodeError as e:
        raise ValueError(f"Vision LLM returned invalid JSON: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1 — MULTIMODAL PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_discourse_markers(text: str) -> bool:
    """
    GAP 3 FIX — Rule-based discourse marker detection (doc: 'cheap to detect with rules').
    Returns True if this text starts with a marker indicating a NEW suggestion topic.
    """
    markers = [
        "also,", "also ", "but ", "but,", "additionally", "on the other hand",
        "another thing", "furthermore", "moreover", "however", "besides",
        "in addition", "what's more", "not only that", "and also"
    ]
    t = text.lower().strip()
    return any(t.startswith(m) for m in markers)


def _compute_span_similarities(sentences: list[dict]) -> list[dict]:
    """
    GAP 2 FIX — For each pair of extracted span candidates, compute semantic similarity.
    Doc: 'For each pair of extracted candidates, compute semantic similarity.
          Look at features like shared entities, shared topic keywords,
          or whether they reference the same product/feature/module.'

    Uses token-overlap (Jaccard) as fast heuristic.
    Adds shared_entities (shared keywords) to each sentence.
    """
    if len(sentences) <= 1:
        for s in sentences:
            s["starts_new_topic"] = False
            s["shared_entities"]  = []
        return sentences

    # Extract keywords per sentence (nouns/meaningful tokens)
    def keywords(text: str) -> set:
        stopwords = {"the", "a", "an", "is", "are", "was", "it", "i", "my",
                     "to", "of", "in", "on", "for", "and", "or", "but", "this"}
        return {w.lower().strip(".,!?") for w in text.split() if w.lower() not in stopwords}

    kw_sets = [keywords(s["text"]) for s in sentences]

    for i, sent in enumerate(sentences):
        # Shared entities = keywords this sentence shares with ANY other sentence
        shared = set()
        for j, other_kw in enumerate(kw_sets):
            if i != j:
                shared |= (kw_sets[i] & other_kw)

        sent["shared_entities"]  = list(shared)
        sent["starts_new_topic"] = _detect_discourse_markers(sent["text"])

    return sentences


def preprocess_node(state: PipelineState) -> dict:
    """
    Layer 1 — Multimodal Preprocessing

    TEXT:  normalize, segment into sentences, score each span for suggestion potential
    IMAGE: convert raw bytes → base64, call Claude Vision for structured caption
    AUDIO: transcribe (placeholder — plug in Whisper/Deepgram here)
    """
    logger.info(f"[preprocess] {state['sample_id']}")
    t0 = time.time()

    updates = {}
    all_msgs = []

    # ── TEXT ─────────────────────────────────────────────────────────────────
    try:
        result, msgs = _call_llm(
            TEXT_PREPROCESS_SYSTEM,
            TEXT_PREPROCESS_USER.format(text=state["raw_text"])
        )
        raw_sentences = result.get("sentences", [])
        # GAP 2+3 FIX: enrich spans with discourse markers + inter-span similarity
        enriched_sentences = _compute_span_similarities(raw_sentences)

        updates.update({
            "clean_text":        state["raw_text"].strip(),
            "sentences":         enriched_sentences,
            "language":          result.get("language", "en"),
            "sentiment":         result.get("sentiment", "neutral"),
            "has_angry_markers": result.get("has_angry_markers", False),
            "word_count":        len(state["raw_text"].split()),
        })
        all_msgs.extend(msgs)
    except Exception as e:
        return {"error": f"preprocess text: {e}", "messages": []}

    # ── IMAGE ─────────────────────────────────────────────────────────────────
    image_captions = []
    image_b64_list = []

    for raw_bytes in state.get("raw_images", []):
        try:
            b64 = base64.b64encode(raw_bytes).decode("utf-8")
            image_b64_list.append(b64)

            caption_result, msgs = _call_llm_vision(
                IMAGE_CAPTION_SYSTEM,
                IMAGE_CAPTION_USER,
                b64
            )
            image_captions.append(caption_result)
            all_msgs.extend(msgs)
        except Exception as e:
            logger.warning(f"Image caption failed: {e}")
            image_captions.append({
                "caption": "Image could not be processed",
                "ui_state": "normal",
                "visible_problem": None,
                "missing_elements": [],
                "ui_elements": [],
                "emotional_context": "neutral",
                "suggestion_implied": None
            })

    updates.update({
        "image_captions": image_captions,
        "image_base64":   image_b64_list,
        "has_images":     len(image_b64_list) > 0,
    })

    # ── AUDIO ─────────────────────────────────────────────────────────────────
    # Plug in Whisper / Deepgram here. For now: passthrough if transcript provided.
    audio_transcript = state.get("audio_transcript") or None
    updates.update({
        "audio_transcript": audio_transcript,
        "has_audio":        audio_transcript is not None,
    })

    updates["messages"] = all_msgs
    updates["processing_time"] = time.time() - t0
    return updates


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 2 — TEXT MULTI-VIEW BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def text_view_builder_node(state: PipelineState) -> dict:
    """
    Layer 2 — Three views of the TEXT modality:
      Semantic  → what the user truly wants
      Syntactic → structural patterns signalling suggestions
      Pragmatic → emotion, urgency, speech act
    """
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
        "text_semantic_view":  result.get("semantic"),
        "text_syntactic_view": result.get("syntactic"),
        "text_pragmatic_view": result.get("pragmatic"),
        "text_view_confidence": result.get("text_view_confidence", 0.5),
        "messages": msgs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3 — IMAGE MULTI-VIEW BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def image_view_builder_node(state: PipelineState) -> dict:
    """
    Layer 3 — Three views of the IMAGE modality (per image).
    Uses Claude Vision with the base64-encoded images.

    If no images: returns zero-confidence empty views so downstream
    nodes can still run without branching logic.
    """
    logger.info(f"[image_views] {state['sample_id']} — has_images={state.get('has_images')}")

    if not state.get("has_images"):
        return {
            "image_semantic_view":  None,
            "image_syntactic_view": None,
            "image_pragmatic_view": None,
            "image_view_confidence": 0.0,
            "messages": [],
        }

    all_msgs = []
    # Aggregate views across multiple images (take the most informative)
    best_result = None
    best_confidence = 0.0

    captions = state.get("image_captions", [])
    b64_list = state.get("image_base64", [])

    for i, (b64, caption_data) in enumerate(zip(b64_list, captions)):
        caption_str = caption_data.get("caption", "") if isinstance(caption_data, dict) else str(caption_data)

        try:
            user = IMAGE_VIEW_BUILDER_USER.format(
                text_context=state["clean_text"][:300],
                caption=caption_str,
            )
            result, msgs = _call_llm_vision(IMAGE_VIEW_BUILDER_SYSTEM, user, b64)
            all_msgs.extend(msgs)

            conf = result.get("image_view_confidence", 0.0)
            if conf > best_confidence:
                best_confidence = conf
                best_result = result

        except Exception as e:
            logger.warning(f"Image view build failed for image {i}: {e}")
            continue

    if best_result is None:
        return {
            "image_semantic_view":  None,
            "image_syntactic_view": None,
            "image_pragmatic_view": None,
            "image_view_confidence": 0.0,
            "messages": all_msgs,
        }

    return {
        "image_semantic_view":  best_result.get("semantic"),
        "image_syntactic_view": best_result.get("syntactic"),
        "image_pragmatic_view": best_result.get("pragmatic"),
        "image_view_confidence": best_confidence,
        "messages": all_msgs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4 — AUDIO MULTI-VIEW BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def audio_view_builder_node(state: PipelineState) -> dict:
    """
    Layer 4 — Two views of the AUDIO modality:
      Semantic  → what was said (content-level suggestions)
      Pragmatic → how it was said (tone, pace, emotional urgency)

    If no audio: returns zero-confidence empty views.
    """
    logger.info(f"[audio_views] {state['sample_id']} — has_audio={state.get('has_audio')}")

    if not state.get("has_audio") or not state.get("audio_transcript"):
        return {
            "audio_semantic_view":  None,
            "audio_pragmatic_view": None,
            "audio_view_confidence": 0.0,
            "messages": [],
        }

    user = AUDIO_VIEW_BUILDER_USER.format(
        transcript=state["audio_transcript"],
        acoustic_features="Not available — using transcript only"
    )

    try:
        result, msgs = _call_llm(AUDIO_VIEW_BUILDER_SYSTEM, user, fast=True)
    except Exception as e:
        return {"error": f"audio_view_builder: {e}", "messages": []}

    return {
        "audio_semantic_view":  result.get("semantic"),
        "audio_pragmatic_view": result.get("pragmatic"),
        "audio_view_confidence": result.get("audio_view_confidence", 0.0),
        "messages": msgs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 5 — CROSS-MODAL ALIGNMENT  ← THE KEY NODE
# ═══════════════════════════════════════════════════════════════════════════════

def cross_modal_align_node(state: PipelineState) -> dict:
    """
    Layer 5 — Cross-Modal Alignment Agent

    This is the CORE research contribution node.

    It takes ALL three modalities' views and asks Claude:
      - How well do TEXT, IMAGE, AUDIO signals align?  (0-1)
      - What does each modality uniquely contribute?
      - Are there contradictions?
      - Which modality is dominant for this sample?

    The overall_alignment score drives the VIEW-WEIGHTING SWITCH in the reranker:
      >= 0.6  → COMMON mode  (text + semantic dominate)
      <  0.6  → SPECIFIC mode (image + domain dominate — image reveals hidden signal)
    """
    logger.info(f"[cross_modal_align] {state['sample_id']}")

    # Safely extract fields from views (they may be None if modality absent)
    text_sem   = state.get("text_semantic_view")  or {}
    text_prag  = state.get("text_pragmatic_view") or {}
    img_sem    = state.get("image_semantic_view") or {}
    img_prag   = state.get("image_pragmatic_view") or {}
    audio_sem  = state.get("audio_semantic_view") or {}
    audio_prag = state.get("audio_pragmatic_view") or {}
    captions   = state.get("image_captions") or []

    # Build the prompt with all available signals
    first_caption = captions[0] if captions else {}
    caption_str = first_caption.get("caption", "No image") if isinstance(first_caption, dict) else str(first_caption)

    user = CROSS_MODAL_ALIGNMENT_USER.format(
        raw_text=state["clean_text"],
        true_intent=text_sem.get("true_intent", "unknown"),
        complaint_frame=text_sem.get("complaint_frame", "none"),
        comparison_frame=text_sem.get("comparison_frame", "none"),
        urgency_score=text_prag.get("urgency_score", 0.5),

        image_caption=caption_str,
        image_implied_problem=img_sem.get("implied_problem", "No image"),
        image_implied_suggestion=img_sem.get("implied_suggestion", "No image"),
        image_shows_error=img_prag.get("shows_error", False),

        audio_transcript=state.get("audio_transcript") or "No audio",
        audio_tone=audio_prag.get("tone", "No audio"),
        audio_urgency=audio_prag.get("urgency_score", 0.5),
    )

    try:
        result, msgs = _call_llm(CROSS_MODAL_ALIGNMENT_SYSTEM, user)
    except Exception as e:
        return {"error": f"cross_modal_align: {e}", "messages": []}

    return {
        "cross_modal_alignment": result,
        "messages": msgs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 6 — DOMAIN ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

def domain_router_node(state: PipelineState) -> dict:
    """Layer 6 — Domain classification using ALL modality signals."""
    logger.info(f"[domain_router] {state['sample_id']}")

    text_sem = state.get("text_semantic_view") or {}
    img_sem  = state.get("image_semantic_view") or {}
    captions = state.get("image_captions") or []

    first_caption = captions[0] if captions else {}
    img_context = first_caption.get("caption", "No image") if isinstance(first_caption, dict) else "No image"

    user = DOMAIN_ROUTER_USER.format(
        text=state["clean_text"],
        true_intent=text_sem.get("true_intent", "unknown"),
        image_context=img_context,
        jargon_hints=json.dumps(img_sem.get("ui_elements_shown", [])),
    )

    try:
        result, msgs = _call_llm(DOMAIN_ROUTER_SYSTEM, user, fast=True)
    except Exception as e:
        return {"error": f"domain_router: {e}", "messages": []}

    return {
        "domain": result.get("domain", "general"),
        "domain_jargon": result.get("jargon", []),
        "messages": msgs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 7a — CONSERVATIVE LABELLER (multi-modal)
# ═══════════════════════════════════════════════════════════════════════════════

def conservative_labeller_node(state: PipelineState) -> dict:
    """
    Layer 7a — Conservative Labeller
    Labels ONLY explicit suggestions from ANY modality (confidence >= 0.80).
    Passes ALL modality views to Claude so it can find explicit signals in images and audio too.
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 7b — LIBERAL LABELLER (multi-modal)
# ═══════════════════════════════════════════════════════════════════════════════

def liberal_labeller_node(state: PipelineState) -> dict:
    """
    Layer 7b — Liberal Labeller
    Labels ALL suggestions (explicit + implied) from ALL modalities.
    This is where image-only and cross-modal suggestions get captured.
    """
    logger.info(f"[liberal_labeller] {state['sample_id']}")

    text_sem  = state.get("text_semantic_view")  or {}
    text_prag = state.get("text_pragmatic_view") or {}
    cma       = state.get("cross_modal_alignment") or {}
    captions  = state.get("image_captions") or []

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


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 8 — MERGE LABELS
# ═══════════════════════════════════════════════════════════════════════════════

def merge_labels_node(state: PipelineState) -> dict:
    """Fan-in: combine conservative + liberal labels into all_labels."""
    combined = state.get("_conservative_labels", []) + state.get("_liberal_labels", [])
    return {"all_labels": combined}


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 9 — ARBITRATION
# ═══════════════════════════════════════════════════════════════════════════════

def arbitration_node(state: PipelineState) -> dict:
    """
    Layer 8 — Arbitration
    Consensus rules with cross-modal confidence boosting:
      +0.10 if text AND image both support
      +0.05 if audio also supports
    Then priority scoring.
    """
    logger.info(f"[arbitration] {state['sample_id']}")

    conservative = [s for s in state.get("all_labels", []) if s.get("labeller") == "conservative"]
    liberal      = [s for s in state.get("all_labels", []) if s.get("labeller") == "liberal"]
    text_prag    = state.get("text_pragmatic_view") or {}
    cma          = state.get("cross_modal_alignment") or {}

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


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 10 — CANONICALISER
# ═══════════════════════════════════════════════════════════════════════════════

def canonicaliser_node(state: PipelineState) -> dict:
    """Layer 9 — Standardise and deduplicate. Persist to semantic memory."""
    logger.info(f"[canonicaliser] {state['sample_id']}")

    if not state.get("accepted_suggestions"):
        return {"canonical_suggestions": []}

    user = CANONICALISER_USER.format(
        accepted_suggestions=json.dumps(state["accepted_suggestions"], indent=2),
        domain=state.get("domain", "general"),
    )

    try:
        result, msgs = _call_llm(CANONICALISER_SYSTEM, user)
    except Exception as e:
        return {"error": f"canonicaliser: {e}", "messages": []}

    canonical = result.get("canonical", [])
    get_memory_manager().store_canonical(canonical)

    return {"canonical_suggestions": canonical, "messages": msgs}


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 11 — MEMORY AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def memory_node(state: PipelineState) -> dict:
    """
    Layer 10 — Memory Agent (all 5 memory types from doc)

    Doc memory types: semantic | episodic | human_edit | modality_align | cross_domain
    Each has short-term (session) and long-term (persistent) stores.

    ACTIVELY BOOSTS confidence (doc: "Boost confidence from 0.75 to 0.85")
    Applies human edit corrections if a similar suggestion was previously corrected.
    """
    logger.info(f"[memory_node] {state['sample_id']}")

    mem = get_memory_manager()
    cma = state.get("cross_modal_alignment") or {}

    # apply_memory_to_suggestions boosts confidence in-place using all 5 memory types
    enriched = mem.apply_memory_to_suggestions(
        suggestions=state.get("canonical_suggestions", []),
        alignment_score=cma.get("overall_alignment", 0.5),
        dominant_modality=cma.get("dominant_modality", "text"),
        domain=state.get("domain", "general"),
    )

    return {
        "canonical_suggestions": enriched,
        "memory_hits":           mem.lookup(
                                     state.get("canonical_suggestions", []),
                                     cma.get("overall_alignment", 0.5),
                                     cma.get("dominant_modality", "text"),
                                     state.get("domain", "general"),
                                 ),
        "memory_context":        mem.stats(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 12 — RERANKER  ← VIEW-WEIGHTING SWITCH lives here
# ═══════════════════════════════════════════════════════════════════════════════

def reranker_node(state: PipelineState) -> dict:
    """
    Layer 11 - Reranker with View-Weighting Switch

    GAP 4 FIX: now passes full doc feature set:
      cluster_size, canonical_frequency, user_segment_importance,
      view_agreement_count (out of 9 views), modality_alignment
    """
    logger.info(f"[reranker] {state['sample_id']}")

    text_prag = state.get("text_pragmatic_view") or {}
    cma       = state.get("cross_modal_alignment") or {}
    metadata  = state.get("source_metadata") or {}

    # GAP 4 FIX: user_segment_importance from metadata
    segment_weights = {"power_user": 0.9, "regular": 0.7, "casual": 0.5}
    user_segment = metadata.get("user_segment", "regular")
    user_segment_importance = segment_weights.get(user_segment, 0.7)

    # GAP 6 FIX: compute view_agreement_count out of 9 total views
    TOTAL_VIEWS = 9
    enriched = []
    for s in state.get("canonical_suggestions", []):
        supporting_views = s.get("supporting_views", [])
        enriched.append({
            **s,
            "view_agreement_count": len(supporting_views),
            "view_agreement_ratio": round(len(supporting_views) / TOTAL_VIEWS, 3),
            "user_segment_importance": user_segment_importance,
            "modality_alignment": cma.get("overall_alignment", 0.5),
            # cluster_size already set by cluster_node
        })

    user = RERANKER_USER.format(
        canonical_suggestions=json.dumps(enriched, indent=2),
        overall_alignment=cma.get("overall_alignment", 0.5),
        dominant_modality=cma.get("dominant_modality", "text"),
        memory_context=json.dumps(state.get("memory_context", {}), indent=2),
        domain=state.get("domain", "general"),
        urgency_score=text_prag.get("urgency_score", 0.5),
        user_segment=f"{user_segment} (importance={user_segment_importance})",
    )

    try:
        result, msgs = _call_llm(RERANKER_SYSTEM, user)
    except Exception as e:
        return {"error": f"reranker: {e}", "messages": []}

    ranked = result.get("ranked", [])

    mem = get_memory_manager()
    cma_inner = state.get("cross_modal_alignment") or {}
    for item in ranked:
        mem.record_decision(
            item["text"], "accept", human=False,
            alignment_score=cma_inner.get("overall_alignment", 0.5),
            dominant_modality=cma_inner.get("dominant_modality", "text"),
            domain=state.get("domain", "general"),
        )

    return {"ranked_suggestions": ranked, "messages": msgs}


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 13 — HUMAN REVIEW GATE
# ═══════════════════════════════════════════════════════════════════════════════

def human_review_gate_node(state: PipelineState) -> dict:
    """
    Layer 12 — Human-in-the-Loop Gate

    Flags suggestions for human review when:
      - score < 6.0                    (low confidence)
      - memory_hit = False             (novel, never seen)
      - source_modality = image only   (no text corroboration — needs human check)
      - has_contradiction = True       (modalities disagree — ambiguous)
    """
    logger.info(f"[human_gate] {state['sample_id']}")

    cma = state.get("cross_modal_alignment") or {}
    has_contradiction = cma.get("has_contradiction", False)

    uncertain = []
    for item in state.get("ranked_suggestions", []):
        reasons = []
        feats   = item.get("features", {})

        if item.get("score", 10) < 6.0:
            reasons.append("low_score")
        if not feats.get("memory_hit", True):
            reasons.append("novel_suggestion")
        if feats.get("dominant_modality") == "image" and not feats.get("text_corroborated"):
            reasons.append("image_only_no_text_corroboration")
        if has_contradiction:
            reasons.append("cross_modal_contradiction")

        if reasons:
            uncertain.append({
                "text":    item["text"],
                "score":   item.get("score"),
                "reasons": reasons,
                "dominant_modality": item.get("dominant_modality"),
            })

    return {
        "uncertain_cases":    uncertain,
        "needs_human_review": len(uncertain) > 0,
    }
