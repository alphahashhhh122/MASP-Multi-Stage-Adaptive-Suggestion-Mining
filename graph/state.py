"""
graph/state.py

PipelineState — the single TypedDict flowing through every LangGraph node.

State schema for the MASP pipeline:
  - View-Weighting Switch: formalized gating mechanism (Section 3.2)
  - Dual-labeller arbitration (Section 3.3)
  - 5-type memory features (Section 3.4)
  - Evidence provenance and grounding scores (Section 3.5)
  - Explicit/implicit type switch with differentiated evaluation (Section 3.6)

Pipeline topology:
  preprocess → [text|image|audio views] → cross_modal_align → domain_router
  → [conservative|liberal labellers] → merge → arbitration
  → evidence_provenance → suggestion_switch                    ← NEW
  → canonicaliser → cluster → memory → reranker → human_gate → END
"""

from typing import TypedDict, Annotated, Optional, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


def _keep_last_error(old: Optional[str], new: Optional[str]) -> Optional[str]:
    """Keep the latest non-empty error when parallel branches converge."""
    return new if new else old


# ─── Sentence span ────────────────────────────────────────────────────────────
class Sentence(TypedDict):
    text: str
    start: int
    end: int
    span_confidence: float
    starts_new_topic: bool
    shared_entities: list


# Eight constructed views used for view_agreement_ratio.
VIEW_TYPES = [
    "text_semantic",
    "text_syntactic",
    "text_pragmatic",
    "image_semantic",
    "image_syntactic",
    "image_pragmatic",
    "audio_semantic",
    "audio_pragmatic",
]
TOTAL_VIEWS = len(VIEW_TYPES)  # 8


# ─── TEXT MODALITY VIEWS ──────────────────────────────────────────────────────
class TextSemanticView(TypedDict):
    complaint_frame: Optional[str]
    comparison_frame: Optional[str]
    request_frame: Optional[str]
    true_intent: str
    confidence: float


class TextSyntacticView(TypedDict):
    negative_evaluations: list[str]
    question_patterns: list[str]
    comparative_patterns: list[str]
    modal_verbs: list[str]
    suggestion_indicators: list[str]


class TextPragmaticView(TypedDict):
    communication_type: Literal["direct", "indirect"]
    speech_act: Literal["command", "request", "complaint", "statement", "wish"]
    politeness_level: float
    urgency_score: float
    frustration_level: float
    sentiment_intensity: float


# ─── IMAGE MODALITY VIEWS ─────────────────────────────────────────────────────
class ImageSemanticView(TypedDict):
    described_scene: str
    implied_problem: Optional[str]
    implied_suggestion: Optional[str]
    ui_elements_shown: list[str]
    confidence: float


class ImageSyntacticView(TypedDict):
    layout_issues: list[str]
    missing_ui_elements: list[str]
    comparison_references: list[str]
    error_states_shown: list[str]


class ImagePragmaticView(TypedDict):
    shows_error: bool
    shows_frustration_context: bool
    urgency_visual_cues: list[str]
    implied_user_emotion: Literal["frustrated", "confused", "neutral", "satisfied"]


# ─── AUDIO MODALITY VIEWS ─────────────────────────────────────────────────────
class AudioSemanticView(TypedDict):
    transcript: str
    key_topics: list[str]
    implied_suggestions: list[str]
    confidence: float


class AudioPragmaticView(TypedDict):
    tone: Literal["angry", "frustrated", "neutral", "enthusiastic", "sad"]
    speaking_pace: Literal["fast", "normal", "slow"]
    emphasis_words: list[str]
    urgency_score: float


# ─── CROSS-MODAL ALIGNMENT ────────────────────────────────────────────────────
class CrossModalAlignment(TypedDict):
    text_image_alignment: float
    text_audio_alignment: float
    image_audio_alignment: float
    overall_alignment: float
    text_unique_signals: list[str]
    image_unique_signals: list[str]
    audio_unique_signals: list[str]
    contradictions: list[str]
    has_contradiction: bool
    dominant_modality: Literal["text", "image", "audio", "equal"]


# ─── LABELLED SUGGESTION ─────────────────────────────────────────────────────
class LabelledSuggestion(TypedDict):
    text: str
    confidence: float
    is_implied: bool
    labeller: Literal["conservative", "liberal", "domain_expert"]
    source_modality: Literal["text", "image", "audio", "cross_modal"]
    source_view: Literal["semantic", "syntactic", "pragmatic", "cross_modal"]
    span_start: Optional[int]
    span_end: Optional[int]
    modality_evidence: dict


# ─── ACCEPTED SUGGESTION ─────────────────────────────────────────────────────
class AcceptedSuggestion(TypedDict):
    text: str
    confidence: float
    consensus_score: float
    priority_score: float
    priority_tier: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    supporting_agents: list
    supporting_views: list
    supporting_modalities: list
    modality_agreement_score: float
    factual_confidence: float
    is_implied: bool
    factual_verified: bool


# ─── RANKED SUGGESTION (enriched with provenance + type info) ─────────────────
class RankedSuggestion(TypedDict):
    rank: int
    text: str
    score: float
    view_weighting_mode: Literal["common", "specific"]
    dominant_modality: str
    features: dict


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER PIPELINE STATE
# ═══════════════════════════════════════════════════════════════════════════════


class PipelineState(TypedDict):
    # ── Raw Input ──
    sample_id: str
    raw_text: str
    raw_images: list[bytes]
    raw_audio: Optional[bytes]
    source_metadata: dict

    # ── Layer 1: Multimodal Preprocessing ──
    clean_text: str
    sentences: list[Sentence]
    language: str
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    has_angry_markers: bool
    word_count: int
    image_captions: list[str]
    image_base64: list[str]
    has_images: bool
    audio_transcript: Optional[str]
    has_audio: bool
    acoustic_features: Optional[dict]  # raw prosody features from media/processor.py
    acoustic_description: Optional[str]  # NL description of tone/emphasis/pace

    # ── Layer 2-4: Multi-View ──
    text_semantic_view: Optional[TextSemanticView]
    text_syntactic_view: Optional[TextSyntacticView]
    text_pragmatic_view: Optional[TextPragmaticView]
    text_view_confidence: float
    image_semantic_view: Optional[ImageSemanticView]
    image_syntactic_view: Optional[ImageSyntacticView]
    image_pragmatic_view: Optional[ImagePragmaticView]
    image_view_confidence: float
    audio_semantic_view: Optional[AudioSemanticView]
    audio_pragmatic_view: Optional[AudioPragmaticView]
    audio_view_confidence: float

    # ── Layer 5: Cross-Modal Alignment ──
    cross_modal_alignment: Optional[CrossModalAlignment]

    # ── Layer 6: Domain Router (Agentic RAG framing, A-RAG 2026) ──
    domain: str
    domain_jargon: list[str]

    # ── Layer 7: Labellers (MVP-aware dual labelling, Frontiers 2025) ──
    all_labels: list[LabelledSuggestion]
    _conservative_labels: list
    _liberal_labels: list

    # ── Layer 8: Arbitration ──
    accepted_suggestions: list[AcceptedSuggestion]
    rejected_suggestions: list[dict]

    # Layer 8.5: Evidence provenance and grounding.
    # Each entry is an EvidenceMap.to_dict() with per-modality evidence + grounding score.
    # The node also enriches each accepted_suggestion in-place with:
    #   evidence_map, evidence_highlights, evidence_grounding_score,
    #   evidence_modalities, total_evidence_pieces
    evidence_maps: list[dict]

    # ── Layer 8.6: Suggestion Switch (explicit/implicit type routing) ──
    # The switch enriches accepted_suggestions in-place with:
    #   suggestion_type, switch_confidence, is_actionable, is_feasible,
    #   faithfulness_score, actionability_score, feasibility_score,
    #   specificity_score, inference_chain, type_adjusted_score, switch_eval_metrics
    # Failed suggestions are appended to rejected_suggestions.
    switch_stats: dict  # {total_input, explicit_count, implicit_count, passed_count, failed_count}

    # ── Layer 9: Canonicalisation ──
    canonical_suggestions: list[AcceptedSuggestion]

    # ── Layer 9b: Cluster Agent ──
    cluster_stats: dict

    # ── Layer 10: Memory (Collective Cognition, MAS Memory framework) ──
    memory_hits: dict
    memory_context: dict

    # ── Layer 11: Reranker (View-Weighting Switch) ──
    ranked_suggestions: list[RankedSuggestion]

    # ── Layer 12: Human-in-the-Loop ──
    uncertain_cases: list[dict]
    needs_human_review: bool

    # ── LLM message log ──
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Control ──
    error: Annotated[Optional[str], _keep_last_error]
    processing_time: float
