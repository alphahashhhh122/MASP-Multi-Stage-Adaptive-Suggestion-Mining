"""
graph/state.py

PipelineState — the single TypedDict flowing through every LangGraph node.

RESEARCH CONTRIBUTION:
  Multi-View × Multi-Modal fusion.
  Each modality (text, image, audio) produces its own set of views.
  A cross-modal alignment agent fuses them before labelling.

Modalities:
  - TEXT   : customer review / comment
  - IMAGE  : screenshots, product photos attached to the review
  - AUDIO  : voice reviews (transcribed + acoustic features)

Views (per modality):
  - Semantic   : what the user truly wants
  - Syntactic  : sentence/structural patterns (text) or layout patterns (image)
  - Pragmatic  : emotion, urgency, frustration

Cross-modal signals:
  - alignment_score  : do text and image say the SAME thing?
  - contradiction    : does image CONTRADICT text?
  - unique_image_signal : does image reveal something text doesn't say?

View-Weighting Switch (key innovation):
  cross_modal_alignment >= 0.6  → COMMON mode  (text+semantic dominate)
  cross_modal_alignment <  0.6  → SPECIFIC mode (image+domain dominate)
"""

from typing import TypedDict, Annotated, Optional, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# ─── Sentence span ────────────────────────────────────────────────────────────
class Sentence(TypedDict):
    text: str
    start: int
    end: int
    span_confidence: float          # how likely this span contains a suggestion
    starts_new_topic: bool          # GAP 3 FIX: discourse marker detected before this span
    shared_entities: list           # GAP 2 FIX: entities shared with other spans in review


# GAP 6 FIX — explicit 9-view taxonomy (doc's view_agreement_ratio denominator = 9)
# Doc shows view: "semantic" | "pragmatic" | "text_only" | "text_image"
VIEW_TYPES = [
    "text_semantic",    # 1
    "text_syntactic",   # 2
    "text_pragmatic",   # 3
    "image_semantic",   # 4
    "image_syntactic",  # 5
    "image_pragmatic",  # 6
    "audio_semantic",   # 7
    "audio_pragmatic",  # 8
    "cross_modal",      # 9  ← doc's "text_image" combined view
]
TOTAL_VIEWS = len(VIEW_TYPES)  # 9 — denominator for view_agreement_ratio


# ─── TEXT MODALITY VIEWS ──────────────────────────────────────────────────────
class TextSemanticView(TypedDict):
    complaint_frame: Optional[str]   # "app is so slow"
    comparison_frame: Optional[str]  # "like Instagram does"
    request_frame: Optional[str]     # "compress images before upload"
    true_intent: str                 # "wants faster uploads via compression"
    confidence: float


class TextSyntacticView(TypedDict):
    negative_evaluations: list[str]  # ["so slow", "too difficult"]
    question_patterns: list[str]     # ["why can't it", "how come"]
    comparative_patterns: list[str]  # ["like X does", "better than Y"]
    modal_verbs: list[str]           # ["should compress", "could batch"]
    suggestion_indicators: list[str] # ["please add", "would love", "wish it had"]


class TextPragmaticView(TypedDict):
    communication_type: Literal["direct", "indirect"]
    speech_act: Literal["command", "request", "complaint", "statement", "wish"]
    politeness_level: float          # 0-1
    urgency_score: float             # 0-1
    frustration_level: float         # 0-1
    sentiment_intensity: float       # 0-1  (how strong is the sentiment?)


# ─── IMAGE MODALITY VIEWS ─────────────────────────────────────────────────────
class ImageSemanticView(TypedDict):
    """What the image MEANS in the context of a suggestion."""
    described_scene: str             # "Screenshot of upload screen stuck at 23%"
    implied_problem: Optional[str]   # "Upload is frozen / taking too long"
    implied_suggestion: Optional[str]# "Fix upload progress / add speed indicator"
    ui_elements_shown: list[str]     # ["progress_bar", "upload_button", "spinner"]
    confidence: float


class ImageSyntacticView(TypedDict):
    """Structural / layout patterns visible in the image."""
    layout_issues: list[str]         # ["progress bar too small", "no cancel button"]
    missing_ui_elements: list[str]   # ["no compression toggle", "no speed indicator"]
    comparison_references: list[str] # ["similar to Instagram upload UI"]
    error_states_shown: list[str]    # ["error dialog", "frozen spinner", "crash screen"]


class ImagePragmaticView(TypedDict):
    """Emotional signal from the image itself."""
    shows_error: bool                # red X, error screen, crash
    shows_frustration_context: bool  # repeated attempts, many failed uploads
    urgency_visual_cues: list[str]   # ["error badge", "warning icon", "red text"]
    implied_user_emotion: Literal["frustrated", "confused", "neutral", "satisfied"]


# ─── AUDIO MODALITY VIEWS ─────────────────────────────────────────────────────
class AudioSemanticView(TypedDict):
    """What was said in a voice review."""
    transcript: str
    key_topics: list[str]
    implied_suggestions: list[str]
    confidence: float


class AudioPragmaticView(TypedDict):
    """How it was said — tone, pace, emotion."""
    tone: Literal["angry", "frustrated", "neutral", "enthusiastic", "sad"]
    speaking_pace: Literal["fast", "normal", "slow"]    # fast = urgent
    emphasis_words: list[str]        # words spoken loudly / slowly
    urgency_score: float             # 0-1 derived from tone + pace


# ─── CROSS-MODAL ALIGNMENT ────────────────────────────────────────────────────
class CrossModalAlignment(TypedDict):
    """
    The bridge between modalities.
    This drives the VIEW-WEIGHTING SWITCH in the reranker.
    """
    text_image_alignment: float      # 0-1: do text complaint + image show same problem?
    text_audio_alignment: float      # 0-1 (0.5 if no audio)
    image_audio_alignment: float     # 0-1 (0.5 if no audio/image)

    overall_alignment: float         # weighted average → the SWITCH threshold

    # What each modality UNIQUELY contributes (not mentioned in others)
    text_unique_signals: list[str]   # suggestions only text mentions
    image_unique_signals: list[str]  # suggestions only image reveals
    audio_unique_signals: list[str]  # suggestions only voice tone reveals

    # Contradiction detection
    contradictions: list[str]        # e.g. "text says satisfied, image shows error screen"
    has_contradiction: bool

    # Which modality is most informative for this sample
    dominant_modality: Literal["text", "image", "audio", "equal"]


# ─── LABELLED SUGGESTION ─────────────────────────────────────────────────────
class LabelledSuggestion(TypedDict):
    text: str
    confidence: float
    is_implied: bool
    labeller: Literal["conservative", "liberal", "domain_expert"]
    source_modality: Literal["text", "image", "audio", "cross_modal"]
    source_view: Literal["semantic", "syntactic", "pragmatic", "cross_modal"]
    span_start: Optional[int]        # in original text (None if from image/audio)
    span_end: Optional[int]
    modality_evidence: dict          # which modalities support this suggestion


# ─── ACCEPTED SUGGESTION ─────────────────────────────────────────────────────
class AcceptedSuggestion(TypedDict):
    text: str
    confidence: float
    consensus_score: float
    priority_score: float
    priority_tier: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    supporting_agents: list       # doc field name: conservative/liberal
    supporting_views: list        # doc: ["semantic", "text_image"]
    supporting_modalities: list
    modality_agreement_score: float
    factual_confidence: float     # doc field: 0.95 (MCP verification confidence)
    is_implied: bool
    factual_verified: bool


# ─── RANKED SUGGESTION ───────────────────────────────────────────────────────
class RankedSuggestion(TypedDict):
    rank: int
    text: str
    score: float
    view_weighting_mode: Literal["common", "specific"]
    dominant_modality: str
    features: dict                   # all features used for scoring


# ─── MASTER PIPELINE STATE ───────────────────────────────────────────────────
class PipelineState(TypedDict):
    # ── Raw Input ──────────────────────────────────────────────────────────
    sample_id: str
    raw_text: str
    raw_images: list[bytes]          # image bytes (screenshots, product photos)
    raw_audio: Optional[bytes]       # audio bytes (voice review)
    source_metadata: dict

    # ── Layer 1: Multimodal Preprocessing ──────────────────────────────────
    clean_text: str
    sentences: list[Sentence]
    language: str
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    has_angry_markers: bool
    word_count: int

    # Image preprocessing outputs
    image_captions: list[str]         # raw captions per image
    image_base64: list[str]           # base64 encoded for Claude vision API
    has_images: bool

    # Audio preprocessing outputs
    audio_transcript: Optional[str]   # Whisper transcript
    has_audio: bool

    # ── Layer 2: TEXT Multi-View ────────────────────────────────────────────
    text_semantic_view: Optional[TextSemanticView]
    text_syntactic_view: Optional[TextSyntacticView]
    text_pragmatic_view: Optional[TextPragmaticView]
    text_view_confidence: float       # how confident are the text views overall

    # ── Layer 3: IMAGE Multi-View ───────────────────────────────────────────
    image_semantic_view: Optional[ImageSemanticView]
    image_syntactic_view: Optional[ImageSyntacticView]
    image_pragmatic_view: Optional[ImagePragmaticView]
    image_view_confidence: float      # 0.0 if no images

    # ── Layer 4: AUDIO Multi-View ───────────────────────────────────────────
    audio_semantic_view: Optional[AudioSemanticView]
    audio_pragmatic_view: Optional[AudioPragmaticView]
    audio_view_confidence: float      # 0.0 if no audio

    # ── Layer 5: Cross-Modal Alignment ─────────────────────────────────────
    cross_modal_alignment: Optional[CrossModalAlignment]

    # ── Layer 6: Domain Router ──────────────────────────────────────────────
    domain: str
    domain_jargon: list[str]

    # ── Layer 7: Multi-Modal Multi-View Labellers ───────────────────────────
    all_labels: list[LabelledSuggestion]
    _conservative_labels: list
    _liberal_labels: list

    # ── Layer 8: Arbitration ────────────────────────────────────────────────
    accepted_suggestions: list[AcceptedSuggestion]
    rejected_suggestions: list[dict]

    # ── Layer 9: Canonicalisation ───────────────────────────────────────────
    canonical_suggestions: list[AcceptedSuggestion]

    # ── Layer 9b: Cluster Agent (GAP 1 FIX) ────────────────────────────────
    # cluster_size and cluster_id are injected INTO each canonical_suggestion dict
    # by cluster_node so reranker can use "cluster_size: 37" as a feature (per doc)
    cluster_stats: dict             # overall stats: total_clusters, avg_size, etc.

    # ── Layer 10: Memory ────────────────────────────────────────────────────
    memory_hits: dict
    memory_context: dict

    # ── Layer 11: Reranker (View-Weighting Switch) ──────────────────────────
    ranked_suggestions: list[RankedSuggestion]

    # ── Layer 12: Human-in-the-Loop ─────────────────────────────────────────
    uncertain_cases: list[dict]
    needs_human_review: bool

    # ── LLM message log ─────────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Control ─────────────────────────────────────────────────────────────
    error: Optional[str]
    processing_time: float
