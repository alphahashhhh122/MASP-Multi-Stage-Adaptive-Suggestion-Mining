"""
agents/evidence_provenance.py

EVIDENCE PROVENANCE TRACKER
============================

Core Problem:
  When the system says "Add image compression before upload", a product manager
  needs to know: WHERE did this come from? Which exact words? Which part of the
  screenshot? What moment in the voice review?

  Without provenance, suggestions are black-box outputs. With provenance, every
  suggestion is a claim backed by traceable evidence across modalities.

What This Module Does:
  For every suggestion extracted by the pipeline, it builds an EvidenceMap:

  ┌──────────────────────────────────────────────────────────────┐
  │  Suggestion: "Add image compression before upload"           │
  │                                                              │
  │  TEXT EVIDENCE:                                              │
  │    Span 1: chars 48-114 "Why can't it compress images..."   │
  │      → view: semantic (complaint_frame)                      │
  │      → confidence: 0.92                                      │
  │    Span 2: chars 0-47 "The app is so slow when uploading"   │
  │      → view: pragmatic (frustration context)                 │
  │      → confidence: 0.75                                      │
  │                                                              │
  │  IMAGE EVIDENCE:                                             │
  │    Region: upload_progress_bar                               │
  │      → bbox: {x: 120, y: 340, w: 280, h: 40}               │
  │      → description: "Progress bar frozen at 23%"             │
  │      → view: image_syntactic (missing_ui_elements)           │
  │      → confidence: 0.88                                      │
  │    Region: whole_screen                                      │
  │      → description: "No compression toggle visible"          │
  │      → view: image_semantic (implied_suggestion)             │
  │                                                              │
  │  AUDIO EVIDENCE:                                             │
  │    Segment: 00:12-00:18                                      │
  │      → transcript: "it took me like five clicks"             │
  │      → view: audio_pragmatic (emphasis on "REALLY slow")     │
  │      → tone: frustrated, pace: fast                          │
  │      → confidence: 0.80                                      │
  │                                                              │
  │  CROSS-MODAL EVIDENCE:                                       │
  │    text_span[48-114] + image_region[progress_bar]            │
  │      → alignment: 0.87 (both point to upload problem)        │
  │      → fusion_insight: "Text complains + image proves stuck" │
  └──────────────────────────────────────────────────────────────┘

Research Value:
  - Makes the system AUDITABLE (critical for enterprise adoption)
  - Enables fine-grained error analysis (which modality/view got it right?)
  - Supports human-in-the-loop (reviewer sees exactly what to verify)
  - Paper contribution: "evidence-grounded multimodal suggestion mining"
"""

import re
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EVIDENCE TYPES — one per modality
# ═══════════════════════════════════════════════════════════════════════════════


class EvidenceType(Enum):
    TEXT_SPAN = "text_span"
    IMAGE_REGION = "image_region"
    AUDIO_SEGMENT = "audio_segment"
    CROSS_MODAL = "cross_modal"


@dataclass
class TextEvidence:
    """
    Points back to exact character positions in the original review text.
    Multiple spans can support one suggestion (e.g., complaint + comparison).
    """

    span_start: int  # character offset in raw_text
    span_end: int
    matched_text: str  # the exact substring
    sentence_index: int  # which sentence (from preprocessing)
    source_view: str  # which view found this: semantic/syntactic/pragmatic
    evidence_role: str  # what role this span plays:
    #   "direct_request", "complaint_frame",
    #   "comparison_frame", "urgency_marker",
    #   "discourse_marker", "modal_verb"
    confidence: float
    highlight_color: str = "#3B8BD4"  # blue for text evidence (for UI rendering)

    def to_highlight(self) -> dict:
        """Returns data needed to render a text highlight in UI."""
        return {
            "type": "text",
            "start": self.span_start,
            "end": self.span_end,
            "text": self.matched_text,
            "role": self.evidence_role,
            "view": self.source_view,
            "confidence": self.confidence,
            "color": self.highlight_color,
        }


@dataclass
class ImageEvidence:
    """
    Points back to a specific region or element in the review's image.
    For screenshots: identifies UI elements, error states, layout issues.
    For product photos: identifies defects, damage, quality issues.
    """

    image_index: int  # which image (0-based, for multi-image reviews)
    region_type: str  # "ui_element", "error_state", "whole_image",
    # "bounding_box", "visual_defect"
    region_id: str  # e.g., "progress_bar", "error_dialog", "packaging"
    bbox: Optional[dict] = None  # {x, y, width, height} normalized 0-1
    # None = whole image
    description: str = ""  # what this region shows
    ui_elements: list = field(default_factory=list)  # ["progress_bar", "spinner"]
    source_view: str = ""  # image_semantic / image_syntactic / image_pragmatic
    evidence_role: str = ""  # "shows_problem", "missing_element",
    # "error_state", "visual_proof", "severity_indicator"
    confidence: float = 0.0
    highlight_color: str = "#1D9E75"  # teal for image evidence

    def to_highlight(self) -> dict:
        """Returns data needed to render an image overlay in UI."""
        return {
            "type": "image",
            "image_index": self.image_index,
            "region_id": self.region_id,
            "region_type": self.region_type,
            "bbox": self.bbox,
            "description": self.description,
            "role": self.evidence_role,
            "view": self.source_view,
            "confidence": self.confidence,
            "color": self.highlight_color,
        }


@dataclass
class AudioEvidence:
    """
    Points back to a specific time range in the audio/transcript.
    Captures both WHAT was said and HOW it was said.
    """

    timestamp_start: float  # seconds from start of audio
    timestamp_end: float
    transcript_span: str  # the words spoken in this segment
    transcript_char_start: int  # char offset in full transcript
    transcript_char_end: int
    source_view: str  # audio_semantic / audio_pragmatic
    evidence_role: str  # "content_mention", "emphasis_word",
    # "tone_shift", "pace_change",
    # "volume_spike", "hesitation"
    acoustic_features: dict = field(default_factory=dict)
    # {tone, pace, volume, emphasis_words}
    confidence: float = 0.0
    highlight_color: str = "#D85A30"  # coral for audio evidence

    def to_highlight(self) -> dict:
        """Returns data needed to render an audio highlight in UI."""
        return {
            "type": "audio",
            "start_seconds": self.timestamp_start,
            "end_seconds": self.timestamp_end,
            "transcript": self.transcript_span,
            "char_start": self.transcript_char_start,
            "char_end": self.transcript_char_end,
            "role": self.evidence_role,
            "view": self.source_view,
            "acoustic": self.acoustic_features,
            "confidence": self.confidence,
            "color": self.highlight_color,
        }


@dataclass
class CrossModalEvidence:
    """
    Evidence that emerges from COMBINING modalities.
    Links specific text spans to image regions and/or audio segments.
    """

    text_evidence: Optional[TextEvidence] = None
    image_evidence: Optional[ImageEvidence] = None
    audio_evidence: Optional[AudioEvidence] = None
    alignment_score: float = 0.0  # how well do these evidence pieces align?
    fusion_insight: str = ""  # what the combination reveals
    evidence_role: str = "cross_modal_confirmation"
    confidence: float = 0.0
    highlight_color: str = "#7F77DD"  # purple for cross-modal

    def to_highlight(self) -> dict:
        pieces = []
        if self.text_evidence:
            pieces.append(self.text_evidence.to_highlight())
        if self.image_evidence:
            pieces.append(self.image_evidence.to_highlight())
        if self.audio_evidence:
            pieces.append(self.audio_evidence.to_highlight())
        return {
            "type": "cross_modal",
            "linked_evidence": pieces,
            "alignment_score": self.alignment_score,
            "fusion_insight": self.fusion_insight,
            "role": self.evidence_role,
            "confidence": self.confidence,
            "color": self.highlight_color,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EVIDENCE MAP — the full provenance record for one suggestion
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EvidenceMap:
    """
    Complete provenance record for a single extracted suggestion.
    Every suggestion in the pipeline gets one of these.
    """

    suggestion_text: str
    suggestion_id: str = ""

    # Per-modality evidence lists
    text_evidence: list[TextEvidence] = field(default_factory=list)
    image_evidence: list[ImageEvidence] = field(default_factory=list)
    audio_evidence: list[AudioEvidence] = field(default_factory=list)
    cross_modal_evidence: list[CrossModalEvidence] = field(default_factory=list)

    # Summary scores
    total_evidence_pieces: int = 0
    modalities_involved: list[str] = field(default_factory=list)
    strongest_evidence: str = ""  # which single piece is most confident
    weakest_link: str = ""  # which modality has lowest confidence
    overall_grounding_score: float = 0.0  # 0-1: how well-grounded overall

    def add_text(self, evidence: TextEvidence):
        self.text_evidence.append(evidence)
        self._update_summary()

    def add_image(self, evidence: ImageEvidence):
        self.image_evidence.append(evidence)
        self._update_summary()

    def add_audio(self, evidence: AudioEvidence):
        self.audio_evidence.append(evidence)
        self._update_summary()

    def add_cross_modal(self, evidence: CrossModalEvidence):
        self.cross_modal_evidence.append(evidence)
        self._update_summary()

    def _update_summary(self):
        """Recompute summary stats after adding evidence."""
        all_pieces = []
        mods = set()

        for e in self.text_evidence:
            all_pieces.append(("text", e.source_view, e.confidence))
            mods.add("text")
        for e in self.image_evidence:
            all_pieces.append(("image", e.source_view, e.confidence))
            mods.add("image")
        for e in self.audio_evidence:
            all_pieces.append(("audio", e.source_view, e.confidence))
            mods.add("audio")
        for e in self.cross_modal_evidence:
            all_pieces.append(("cross_modal", "fusion", e.confidence))
            mods.add("cross_modal")

        self.total_evidence_pieces = len(all_pieces)
        self.modalities_involved = sorted(mods)

        if all_pieces:
            best = max(all_pieces, key=lambda x: x[2])
            self.strongest_evidence = f"{best[0]}:{best[1]} (conf={best[2]:.2f})"
            worst = min(all_pieces, key=lambda x: x[2])
            self.weakest_link = f"{worst[0]}:{worst[1]} (conf={worst[2]:.2f})"
            self.overall_grounding_score = round(
                sum(p[2] for p in all_pieces) / len(all_pieces), 3
            )

    def get_all_highlights(self) -> list[dict]:
        """Returns all evidence highlights for UI rendering."""
        highlights = []
        for e in self.text_evidence:
            highlights.append(e.to_highlight())
        for e in self.image_evidence:
            highlights.append(e.to_highlight())
        for e in self.audio_evidence:
            highlights.append(e.to_highlight())
        for e in self.cross_modal_evidence:
            highlights.append(e.to_highlight())
        return highlights

    def to_dict(self) -> dict:
        return {
            "suggestion_text": self.suggestion_text,
            "suggestion_id": self.suggestion_id,
            "text_evidence": [asdict(e) for e in self.text_evidence],
            "image_evidence": [asdict(e) for e in self.image_evidence],
            "audio_evidence": [asdict(e) for e in self.audio_evidence],
            "cross_modal_evidence": [asdict(e) for e in self.cross_modal_evidence],
            "total_evidence_pieces": self.total_evidence_pieces,
            "modalities_involved": self.modalities_involved,
            "strongest_evidence": self.strongest_evidence,
            "weakest_link": self.weakest_link,
            "overall_grounding_score": self.overall_grounding_score,
            "highlights": self.get_all_highlights(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EVIDENCE EXTRACTORS — one per modality
# ═══════════════════════════════════════════════════════════════════════════════


class TextEvidenceExtractor:
    """
    Extracts text spans that support a suggestion from the original review.
    Uses multiple strategies to find supporting evidence.
    """

    # Patterns that indicate suggestion-relevant language
    ROLE_PATTERNS = {
        "direct_request": [
            r"please\s+\w+",
            r"can\s+you\s+\w+",
            r"add\s+\w+",
            r"implement\s+\w+",
            r"i\s+want",
            r"i\s+need",
            r"should\s+\w+",
            r"must\s+\w+",
            r"would\s+(?:love|like|appreciate)",
            r"wish\s+(?:it|there|you)",
        ],
        "complaint_frame": [
            r"(?:so|too|very|extremely)\s+(?:slow|bad|difficult|annoying|frustrating)",
            r"doesn'?t\s+work",
            r"broken",
            r"terrible",
            r"awful",
            r"can'?t\s+(?:even|use|find|access)",
            r"keeps?\s+(?:crash|freez|fail)",
            r"(?:always|never|constantly)\s+\w+(?:ing|s|ed)",
        ],
        "comparison_frame": [
            r"like\s+\w+\s+does",
            r"(?:better|worse)\s+than",
            r"(?:unlike|compared\s+to)\s+\w+",
            r"other\s+(?:apps?|platforms?|sites?)",
            r"\w+\s+(?:has|have|offers?)\s+(?:this|that|it)",
        ],
        "urgency_marker": [
            r"(?:urgent|critical|blocking|emergency)",
            r"can'?t\s+use\s+(?:at\s+all|anymore)",
            r"(?:please|asap|immediately|right\s+now)",
            r"[!]{2,}",
            r"(?:desperate|stuck|helpless|lost)",
        ],
        "modal_verb": [
            r"(?:should|could|would|might|must|need\s+to)\s+\w+",
        ],
    }

    @classmethod
    def extract(
        cls,
        suggestion_text: str,
        original_text: str,
        sentences: list[dict],
        semantic_view: dict = None,
        syntactic_view: dict = None,
        pragmatic_view: dict = None,
    ) -> list[TextEvidence]:
        """
        Find all text spans in the original review that support this suggestion.

        Strategy:
          1. KEYWORD OVERLAP: Find sentences sharing key terms with the suggestion
          2. PATTERN MATCHING: Find linguistic patterns (requests, complaints, etc.)
          3. VIEW-GUIDED: Use view outputs to locate specific evidence
          4. SEMANTIC FRAME: Match complaint/comparison/request frames from semantic view
        """
        evidence = []
        suggestion_keywords = cls._extract_keywords(suggestion_text)

        # Strategy 1: Keyword overlap per sentence
        for i, sent in enumerate(sentences):
            sent_text = sent.get("text", "")
            sent_start = sent.get("start", 0)
            sent_keywords = cls._extract_keywords(sent_text)

            overlap = suggestion_keywords & sent_keywords
            if not overlap:
                continue

            overlap_ratio = len(overlap) / max(1, len(suggestion_keywords))
            if overlap_ratio < 0.15:
                continue

            # Determine the role of this span
            role = cls._classify_span_role(sent_text)
            view = cls._determine_source_view(sent_text, role, semantic_view)

            # Narrow down to the specific matching substring if possible
            highlight_start, highlight_end, highlight_text = cls._narrow_span(
                sent_text, sent_start, overlap
            )

            evidence.append(
                TextEvidence(
                    span_start=highlight_start,
                    span_end=highlight_end,
                    matched_text=highlight_text,
                    sentence_index=i,
                    source_view=view,
                    evidence_role=role,
                    confidence=round(
                        min(
                            0.99,
                            overlap_ratio * 0.7
                            + sent.get("span_confidence", 0.5) * 0.3,
                        ),
                        3,
                    ),
                )
            )

        # Strategy 2: Pattern-based evidence (finds things keyword overlap misses)
        for role, patterns in cls.ROLE_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, original_text, re.IGNORECASE):
                    start, end = match.start(), match.end()
                    matched = original_text[start:end]

                    # Don't duplicate if already found by keyword overlap
                    if any(
                        e.span_start <= start and e.span_end >= end for e in evidence
                    ):
                        continue

                    # Check if this pattern is relevant to the suggestion
                    pattern_keywords = cls._extract_keywords(matched)
                    if (
                        not (pattern_keywords & suggestion_keywords)
                        and role != "urgency_marker"
                    ):
                        continue

                    sent_idx = cls._find_sentence_index(start, sentences)
                    evidence.append(
                        TextEvidence(
                            span_start=start,
                            span_end=end,
                            matched_text=matched,
                            sentence_index=sent_idx,
                            source_view=cls._role_to_view(role),
                            evidence_role=role,
                            confidence=round(
                                0.6 + (0.1 if role == "direct_request" else 0), 3
                            ),
                        )
                    )

        # Strategy 3: View-guided evidence (semantic view frames)
        if semantic_view:
            for frame_key, frame_role in [
                ("complaint_frame", "complaint_frame"),
                ("comparison_frame", "comparison_frame"),
                ("request_frame", "direct_request"),
            ]:
                frame_text = semantic_view.get(frame_key)
                if frame_text and frame_text not in ("null", "none", None):
                    # Find where this frame appears in original text
                    frame_loc = cls._fuzzy_find(frame_text, original_text)
                    if frame_loc:
                        start, end = frame_loc
                        if not any(
                            e.span_start == start and e.span_end == end
                            for e in evidence
                        ):
                            evidence.append(
                                TextEvidence(
                                    span_start=start,
                                    span_end=end,
                                    matched_text=original_text[start:end],
                                    sentence_index=cls._find_sentence_index(
                                        start, sentences
                                    ),
                                    source_view="semantic",
                                    evidence_role=frame_role,
                                    confidence=semantic_view.get("confidence", 0.7),
                                )
                            )

        # Sort by confidence descending
        evidence.sort(key=lambda e: e.confidence, reverse=True)
        return evidence

    @staticmethod
    def _extract_keywords(text: str) -> set:
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "it",
            "i",
            "my",
            "me",
            "to",
            "of",
            "in",
            "on",
            "for",
            "and",
            "or",
            "but",
            "this",
            "that",
            "with",
            "at",
            "by",
            "from",
            "not",
            "so",
            "be",
            "has",
            "have",
            "had",
            "do",
            "does",
            "did",
            "would",
            "could",
            "should",
            "can",
            "will",
            "just",
            "very",
            "really",
            "too",
            "much",
            "more",
            "also",
            "like",
        }
        words = set(w.lower().strip(".,!?;:'\"()-") for w in text.split() if len(w) > 2)
        return words - stopwords

    @classmethod
    def _classify_span_role(cls, text: str) -> str:
        text_lower = text.lower()
        for role, patterns in cls.ROLE_PATTERNS.items():
            for p in patterns:
                if re.search(p, text_lower):
                    return role
        return "supporting_context"

    @staticmethod
    def _determine_source_view(text: str, role: str, semantic_view: dict = None) -> str:
        role_to_view = {
            "direct_request": "semantic",
            "complaint_frame": "semantic",
            "comparison_frame": "semantic",
            "urgency_marker": "pragmatic",
            "modal_verb": "syntactic",
            "supporting_context": "semantic",
        }
        return role_to_view.get(role, "semantic")

    @staticmethod
    def _role_to_view(role: str) -> str:
        mapping = {
            "direct_request": "syntactic",
            "complaint_frame": "semantic",
            "comparison_frame": "semantic",
            "urgency_marker": "pragmatic",
            "modal_verb": "syntactic",
        }
        return mapping.get(role, "semantic")

    @staticmethod
    def _narrow_span(
        sentence_text: str, sentence_start: int, overlap_keywords: set
    ) -> tuple[int, int, str]:
        """
        Try to narrow the highlight to just the matching portion within a sentence.
        If most of the sentence matches, return the whole sentence.
        """
        words = sentence_text.split()
        matching_indices = []
        for i, w in enumerate(words):
            if w.lower().strip(".,!?;:'\"()-") in overlap_keywords:
                matching_indices.append(i)

        if not matching_indices:
            return sentence_start, sentence_start + len(sentence_text), sentence_text

        # If matches are spread across the sentence, return whole sentence
        spread = matching_indices[-1] - matching_indices[0]
        if spread > len(words) * 0.6:
            return sentence_start, sentence_start + len(sentence_text), sentence_text

        # Return a window around the matching words
        first = max(0, matching_indices[0] - 1)
        last = min(len(words), matching_indices[-1] + 2)
        sub_words = words[first:last]
        sub_text = " ".join(sub_words)

        # Calculate char offset within sentence
        char_offset = len(" ".join(words[:first])) + (1 if first > 0 else 0)
        return (
            sentence_start + char_offset,
            sentence_start + char_offset + len(sub_text),
            sub_text,
        )

    @staticmethod
    def _find_sentence_index(char_pos: int, sentences: list[dict]) -> int:
        for i, s in enumerate(sentences):
            if s.get("start", 0) <= char_pos <= s.get("end", 0):
                return i
        return 0

    @staticmethod
    def _fuzzy_find(query: str, text: str) -> Optional[tuple[int, int]]:
        """Find approximate location of query in text."""
        query_lower = query.lower()
        text_lower = text.lower()

        # Exact substring match first
        idx = text_lower.find(query_lower)
        if idx >= 0:
            return (idx, idx + len(query))

        # Token overlap search: find the window with highest overlap
        query_tokens = set(query_lower.split())
        words = text.split()
        best_score, best_start, best_end = 0, 0, 0

        for window_size in range(len(query_tokens), len(words) + 1):
            for start_idx in range(len(words) - window_size + 1):
                window = words[start_idx : start_idx + window_size]
                window_tokens = set(w.lower().strip(".,!?") for w in window)
                overlap = len(query_tokens & window_tokens) / max(1, len(query_tokens))
                if overlap > best_score:
                    best_score = overlap
                    char_start = len(" ".join(words[:start_idx])) + (
                        1 if start_idx > 0 else 0
                    )
                    window_text = " ".join(window)
                    best_start = char_start
                    best_end = char_start + len(window_text)

        return (best_start, best_end) if best_score > 0.5 else None


class ImageEvidenceExtractor:
    """
    Extracts image regions that support a suggestion.
    Works with the image captions and view outputs from the pipeline.
    """

    # UI element keywords → region mapping
    UI_ELEMENT_MAP = {
        "progress_bar": {
            "region_type": "ui_element",
            "default_bbox": {"x": 0.1, "y": 0.4, "w": 0.8, "h": 0.08},
        },
        "upload_button": {
            "region_type": "ui_element",
            "default_bbox": {"x": 0.3, "y": 0.7, "w": 0.4, "h": 0.1},
        },
        "error_dialog": {
            "region_type": "error_state",
            "default_bbox": {"x": 0.15, "y": 0.2, "w": 0.7, "h": 0.5},
        },
        "error_message": {
            "region_type": "error_state",
            "default_bbox": {"x": 0.1, "y": 0.3, "w": 0.8, "h": 0.15},
        },
        "spinner": {
            "region_type": "ui_element",
            "default_bbox": {"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2},
        },
        "navigation": {
            "region_type": "ui_element",
            "default_bbox": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.1},
        },
        "cancel_button": {
            "region_type": "ui_element",
            "default_bbox": {"x": 0.6, "y": 0.7, "w": 0.3, "h": 0.08},
        },
        "font_text": {
            "region_type": "ui_element",
            "default_bbox": {"x": 0.05, "y": 0.15, "w": 0.9, "h": 0.6},
        },
        "packaging": {"region_type": "visual_defect", "default_bbox": None},
        "product_damage": {"region_type": "visual_defect", "default_bbox": None},
    }

    @classmethod
    def extract(
        cls,
        suggestion_text: str,
        image_captions: list[dict],
        image_semantic_view: dict = None,
        image_syntactic_view: dict = None,
        image_pragmatic_view: dict = None,
    ) -> list[ImageEvidence]:
        """
        Find image regions that support this suggestion.

        Strategy:
          1. Match suggestion keywords against caption text
          2. Check image views for relevant UI elements / problems
          3. Map identified elements to approximate regions (bbox)
          4. Cross-reference: does the image problem match the suggestion topic?
        """
        evidence = []
        suggestion_keywords = TextEvidenceExtractor._extract_keywords(suggestion_text)
        sem = image_semantic_view or {}
        syn = image_syntactic_view or {}
        prag = image_pragmatic_view or {}

        for img_idx, caption_data in enumerate(image_captions):
            caption = (
                caption_data
                if isinstance(caption_data, str)
                else caption_data.get("caption", "")
            )
            caption_keywords = TextEvidenceExtractor._extract_keywords(caption)

            # Strategy 1: Caption keyword overlap
            overlap = suggestion_keywords & caption_keywords
            if overlap:
                evidence.append(
                    ImageEvidence(
                        image_index=img_idx,
                        region_type="whole_image",
                        region_id=f"caption_match_{img_idx}",
                        bbox=None,
                        description=caption,
                        source_view="image_semantic",
                        evidence_role="shows_problem",
                        confidence=round(
                            len(overlap) / max(1, len(suggestion_keywords)) * 0.8, 3
                        ),
                    )
                )

            # Strategy 2: UI elements from views
            shown_elements = sem.get("ui_elements_shown", [])
            missing_elements = syn.get("missing_ui_elements", [])
            error_states = syn.get("error_states_shown", [])

            for element in shown_elements + error_states:
                element_lower = element.lower().replace(" ", "_")
                if element_lower in cls.UI_ELEMENT_MAP:
                    info = cls.UI_ELEMENT_MAP[element_lower]
                    evidence.append(
                        ImageEvidence(
                            image_index=img_idx,
                            region_type=info["region_type"],
                            region_id=element_lower,
                            bbox=info["default_bbox"],
                            description=f"Visible: {element}",
                            ui_elements=[element],
                            source_view="image_syntactic",
                            evidence_role="shows_problem"
                            if element in error_states
                            else "ui_context",
                            confidence=0.75,
                        )
                    )

            for missing in missing_elements:
                evidence.append(
                    ImageEvidence(
                        image_index=img_idx,
                        region_type="missing_element",
                        region_id=f"missing_{missing.lower().replace(' ', '_')}",
                        bbox=None,
                        description=f"Missing from image: {missing}",
                        source_view="image_syntactic",
                        evidence_role="missing_element",
                        confidence=0.70,
                    )
                )

            # Strategy 3: Image pragmatic — error/frustration indicators
            if prag.get("shows_error"):
                evidence.append(
                    ImageEvidence(
                        image_index=img_idx,
                        region_type="error_state",
                        region_id="error_indicator",
                        description="Image shows error state",
                        source_view="image_pragmatic",
                        evidence_role="error_state",
                        confidence=0.85,
                    )
                )

            # Strategy 4: Implied suggestion from image semantic view
            implied = sem.get("implied_suggestion")
            if implied and implied not in ("null", None):
                implied_keywords = TextEvidenceExtractor._extract_keywords(implied)
                if implied_keywords & suggestion_keywords:
                    evidence.append(
                        ImageEvidence(
                            image_index=img_idx,
                            region_type="whole_image",
                            region_id="implied_suggestion_match",
                            description=f"Image implies: {implied}",
                            source_view="image_semantic",
                            evidence_role="visual_proof",
                            confidence=sem.get("confidence", 0.7),
                        )
                    )

        evidence.sort(key=lambda e: e.confidence, reverse=True)
        return evidence


class AudioEvidenceExtractor:
    """
    Extracts audio segments that support a suggestion.
    Works with the transcript and acoustic features.
    """

    # Words that when emphasized indicate importance
    EMPHASIS_INDICATORS = {
        "really",
        "very",
        "extremely",
        "absolutely",
        "always",
        "never",
        "constantly",
        "literally",
        "seriously",
        "terrible",
        "awful",
        "impossible",
        "ridiculous",
        "unacceptable",
    }

    @classmethod
    def extract(
        cls,
        suggestion_text: str,
        transcript: str,
        audio_semantic_view: dict = None,
        audio_pragmatic_view: dict = None,
    ) -> list[AudioEvidence]:
        """
        Find audio segments (time-stamped transcript regions) supporting this suggestion.

        Strategy:
          1. Find transcript segments with keyword overlap
          2. Identify emphasis words and tone shifts
          3. Map to approximate timestamps (word-count based estimation)
          4. Cross-reference with acoustic features (tone, pace)
        """
        if not transcript:
            return []

        evidence = []
        suggestion_keywords = TextEvidenceExtractor._extract_keywords(suggestion_text)
        prag = audio_pragmatic_view or {}

        # Split transcript into segments (by sentences or pauses)
        segments = cls._segment_transcript(transcript)

        for seg in segments:
            seg_keywords = TextEvidenceExtractor._extract_keywords(seg["text"])
            overlap = suggestion_keywords & seg_keywords

            if not overlap:
                continue

            overlap_ratio = len(overlap) / max(1, len(suggestion_keywords))
            if overlap_ratio < 0.15:
                continue

            # Check for emphasis words in this segment
            emphasis_found = []
            for word in seg["text"].split():
                clean = word.lower().strip(".,!?")
                if clean in cls.EMPHASIS_INDICATORS or word.isupper():
                    emphasis_found.append(word)

            role = "content_mention"
            if emphasis_found:
                role = "emphasis_word"

            acoustic = {}
            if prag:
                acoustic = {
                    "tone": prag.get("tone", "neutral"),
                    "pace": prag.get("speaking_pace", "normal"),
                    "emphasis_words": emphasis_found,
                }

            evidence.append(
                AudioEvidence(
                    timestamp_start=seg["est_start"],
                    timestamp_end=seg["est_end"],
                    transcript_span=seg["text"],
                    transcript_char_start=seg["char_start"],
                    transcript_char_end=seg["char_end"],
                    source_view="audio_semantic"
                    if not emphasis_found
                    else "audio_pragmatic",
                    evidence_role=role,
                    acoustic_features=acoustic,
                    confidence=round(
                        min(
                            0.95,
                            overlap_ratio * 0.6 + (0.2 if emphasis_found else 0) + 0.2,
                        ),
                        3,
                    ),
                )
            )

        # Add tone-level evidence if pragmatic view suggests urgency
        if prag.get("urgency_score", 0) > 0.7:
            evidence.append(
                AudioEvidence(
                    timestamp_start=0,
                    timestamp_end=cls._estimate_duration(transcript),
                    transcript_span="[Overall tone analysis]",
                    transcript_char_start=0,
                    transcript_char_end=len(transcript),
                    source_view="audio_pragmatic",
                    evidence_role="tone_shift",
                    acoustic_features={
                        "tone": prag.get("tone", "frustrated"),
                        "urgency_score": prag.get("urgency_score", 0.7),
                    },
                    confidence=round(prag.get("urgency_score", 0.7), 3),
                )
            )

        evidence.sort(key=lambda e: e.confidence, reverse=True)
        return evidence

    @staticmethod
    def _segment_transcript(transcript: str) -> list[dict]:
        """
        Split transcript into segments for evidence extraction.
        Estimates timestamps based on average speaking rate (~150 words/min).
        """
        # Split on sentence boundaries and pause indicators
        raw_segments = re.split(
            r"[.!?]+\s*|(?:\.{3}|…)\s*|,\s+(?=(?:um|uh|like|so|and)\s)", transcript
        )
        segments = []
        char_pos = 0
        time_pos = 0.0
        words_per_second = 2.5  # ~150 wpm

        for seg_text in raw_segments:
            seg_text = seg_text.strip()
            if not seg_text:
                continue

            word_count = len(seg_text.split())
            duration = word_count / words_per_second

            # Find actual char position in original transcript
            idx = transcript.find(seg_text, char_pos)
            if idx < 0:
                idx = char_pos

            segments.append(
                {
                    "text": seg_text,
                    "char_start": idx,
                    "char_end": idx + len(seg_text),
                    "est_start": round(time_pos, 1),
                    "est_end": round(time_pos + duration, 1),
                    "word_count": word_count,
                }
            )

            char_pos = idx + len(seg_text)
            time_pos += duration

        return segments

    @staticmethod
    def _estimate_duration(transcript: str) -> float:
        words = len(transcript.split())
        return round(words / 2.5, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PROVENANCE BUILDER — orchestrates all extractors
# ═══════════════════════════════════════════════════════════════════════════════


class ProvenanceBuilder:
    """
    Builds complete EvidenceMaps for each suggestion in the pipeline.
    Called after labeling, before canonicalization.
    """

    @staticmethod
    def build_evidence_map(
        suggestion: dict,
        original_text: str,
        sentences: list[dict],
        image_captions: list[dict] = None,
        audio_transcript: str = None,
        text_semantic_view: dict = None,
        text_syntactic_view: dict = None,
        text_pragmatic_view: dict = None,
        image_semantic_view: dict = None,
        image_syntactic_view: dict = None,
        image_pragmatic_view: dict = None,
        audio_semantic_view: dict = None,
        audio_pragmatic_view: dict = None,
        cross_modal_alignment: dict = None,
    ) -> EvidenceMap:
        """Build complete provenance for one suggestion."""

        suggestion_text = suggestion.get("text", "")
        emap = EvidenceMap(
            suggestion_text=suggestion_text,
            suggestion_id=suggestion.get(
                "id", hashlib.md5(suggestion_text.encode()).hexdigest()[:8]
            ),
        )

        # 1. Text evidence
        text_ev = TextEvidenceExtractor.extract(
            suggestion_text=suggestion_text,
            original_text=original_text,
            sentences=sentences,
            semantic_view=text_semantic_view,
            syntactic_view=text_syntactic_view,
            pragmatic_view=text_pragmatic_view,
        )
        for e in text_ev:
            emap.add_text(e)

        # 2. Image evidence
        if image_captions:
            img_ev = ImageEvidenceExtractor.extract(
                suggestion_text=suggestion_text,
                image_captions=image_captions,
                image_semantic_view=image_semantic_view,
                image_syntactic_view=image_syntactic_view,
                image_pragmatic_view=image_pragmatic_view,
            )
            for e in img_ev:
                emap.add_image(e)

        # 3. Audio evidence
        if audio_transcript:
            audio_ev = AudioEvidenceExtractor.extract(
                suggestion_text=suggestion_text,
                transcript=audio_transcript,
                audio_semantic_view=audio_semantic_view,
                audio_pragmatic_view=audio_pragmatic_view,
            )
            for e in audio_ev:
                emap.add_audio(e)

        # 4. Cross-modal evidence
        cma = cross_modal_alignment or {}
        if emap.text_evidence and emap.image_evidence:
            best_text = emap.text_evidence[0] if emap.text_evidence else None
            best_image = emap.image_evidence[0] if emap.image_evidence else None
            emap.add_cross_modal(
                CrossModalEvidence(
                    text_evidence=best_text,
                    image_evidence=best_image,
                    alignment_score=cma.get("text_image_alignment", 0.5),
                    fusion_insight=cma.get("fusion_insight", ""),
                    evidence_role="text_image_confirmation",
                    confidence=cma.get("overall_alignment", 0.5),
                )
            )

        if emap.text_evidence and emap.audio_evidence:
            best_text = emap.text_evidence[0]
            best_audio = emap.audio_evidence[0]
            emap.add_cross_modal(
                CrossModalEvidence(
                    text_evidence=best_text,
                    audio_evidence=best_audio,
                    alignment_score=cma.get("text_audio_alignment", 0.5),
                    fusion_insight="Audio tone confirms text complaint",
                    evidence_role="text_audio_confirmation",
                    confidence=min(best_text.confidence, best_audio.confidence),
                )
            )

        return emap


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LANGGRAPH NODE — integrates into the pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def evidence_provenance_node(state: dict) -> dict:
    """
    LangGraph node: builds evidence maps for all accepted suggestions.
    Sits between arbitration and canonicalization (alongside suggestion_switch).

    Pipeline position:
      ... → arbitration → evidence_provenance → suggestion_switch → canonicaliser → ...
    """
    logger.info(f"[evidence_provenance] {state['sample_id']}")

    accepted = state.get("accepted_suggestions", [])
    original_text = state.get("clean_text", "")
    sentences = state.get("sentences", [])
    captions = state.get("image_captions") or []
    transcript = state.get("audio_transcript")
    cma = state.get("cross_modal_alignment") or {}

    evidence_maps = []

    for suggestion in accepted:
        emap = ProvenanceBuilder.build_evidence_map(
            suggestion=suggestion,
            original_text=original_text,
            sentences=sentences,
            image_captions=captions,
            audio_transcript=transcript,
            text_semantic_view=state.get("text_semantic_view"),
            text_syntactic_view=state.get("text_syntactic_view"),
            text_pragmatic_view=state.get("text_pragmatic_view"),
            image_semantic_view=state.get("image_semantic_view"),
            image_syntactic_view=state.get("image_syntactic_view"),
            image_pragmatic_view=state.get("image_pragmatic_view"),
            audio_semantic_view=state.get("audio_semantic_view"),
            audio_pragmatic_view=state.get("audio_pragmatic_view"),
            cross_modal_alignment=cma,
        )
        evidence_maps.append(emap.to_dict())

        # Enrich the suggestion with its provenance
        suggestion["evidence_map"] = emap.to_dict()
        suggestion["evidence_highlights"] = emap.get_all_highlights()
        suggestion["evidence_grounding_score"] = emap.overall_grounding_score
        suggestion["evidence_modalities"] = emap.modalities_involved
        suggestion["total_evidence_pieces"] = emap.total_evidence_pieces

    logger.info(
        f"[evidence_provenance] Built {len(evidence_maps)} evidence maps, "
        f"avg {sum(e['total_evidence_pieces'] for e in evidence_maps) / max(1, len(evidence_maps)):.1f} "
        f"evidence pieces per suggestion"
    )

    return {
        "accepted_suggestions": accepted,
        "evidence_maps": evidence_maps,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DEMO
# ═══════════════════════════════════════════════════════════════════════════════


def demo():
    """Run evidence provenance on sample data."""
    print("\n" + "=" * 80)
    print("  EVIDENCE PROVENANCE TRACKER — DEMO")
    print("=" * 80)

    # Multimodal example
    original_text = (
        "The app is so slow when uploading photos! "
        "Why can't it compress images before uploading like Instagram does?"
    )
    sentences = [
        {
            "text": "The app is so slow when uploading photos!",
            "start": 0,
            "end": 44,
            "span_confidence": 0.65,
        },
        {
            "text": "Why can't it compress images before uploading like Instagram does?",
            "start": 45,
            "end": 112,
            "span_confidence": 0.90,
        },
    ]
    image_captions = [
        {
            "caption": "Screenshot showing upload progress bar frozen at 23%",
            "ui_state": "frozen",
            "visible_problem": "Upload stuck",
        }
    ]
    audio_transcript = (
        "Um, yeah so like... the checkout is REALLY slow, I mean REALLY slow. "
        "It took me like five clicks just to buy one item. "
        "Amazon does it in ONE click. Why can't you?"
    )

    suggestions = [
        {
            "text": "Add image compression before upload",
            "confidence": 0.92,
            "is_implied": False,
        },
        {"text": "Improve upload speed", "confidence": 0.75, "is_implied": True},
    ]

    for sugg in suggestions:
        emap = ProvenanceBuilder.build_evidence_map(
            suggestion=sugg,
            original_text=original_text,
            sentences=sentences,
            image_captions=image_captions,
            audio_transcript=audio_transcript,
            text_semantic_view={
                "complaint_frame": "app is so slow when uploading photos",
                "comparison_frame": "like Instagram does",
                "request_frame": "compress images before uploading",
                "confidence": 0.88,
            },
            image_semantic_view={
                "implied_suggestion": "Fix upload progress and add speed indicator",
                "ui_elements_shown": ["progress_bar", "spinner"],
                "confidence": 0.85,
            },
            image_syntactic_view={
                "missing_ui_elements": ["cancel button", "speed indicator"],
                "error_states_shown": ["frozen spinner"],
            },
            image_pragmatic_view={"shows_error": True},
            audio_pragmatic_view={
                "tone": "frustrated",
                "urgency_score": 0.8,
                "speaking_pace": "fast",
            },
            cross_modal_alignment={
                "text_image_alignment": 0.85,
                "overall_alignment": 0.80,
                "fusion_insight": "Text complains about speed, image proves upload is stuck",
            },
        )

        print(f"\n{'─' * 70}")
        print(f'  Suggestion: "{emap.suggestion_text}"')
        print(
            f"  Grounding:  {emap.overall_grounding_score:.2f} | "
            f"Pieces: {emap.total_evidence_pieces} | "
            f"Modalities: {', '.join(emap.modalities_involved)}"
        )
        print(f"  Strongest:  {emap.strongest_evidence}")

        if emap.text_evidence:
            print(f"\n  📝 TEXT EVIDENCE ({len(emap.text_evidence)} spans):")
            for e in emap.text_evidence:
                print(f'     [{e.span_start}:{e.span_end}] "{e.matched_text}"')
                print(
                    f"       role={e.evidence_role}  view={e.source_view}  conf={e.confidence}"
                )

        if emap.image_evidence:
            print(f"\n  🖼️  IMAGE EVIDENCE ({len(emap.image_evidence)} regions):")
            for e in emap.image_evidence:
                bbox_str = f"bbox={e.bbox}" if e.bbox else "whole image"
                print(f"     [{e.region_id}] {e.description}")
                print(
                    f"       role={e.evidence_role}  view={e.source_view}  {bbox_str}  conf={e.confidence}"
                )

        if emap.audio_evidence:
            print(f"\n  🎤 AUDIO EVIDENCE ({len(emap.audio_evidence)} segments):")
            for e in emap.audio_evidence:
                print(
                    f'     [{e.timestamp_start:.1f}s-{e.timestamp_end:.1f}s] "{e.transcript_span[:60]}"'
                )
                print(
                    f"       role={e.evidence_role}  view={e.source_view}  conf={e.confidence}"
                )
                if e.acoustic_features:
                    print(f"       acoustic: {e.acoustic_features}")

        if emap.cross_modal_evidence:
            print(f"\n  🔗 CROSS-MODAL EVIDENCE ({len(emap.cross_modal_evidence)}):")
            for e in emap.cross_modal_evidence:
                print(f"     alignment={e.alignment_score:.2f}  role={e.evidence_role}")
                if e.fusion_insight:
                    print(f"     insight: {e.fusion_insight}")


if __name__ == "__main__":
    demo()
