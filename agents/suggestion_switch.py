"""
agents/suggestion_switch.py

EXPLICIT vs IMPLICIT SUGGESTION SWITCH
=======================================

Core Innovation:
  Explicit and implicit suggestions are fundamentally different beasts:
    - EXPLICIT: "Please add dark mode" → user stated what they want
    - IMPLICIT: "I can't use the app at night, my eyes hurt" → user described pain, fix is inferred

  They require DIFFERENT extraction logic, DIFFERENT confidence thresholds,
  DIFFERENT verification, and DIFFERENT evaluation metrics.

  This module implements a formal switch that routes suggestions through
  the appropriate pipeline based on their type.

The Switch Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │                    SUGGESTION CLASSIFIER                        │
  │  Input: raw labeled suggestion from conservative/liberal agent  │
  │  Output: explicit | implicit | ambiguous                       │
  └──────────────┬──────────────────────────┬──────────────────────┘
                 │                          │
        ┌────────▼────────┐       ┌────────▼─────────┐
        │  EXPLICIT PATH  │       │  IMPLICIT PATH   │
        │                 │       │                   │
        │ • Span verify   │       │ • Intent recon-   │
        │ • Lexical match │       │   struction       │
        │ • Direct action │       │ • Actionability   │
        │   extraction    │       │   transform       │
        │ • High thresh   │       │ • Feasibility     │
        │   (≥0.80)       │       │   gate            │
        │                 │       │ • Lower thresh    │
        │                 │       │   (≥0.60) BUT     │
        │                 │       │   extra checks    │
        └────────┬────────┘       └────────┬─────────┘
                 │                          │
        ┌────────▼────────┐       ┌────────▼─────────┐
        │ EXPLICIT EVAL   │       │ IMPLICIT EVAL    │
        │                 │       │                   │
        │ • Span F1       │       │ • Intent match    │
        │ • Exact match   │       │ • Actionability   │
        │ • Precision@K   │       │   score (1-5)     │
        │                 │       │ • Feasibility     │
        │                 │       │   score (1-5)     │
        │                 │       │ • Faithfulness    │
        │                 │       │   (grounded in    │
        │                 │       │    source text?)   │
        └────────┬────────┘       └────────┬─────────┘
                 │                          │
                 └──────────┬───────────────┘
                            │
                   ┌────────▼────────┐
                   │  UNIFIED RANKER │
                   │  (but type-     │
                   │   aware weights)│
                   └─────────────────┘

Research Justification:
  - Explicit suggestions: "Add X" has a clear span → evaluate with span F1
  - Implicit suggestions: "This is so slow" → no extractable span, evaluate
    by whether the INFERRED action is faithful, actionable, and feasible
  - Mixing metrics (span F1 for implicit) is methodologically wrong because
    implicit suggestions have no gold span — only a gold intent

Papers that support this split:
  - SemEval-2019 Task 9 already distinguishes explicit vs implicit
  - "Emotion meets coordination" (PLOS ONE, 2026) uses per-type agent paths
  - Multi-agent sentiment papers show type-specific agents outperform monolithic
"""

import json
import logging
import re
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SUGGESTION TYPE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class SuggestionType(Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    AMBIGUOUS = "ambiguous"    # sent to both paths, merged later


# Lexical signals for classification
EXPLICIT_SIGNALS = {
    "strong": [
        "please add", "you should", "i want", "i need", "can you add",
        "would be great if", "it would help if", "consider adding",
        "i wish", "i'd love", "how about adding", "why not add",
        "implement", "introduce", "provide", "enable", "allow",
        "make it possible", "add support for", "include",
    ],
    "moderate": [
        "why can't", "why doesn't", "why isn't there", "how come there's no",
        "there should be", "it should", "must have", "need to have",
        "could you", "would you", "is it possible to",
    ],
}

IMPLICIT_SIGNALS = {
    "complaint_to_fix": [
        "so slow", "too slow", "takes forever", "always crashes",
        "keeps freezing", "doesn't work", "broken", "terrible",
        "frustrating", "impossible to", "can't even", "unusable",
    ],
    "comparison_to_feature": [
        "like instagram", "like spotify", "like amazon",
        "competitors have", "other apps", "unlike", "compared to",
        "everyone else has", "industry standard",
    ],
    "rhetorical_question": [
        "why is there no", "how is it possible that",
        "am i the only one who", "does no one else",
        "seriously?", "really?", "come on",
    ],
    "sarcasm_reversal": [
        "oh great", "wonderful", "love how", "amazing that",
        "🙄", "/s", "sure, that works",
    ],
}


def classify_suggestion_type(
    text: str,
    confidence: float,
    is_implied: bool,
    source_modality: str = "text",
    pragmatic_view: dict = None,
) -> SuggestionType:
    """
    Classify whether a suggestion is explicit, implicit, or ambiguous.

    Rules:
      1. If labeler already marked is_implied=False AND confidence >= 0.80 → EXPLICIT
      2. If labeler marked is_implied=True → check if it's truly implicit
      3. If source_modality is "image" or "audio" only → always IMPLICIT
         (you can't have an explicit text request from an image)
      4. Lexical signal matching for edge cases
      5. Sarcasm/irony detected by pragmatic view → IMPLICIT
      6. If ambiguous signals from both sides → AMBIGUOUS (run both paths)
    """
    text_lower = text.lower().strip()
    prag = pragmatic_view or {}

    # Rule 3: Non-text modalities are always implicit
    if source_modality in ("image", "audio", "cross_modal"):
        return SuggestionType.IMPLICIT

    # Rule 5: Sarcasm → implicit (the real meaning is hidden)
    if prag.get("speech_act") == "sarcasm" or prag.get("tone") in ("sarcastic", "ironic"):
        return SuggestionType.IMPLICIT

    # Rule 1: Labeler says explicit with high confidence
    if not is_implied and confidence >= 0.80:
        # Verify with lexical check
        has_explicit_signal = any(
            sig in text_lower
            for signals in EXPLICIT_SIGNALS.values()
            for sig in signals
        )
        if has_explicit_signal:
            return SuggestionType.EXPLICIT

    # Rule 2: Labeler says implied
    if is_implied:
        return SuggestionType.IMPLICIT

    # Rule 4: Lexical signal matching for unlabeled or edge cases
    explicit_score = 0
    implicit_score = 0

    for strength, signals in EXPLICIT_SIGNALS.items():
        weight = 2.0 if strength == "strong" else 1.0
        for sig in signals:
            if sig in text_lower:
                explicit_score += weight

    for category, signals in IMPLICIT_SIGNALS.items():
        for sig in signals:
            if sig in text_lower:
                implicit_score += 1.5

    # Decision
    if explicit_score > 0 and implicit_score == 0:
        return SuggestionType.EXPLICIT
    elif implicit_score > 0 and explicit_score == 0:
        return SuggestionType.IMPLICIT
    elif explicit_score > 0 and implicit_score > 0:
        return SuggestionType.AMBIGUOUS
    else:
        # No clear signals — use confidence as tiebreaker
        return SuggestionType.EXPLICIT if confidence >= 0.80 else SuggestionType.IMPLICIT


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EXPLICIT SUGGESTION PATH
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExplicitResult:
    """Output of the explicit suggestion pipeline."""
    text: str
    suggestion_type: str = "explicit"
    confidence: float = 0.0
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    action_verb: str = ""           # "Add", "Fix", "Improve", etc.
    target_noun: str = ""           # "dark mode", "compression", etc.
    is_actionable: bool = True      # explicit suggestions are actionable by definition
    is_feasible: bool = True
    faithfulness_score: float = 1.0  # high — directly from user's words
    eval_metrics: dict = field(default_factory=dict)


def explicit_path(
    suggestion: dict,
    original_text: str,
) -> ExplicitResult:
    """
    EXPLICIT PATH — For suggestions the user directly stated.

    Steps:
      1. Verify span exists in source text (faithfulness check)
      2. Extract action verb + target noun (canonical form)
      3. Validate: is the request specific enough to be actionable?
      4. High confidence threshold: >= 0.80

    Evaluation approach:
      - Span F1 (token overlap with gold span)
      - Exact match (binary: did we extract the right suggestion?)
      - Precision@K (are top-K truly explicit suggestions?)
    """
    text = suggestion.get("text", "")
    conf = suggestion.get("confidence", 0.0)
    span_s = suggestion.get("span_start")
    span_e = suggestion.get("span_end")

    # Step 1: Span verification — is this text actually in the source?
    faithfulness = _verify_span_in_source(text, original_text)

    # Step 2: Extract action-target structure
    action_verb, target_noun = _extract_action_target(text)

    # Step 3: Actionability — explicit suggestions must be specific
    is_actionable = _check_explicit_actionability(text, action_verb, target_noun)

    # Step 4: Confidence gate
    passed = conf >= 0.80 and faithfulness >= 0.60 and is_actionable

    return ExplicitResult(
        text=text,
        confidence=conf if passed else conf * 0.5,   # penalize if failed checks
        span_start=span_s,
        span_end=span_e,
        action_verb=action_verb,
        target_noun=target_noun,
        is_actionable=is_actionable,
        is_feasible=True,   # explicit requests assumed feasible unless MCP says otherwise
        faithfulness_score=faithfulness,
        eval_metrics={
            "path": "explicit",
            "span_verified": faithfulness >= 0.60,
            "has_action_verb": bool(action_verb),
            "has_target_noun": bool(target_noun),
            "passed_threshold": passed,
        },
    )


def _verify_span_in_source(suggestion_text: str, original_text: str) -> float:
    """
    Check if the suggestion is grounded in the source text.
    For EXPLICIT suggestions, we expect high lexical overlap.
    Returns a faithfulness score 0-1.
    """
    if not original_text:
        return 0.0
    sugg_tokens = set(suggestion_text.lower().split())
    orig_tokens = set(original_text.lower().split())
    if not sugg_tokens:
        return 0.0
    overlap = len(sugg_tokens & orig_tokens)
    return round(overlap / len(sugg_tokens), 3)


def _extract_action_target(text: str) -> tuple[str, str]:
    """
    Extract (action_verb, target_noun) from a suggestion.
    E.g., "Add image compression before upload" → ("Add", "image compression before upload")
    """
    action_verbs = [
        "add", "implement", "fix", "improve", "enable", "allow",
        "introduce", "provide", "include", "remove", "reduce",
        "increase", "decrease", "optimize", "redesign", "update",
        "create", "build", "support", "make", "integrate",
    ]
    words = text.strip().split()
    if not words:
        return ("", "")

    first_word = words[0].lower().rstrip(",.:;")
    if first_word in action_verbs:
        return (words[0], " ".join(words[1:]))

    # Try to find action verb elsewhere
    for i, w in enumerate(words):
        if w.lower().rstrip(",.:;") in action_verbs:
            return (w, " ".join(words[i+1:]))

    return ("", text)


def _check_explicit_actionability(text: str, action_verb: str, target_noun: str) -> bool:
    """
    Explicit suggestion is actionable if:
      1. Has a clear action verb (Add/Fix/Improve/etc.)
      2. Has a specific target (not just "make it better")
      3. Target is concrete enough (>= 2 meaningful words)
    """
    if not action_verb:
        return False

    # "make it better" / "fix everything" are too vague
    vague_targets = {"it", "this", "that", "everything", "things", "stuff", "better", "worse"}
    target_words = set(target_noun.lower().split()) - {"the", "a", "an", "to", "for", "in", "on"}
    meaningful_words = target_words - vague_targets

    return len(meaningful_words) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 3. IMPLICIT SUGGESTION PATH
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ImplicitResult:
    """Output of the implicit suggestion pipeline."""
    text: str                        # the INFERRED suggestion
    suggestion_type: str = "implicit"
    original_complaint: str = ""     # what the user actually wrote
    inference_chain: str = ""        # reasoning: complaint → fix
    confidence: float = 0.0
    source_modality: str = "text"

    # THE EXTRA GATES for implicit (not needed for explicit)
    actionability_score: float = 0.0    # 1-5: can someone act on this?
    feasibility_score: float = 0.0      # 1-5: is this technically possible?
    specificity_score: float = 0.0      # 1-5: is it specific enough?
    faithfulness_score: float = 0.0     # 0-1: is inference grounded in source?

    is_actionable: bool = False
    is_feasible: bool = False
    passed_all_gates: bool = False
    eval_metrics: dict = field(default_factory=dict)


def implicit_path(
    suggestion: dict,
    original_text: str,
    image_context: str = None,
    audio_context: str = None,
    pragmatic_view: dict = None,
    semantic_view: dict = None,
) -> ImplicitResult:
    """
    IMPLICIT PATH — For suggestions inferred from complaints, comparisons, sarcasm.

    This is the HARDER path. The user didn't say "add X" — they said "this is terrible"
    and we INFERRED that they want X. So we need extra verification:

    Steps:
      1. INTENT RECONSTRUCTION: Map complaint → actionable fix
      2. FAITHFULNESS CHECK: Is the inferred fix grounded in what user actually said?
      3. ACTIONABILITY GATE: Can a product team act on this? (score 1-5)
      4. FEASIBILITY GATE: Is this technically possible? (score 1-5)
      5. SPECIFICITY GATE: Is it specific enough to implement? (score 1-5)

    Confidence threshold: >= 0.60 (lower than explicit, BUT must pass all 3 gates)

    Evaluation approach (DIFFERENT from explicit):
      - Intent Match (did we infer the RIGHT fix? semantic similarity to gold intent)
      - Actionability Score (human-rated 1-5, >= 3 to pass)
      - Feasibility Score (human-rated 1-5, >= 3 to pass)
      - Faithfulness (is inference grounded? not hallucinated?)
      - NO span F1 (there IS no gold span for implicit suggestions)
    """
    text = suggestion.get("text", "")
    conf = suggestion.get("confidence", 0.0)
    source_mod = suggestion.get("source_modality", "text")
    prag = pragmatic_view or {}
    sem = semantic_view or {}

    # Step 1: Intent reconstruction
    complaint = _extract_complaint(original_text, prag, sem)
    inference_chain = _build_inference_chain(complaint, text, source_mod, image_context)

    # Step 2: Faithfulness — is the inference grounded?
    faithfulness = _check_implicit_faithfulness(
        inferred_suggestion=text,
        original_text=original_text,
        complaint=complaint,
        image_context=image_context,
        audio_context=audio_context,
    )

    # Step 3: Actionability
    actionability = _score_actionability(text)

    # Step 4: Feasibility
    feasibility = _score_feasibility(text)

    # Step 5: Specificity
    specificity = _score_specificity(text)

    # Gate: ALL three must pass (>= 3.0 out of 5)
    passed_actionability = actionability >= 3.0
    passed_feasibility = feasibility >= 3.0
    passed_specificity = specificity >= 2.5   # slightly lower bar
    passed_faithfulness = faithfulness >= 0.40  # lower than explicit (0.60)

    # Confidence adjustment based on gates
    gate_penalty = 0.0
    if not passed_actionability:
        gate_penalty += 0.15
    if not passed_feasibility:
        gate_penalty += 0.10
    if not passed_specificity:
        gate_penalty += 0.10
    if not passed_faithfulness:
        gate_penalty += 0.20

    adjusted_conf = max(0.0, conf - gate_penalty)
    passed_all = (
        adjusted_conf >= 0.60
        and passed_actionability
        and passed_feasibility
        and passed_faithfulness
    )

    return ImplicitResult(
        text=text,
        original_complaint=complaint,
        inference_chain=inference_chain,
        confidence=round(adjusted_conf, 3),
        source_modality=source_mod,
        actionability_score=actionability,
        feasibility_score=feasibility,
        specificity_score=specificity,
        faithfulness_score=faithfulness,
        is_actionable=passed_actionability,
        is_feasible=passed_feasibility,
        passed_all_gates=passed_all,
        eval_metrics={
            "path": "implicit",
            "gate_actionability": passed_actionability,
            "gate_feasibility": passed_feasibility,
            "gate_specificity": passed_specificity,
            "gate_faithfulness": passed_faithfulness,
            "gate_penalty": round(gate_penalty, 3),
            "original_confidence": conf,
            "adjusted_confidence": round(adjusted_conf, 3),
            "passed_all_gates": passed_all,
        },
    )


def _extract_complaint(original_text: str, pragmatic_view: dict, semantic_view: dict) -> str:
    """Extract the core complaint from the original text."""
    # Use semantic view if available
    complaint = semantic_view.get("complaint_frame")
    if complaint and complaint != "null" and complaint != "none":
        return complaint

    # Fallback: extract negative phrases
    negative_markers = [
        "slow", "crash", "freeze", "broken", "terrible", "awful",
        "frustrating", "impossible", "useless", "annoying", "horrible",
        "doesn't work", "can't", "won't", "never", "always fails",
    ]
    sentences = original_text.split(".")
    for sent in sentences:
        if any(m in sent.lower() for m in negative_markers):
            return sent.strip()
    return original_text[:200]


def _build_inference_chain(
    complaint: str,
    inferred_fix: str,
    source_modality: str,
    image_context: str = None,
) -> str:
    """
    Build an explicit reasoning chain: complaint → inference → fix.
    This is logged for transparency and human review.
    """
    chain_parts = [f"User complaint: '{complaint}'"]

    if source_modality == "image" and image_context:
        chain_parts.append(f"Image evidence: '{image_context}'")
    elif source_modality == "cross_modal":
        chain_parts.append("Cross-modal signal: text + image combine to reveal issue")

    chain_parts.append(f"Inferred fix: '{inferred_fix}'")
    return " → ".join(chain_parts)


def _check_implicit_faithfulness(
    inferred_suggestion: str,
    original_text: str,
    complaint: str,
    image_context: str = None,
    audio_context: str = None,
) -> float:
    """
    For IMPLICIT suggestions, faithfulness means:
      "Is the inferred fix a REASONABLE response to the user's actual complaint?"

    This is DIFFERENT from explicit faithfulness (lexical overlap).
    Here we check SEMANTIC grounding:
      - Does the fix address the complaint topic?
      - Is the fix related to what the user described?
      - Did we NOT hallucinate a completely unrelated suggestion?

    Returns 0-1 faithfulness score.
    """
    # Combine all evidence sources
    evidence = original_text.lower()
    if image_context:
        evidence += " " + image_context.lower()
    if audio_context:
        evidence += " " + audio_context.lower()

    evidence_tokens = set(evidence.split())
    suggestion_tokens = set(inferred_suggestion.lower().split())

    # Remove stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "it", "i", "my", "to",
                 "of", "in", "on", "for", "and", "or", "but", "this", "that",
                 "add", "fix", "improve", "implement", "make", "should", "would"}
    evidence_meaningful = evidence_tokens - stopwords
    suggestion_meaningful = suggestion_tokens - stopwords

    if not suggestion_meaningful:
        return 0.0

    # Topic overlap: does the suggestion reference the same domain as the complaint?
    topic_overlap = len(suggestion_meaningful & evidence_meaningful)
    topic_ratio = topic_overlap / len(suggestion_meaningful)

    # Complaint-fix alignment: does the fix address the complaint?
    complaint_tokens = set(complaint.lower().split()) - stopwords
    fix_relevance = len(suggestion_meaningful & complaint_tokens) / max(1, len(complaint_tokens))

    # Combined score
    faithfulness = 0.6 * topic_ratio + 0.4 * fix_relevance
    return round(min(1.0, faithfulness), 3)


def _score_actionability(text: str) -> float:
    """
    Score 1-5: Can a product team act on this suggestion?

    5 = Immediately actionable ("Add a cancel button to the upload screen")
    4 = Clear action, minor ambiguity ("Improve the upload experience")
    3 = Needs interpretation but direction clear ("Make uploads faster")
    2 = Vague direction ("Fix the app")
    1 = Not actionable ("This sucks")
    """
    action_verbs = {"add", "implement", "fix", "improve", "enable", "remove",
                    "reduce", "increase", "create", "build", "integrate",
                    "redesign", "optimize", "support", "allow", "introduce"}

    words = text.lower().split()
    first_word = words[0].rstrip(".,!") if words else ""

    # Has action verb at start?
    has_action = first_word in action_verbs
    # Has specific target?
    meaningful_nouns = set(words) - action_verbs - {"the", "a", "an", "to", "for", "in"}
    has_target = len(meaningful_nouns) >= 2
    # Has quantifier or specific reference?
    has_specific = any(w in text.lower() for w in [
        "button", "screen", "page", "menu", "option", "feature", "mode",
        "speed", "time", "size", "font", "color", "layout", "upload",
        "download", "notification", "search", "filter", "sort",
    ])

    score = 1.0
    if has_action:
        score += 1.5
    if has_target:
        score += 1.0
    if has_specific:
        score += 1.0
    if len(words) >= 4:
        score += 0.5

    return min(5.0, round(score, 1))


def _score_feasibility(text: str) -> float:
    """
    Score 1-5: Is this technically possible to implement?

    5 = Standard feature ("Add dark mode")
    4 = Common but non-trivial ("Add real-time collaboration")
    3 = Possible but significant effort ("Rebuild the search engine")
    2 = Very hard ("Make the app work with no internet ever")
    1 = Impossible or nonsensical ("Read my mind")
    """
    # Known infeasible patterns
    infeasible = ["read my mind", "always work perfectly", "never crash",
                  "instant", "zero latency", "100% uptime", "free forever"]
    for pattern in infeasible:
        if pattern in text.lower():
            return 1.5

    # Known standard features (high feasibility)
    standard_features = [
        "dark mode", "filter", "sort", "search", "export", "notification",
        "font size", "compression", "cancel button", "undo", "bookmark",
        "offline", "password", "two-factor", "profile", "settings",
    ]
    for feat in standard_features:
        if feat in text.lower():
            return 4.5

    # Default: moderate feasibility
    return 3.5


def _score_specificity(text: str) -> float:
    """
    Score 1-5: How specific is this suggestion?

    5 = Very specific ("Add a progress bar with percentage to the upload screen")
    4 = Specific target ("Add image compression before upload")
    3 = Clear direction ("Make uploads faster")
    2 = Vague ("Improve the app")
    1 = Too vague ("Make it better")
    """
    words = text.split()
    word_count = len(words)

    if word_count <= 2:
        return 1.5
    elif word_count <= 4:
        return 2.5
    elif word_count <= 7:
        return 3.5
    elif word_count <= 12:
        return 4.0
    else:
        return 4.5


# ═══════════════════════════════════════════════════════════════════════════════
# 4. THE SWITCH — Routes suggestions through the right path
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SwitchResult:
    """Unified output after the switch merges explicit/implicit results."""
    text: str
    suggestion_type: str            # "explicit" | "implicit" | "ambiguous"
    confidence: float
    is_actionable: bool
    is_feasible: bool
    passed: bool                    # did it pass its type-specific pipeline?
    faithfulness_score: float

    # Type-specific fields
    action_verb: str = ""           # explicit only
    target_noun: str = ""           # explicit only
    inference_chain: str = ""       # implicit only
    actionability_score: float = 0  # implicit only (1-5)
    feasibility_score: float = 0    # implicit only (1-5)
    specificity_score: float = 0    # implicit only (1-5)

    # For ranking
    type_adjusted_score: float = 0.0
    eval_metrics: dict = field(default_factory=dict)


def run_suggestion_switch(
    suggestion: dict,
    original_text: str,
    image_context: str = None,
    audio_context: str = None,
    pragmatic_view: dict = None,
    semantic_view: dict = None,
) -> SwitchResult:
    """
    THE SWITCH: classify → route → evaluate → merge.

    This is the main entry point. For each suggestion:
      1. Classify as explicit/implicit/ambiguous
      2. Route through the appropriate pipeline
      3. Apply type-specific evaluation
      4. Return unified result with type-adjusted score

    For AMBIGUOUS cases: run BOTH paths, take the one with higher confidence.
    """
    # Step 1: Classify
    stype = classify_suggestion_type(
        text=suggestion.get("text", ""),
        confidence=suggestion.get("confidence", 0.5),
        is_implied=suggestion.get("is_implied", False),
        source_modality=suggestion.get("source_modality", "text"),
        pragmatic_view=pragmatic_view,
    )

    logger.info(
        f"[switch] '{suggestion.get('text', '')[:50]}' → {stype.value} "
        f"(conf={suggestion.get('confidence', 0):.2f})"
    )

    # Step 2: Route
    if stype == SuggestionType.EXPLICIT:
        result = _route_explicit(suggestion, original_text)
    elif stype == SuggestionType.IMPLICIT:
        result = _route_implicit(
            suggestion, original_text,
            image_context, audio_context,
            pragmatic_view, semantic_view,
        )
    else:
        # AMBIGUOUS: run both, pick winner
        explicit_result = _route_explicit(suggestion, original_text)
        implicit_result = _route_implicit(
            suggestion, original_text,
            image_context, audio_context,
            pragmatic_view, semantic_view,
        )
        # Pick the one that passed with higher confidence
        if explicit_result.passed and not implicit_result.passed:
            result = explicit_result
            result.suggestion_type = "explicit"
        elif implicit_result.passed and not explicit_result.passed:
            result = implicit_result
            result.suggestion_type = "implicit"
        elif explicit_result.confidence >= implicit_result.confidence:
            result = explicit_result
            result.suggestion_type = "explicit"
        else:
            result = implicit_result
            result.suggestion_type = "implicit"

        result.eval_metrics["ambiguous_resolution"] = result.suggestion_type

    return result


def _route_explicit(suggestion: dict, original_text: str) -> SwitchResult:
    """Route through explicit path, convert to SwitchResult."""
    er = explicit_path(suggestion, original_text)
    return SwitchResult(
        text=er.text,
        suggestion_type="explicit",
        confidence=er.confidence,
        is_actionable=er.is_actionable,
        is_feasible=er.is_feasible,
        passed=er.eval_metrics.get("passed_threshold", False),
        faithfulness_score=er.faithfulness_score,
        action_verb=er.action_verb,
        target_noun=er.target_noun,
        type_adjusted_score=er.confidence * 1.1 if er.is_actionable else er.confidence * 0.8,
        eval_metrics=er.eval_metrics,
    )


def _route_implicit(
    suggestion, original_text, image_context, audio_context,
    pragmatic_view, semantic_view,
) -> SwitchResult:
    """Route through implicit path, convert to SwitchResult."""
    ir = implicit_path(
        suggestion, original_text,
        image_context, audio_context,
        pragmatic_view, semantic_view,
    )
    return SwitchResult(
        text=ir.text,
        suggestion_type="implicit",
        confidence=ir.confidence,
        is_actionable=ir.is_actionable,
        is_feasible=ir.is_feasible,
        passed=ir.passed_all_gates,
        faithfulness_score=ir.faithfulness_score,
        inference_chain=ir.inference_chain,
        actionability_score=ir.actionability_score,
        feasibility_score=ir.feasibility_score,
        specificity_score=ir.specificity_score,
        type_adjusted_score=ir.confidence * 1.0 if ir.passed_all_gates else ir.confidence * 0.6,
        eval_metrics=ir.eval_metrics,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TYPE-SPECIFIC EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class ExplicitEvaluator:
    """
    Evaluation metrics for EXPLICIT suggestions.
    These have gold spans → use span-based metrics.
    """

    @staticmethod
    def span_f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
        """Token-level F1 between predicted span and gold span."""
        pred_set = set(t.lower() for t in pred_tokens)
        gold_set = set(t.lower() for t in gold_tokens)
        if not pred_set or not gold_set:
            return 0.0
        tp = len(pred_set & gold_set)
        precision = tp / len(pred_set)
        recall = tp / len(gold_set)
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)

    @staticmethod
    def exact_match(pred_text: str, gold_text: str) -> bool:
        """Exact match after normalization."""
        normalize = lambda t: " ".join(t.lower().strip().split())
        return normalize(pred_text) == normalize(gold_text)

    @staticmethod
    def evaluate_batch(predictions: list[dict], gold: list[dict]) -> dict:
        """
        Evaluate a batch of explicit predictions against gold labels.

        Returns:
            dict with span_f1, exact_match_rate, precision, recall, f1
        """
        evaluator = ExplicitEvaluator()
        span_f1s = []
        exact_matches = 0
        tp, fp, fn = 0, 0, 0

        gold_texts = {g["text"].lower().strip() for g in gold if g.get("suggestion_type") == "explicit"}

        for pred in predictions:
            pred_text = pred["text"].lower().strip()
            # Find best matching gold
            best_f1 = 0
            best_gold = None
            for g in gold:
                if g.get("suggestion_type") != "explicit":
                    continue
                f1 = evaluator.span_f1(pred_text.split(), g["text"].lower().split())
                if f1 > best_f1:
                    best_f1 = f1
                    best_gold = g

            span_f1s.append(best_f1)
            if best_gold and evaluator.exact_match(pred["text"], best_gold["text"]):
                exact_matches += 1

            if best_f1 >= 0.5:
                tp += 1
            else:
                fp += 1

        fn = len(gold_texts) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "type": "explicit",
            "avg_span_f1": round(sum(span_f1s) / len(span_f1s), 4) if span_f1s else 0,
            "exact_match_rate": round(exact_matches / len(predictions), 4) if predictions else 0,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n_predictions": len(predictions),
            "n_gold": len(gold_texts),
        }


class ImplicitEvaluator:
    """
    Evaluation metrics for IMPLICIT suggestions.
    These have NO gold spans → use intent-based and quality metrics.

    Key difference from explicit:
      - No span F1 (meaningless for implicit)
      - Instead: intent match, actionability, feasibility, faithfulness
    """

    @staticmethod
    def intent_match(pred_text: str, gold_intent: str) -> float:
        """
        Semantic similarity between predicted suggestion and gold intent.
        Uses Jaccard as proxy (replace with embedding similarity in production).
        """
        pred_tokens = set(pred_text.lower().split())
        gold_tokens = set(gold_intent.lower().split())
        stopwords = {"the", "a", "an", "to", "of", "in", "for", "and", "or", "is", "it"}
        pred_meaningful = pred_tokens - stopwords
        gold_meaningful = gold_tokens - stopwords
        if not pred_meaningful or not gold_meaningful:
            return 0.0
        overlap = len(pred_meaningful & gold_meaningful)
        union = len(pred_meaningful | gold_meaningful)
        return round(overlap / union, 4)

    @staticmethod
    def evaluate_batch(predictions: list[dict], gold: list[dict]) -> dict:
        """
        Evaluate a batch of implicit predictions.

        Metrics:
          - Intent match score (semantic similarity to gold intent)
          - Avg actionability (from the implicit path gates)
          - Avg feasibility
          - Avg faithfulness
          - Gate pass rate (% that passed all gates)
          - Precision/Recall/F1 at intent level
        """
        evaluator = ImplicitEvaluator()
        intent_scores = []
        actionabilities = []
        feasibilities = []
        faithfulnesses = []
        gate_passes = 0
        tp, fp = 0, 0

        gold_intents = [g for g in gold if g.get("suggestion_type") == "implicit"]

        for pred in predictions:
            # Find best matching gold intent
            best_intent_score = 0
            for g in gold_intents:
                score = evaluator.intent_match(pred["text"], g.get("text", ""))
                best_intent_score = max(best_intent_score, score)

            intent_scores.append(best_intent_score)

            # Collect gate scores
            metrics = pred.get("eval_metrics", {})
            if "actionability_score" in pred:
                actionabilities.append(pred["actionability_score"])
            if "feasibility_score" in pred:
                feasibilities.append(pred["feasibility_score"])
            if "faithfulness_score" in pred:
                faithfulnesses.append(pred["faithfulness_score"])
            if pred.get("passed_all_gates") or metrics.get("passed_all_gates"):
                gate_passes += 1

            if best_intent_score >= 0.3:   # lower threshold — intent matching is hard
                tp += 1
            else:
                fp += 1

        fn = len(gold_intents) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        n = len(predictions) or 1
        return {
            "type": "implicit",
            "avg_intent_match": round(sum(intent_scores) / n, 4) if intent_scores else 0,
            "avg_actionability": round(sum(actionabilities) / n, 2) if actionabilities else 0,
            "avg_feasibility": round(sum(feasibilities) / n, 2) if feasibilities else 0,
            "avg_faithfulness": round(sum(faithfulnesses) / n, 4) if faithfulnesses else 0,
            "gate_pass_rate": round(gate_passes / n, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n_predictions": len(predictions),
            "n_gold": len(gold_intents),
        }


class UnifiedEvaluator:
    """
    Combines explicit and implicit evaluation with type-aware weighting.
    This is what you report in the paper.
    """

    @staticmethod
    def evaluate_all(
        explicit_preds: list[dict],
        implicit_preds: list[dict],
        gold: list[dict],
    ) -> dict:
        explicit_eval = ExplicitEvaluator.evaluate_batch(explicit_preds, gold)
        implicit_eval = ImplicitEvaluator.evaluate_batch(implicit_preds, gold)

        # Unified metrics (weighted by count)
        n_explicit = len(explicit_preds)
        n_implicit = len(implicit_preds)
        total = n_explicit + n_implicit

        if total == 0:
            return {"explicit": explicit_eval, "implicit": implicit_eval, "unified": {}}

        # Weighted average F1
        unified_f1 = (
            explicit_eval["f1"] * n_explicit + implicit_eval["f1"] * n_implicit
        ) / total

        # Weighted precision/recall
        unified_prec = (
            explicit_eval["precision"] * n_explicit + implicit_eval["precision"] * n_implicit
        ) / total
        unified_rec = (
            explicit_eval["recall"] * n_explicit + implicit_eval["recall"] * n_implicit
        ) / total

        return {
            "explicit": explicit_eval,
            "implicit": implicit_eval,
            "unified": {
                "weighted_f1": round(unified_f1, 4),
                "weighted_precision": round(unified_prec, 4),
                "weighted_recall": round(unified_rec, 4),
                "total_predictions": total,
                "explicit_ratio": round(n_explicit / total, 3),
                "implicit_ratio": round(n_implicit / total, 3),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LANGGRAPH NODE — integrates into your existing pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def suggestion_switch_node(state: dict) -> dict:
    """
    LangGraph node that runs the suggestion switch on all accepted suggestions.
    Sits between arbitration_node and canonicaliser_node in the pipeline.

    New pipeline topology:
      ... → arbitration → SUGGESTION_SWITCH → canonicaliser → cluster → ...
    """
    logger.info(f"[suggestion_switch] {state['sample_id']}")

    accepted = state.get("accepted_suggestions", [])
    original_text = state.get("clean_text", "")
    prag = state.get("text_pragmatic_view") or {}
    sem = state.get("text_semantic_view") or {}
    captions = state.get("image_captions") or []
    first_cap = captions[0] if captions else {}
    img_ctx = first_cap.get("caption") if isinstance(first_cap, dict) else None
    audio_ctx = state.get("audio_transcript")

    explicit_results = []
    implicit_results = []
    all_results = []

    for suggestion in accepted:
        result = run_suggestion_switch(
            suggestion=suggestion,
            original_text=original_text,
            image_context=img_ctx,
            audio_context=audio_ctx,
            pragmatic_view=prag,
            semantic_view=sem,
        )

        result_dict = {
            **suggestion,
            "suggestion_type": result.suggestion_type,
            "switch_confidence": result.confidence,
            "is_actionable": result.is_actionable,
            "is_feasible": result.is_feasible,
            "faithfulness_score": result.faithfulness_score,
            "passed_switch": result.passed,
            "type_adjusted_score": result.type_adjusted_score,
            "inference_chain": result.inference_chain,
            "actionability_score": result.actionability_score,
            "feasibility_score": result.feasibility_score,
            "specificity_score": result.specificity_score,
            "switch_eval_metrics": result.eval_metrics,
        }

        all_results.append(result_dict)

        if result.suggestion_type == "explicit":
            explicit_results.append(result_dict)
        else:
            implicit_results.append(result_dict)

    # Filter: only keep suggestions that passed their type-specific pipeline
    passed = [r for r in all_results if r["passed_switch"]]
    failed = [r for r in all_results if not r["passed_switch"]]

    logger.info(
        f"[suggestion_switch] {len(accepted)} in → "
        f"{len(passed)} passed ({len(explicit_results)} explicit, "
        f"{len(implicit_results)} implicit) | {len(failed)} failed gates"
    )

    return {
        "accepted_suggestions": passed,
        "rejected_suggestions": state.get("rejected_suggestions", []) + [
            {"text": r["text"], "reason": "failed_switch_gates", **r.get("switch_eval_metrics", {})}
            for r in failed
        ],
        "switch_stats": {
            "total_input": len(accepted),
            "explicit_count": len(explicit_results),
            "implicit_count": len(implicit_results),
            "passed_count": len(passed),
            "failed_count": len(failed),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEMO — Run on your dataset
# ═══════════════════════════════════════════════════════════════════════════════

def demo_with_dataset():
    """Demo the switch on sample entries from the dataset."""
    samples = [
        {
            "text": "add split-screen multitasking mode",
            "confidence": 0.90, "is_implied": False,
            "source_modality": "text",
            "original": "Would be great if the app had a split-screen mode for multitasking.",
        },
        {
            "text": "fix app freezing issue that occurs within 10 minutes of use",
            "confidence": 0.75, "is_implied": True,
            "source_modality": "text",
            "original": "I literally cannot use this app for more than 10 minutes without it freezing.",
        },
        {
            "text": "add verified purchase review filter",
            "confidence": 0.70, "is_implied": True,
            "source_modality": "text",
            "original": "Why is there no way to filter reviews by verified purchase?",
        },
        {
            "text": "improve packaging to prevent product damage during shipping",
            "confidence": 0.65, "is_implied": True,
            "source_modality": "image",
            "original": "Got my package today!",
        },
        {
            "text": "make patient portal accessible on mobile devices",
            "confidence": 0.88, "is_implied": False,
            "source_modality": "text",
            "original": "Please make the patient portal accessible on mobile.",
        },
        {
            "text": "reduce appointment wait times for elderly patients",
            "confidence": 0.72, "is_implied": True,
            "source_modality": "text",
            "original": "My mother waited 3 hours past her appointment time. She's 82.",
        },
        {
            "text": "add comprehensive allergen guide to menu",
            "confidence": 0.92, "is_implied": False,
            "source_modality": "text",
            "original": "The menu needs a proper allergen guide. My daughter has a nut allergy.",
        },
    ]

    print("\n" + "="*80)
    print("  EXPLICIT / IMPLICIT SUGGESTION SWITCH — DEMO")
    print("="*80)

    explicit_preds = []
    implicit_preds = []

    for s in samples:
        result = run_suggestion_switch(
            suggestion=s,
            original_text=s["original"],
        )

        icon = "✅" if result.passed else "❌"
        type_icon = "📝" if result.suggestion_type == "explicit" else "🔍"

        print(f"\n{type_icon} [{result.suggestion_type.upper():8s}] {icon} {result.text}")
        print(f"   Confidence: {result.confidence:.2f} | Faithful: {result.faithfulness_score:.2f}")
        print(f"   Actionable: {result.is_actionable} | Feasible: {result.is_feasible}")

        if result.suggestion_type == "implicit":
            print(f"   Actionability: {result.actionability_score}/5 | "
                  f"Feasibility: {result.feasibility_score}/5 | "
                  f"Specificity: {result.specificity_score}/5")
            if result.inference_chain:
                print(f"   Chain: {result.inference_chain[:100]}")
            implicit_preds.append(asdict(result))
        else:
            print(f"   Action: '{result.action_verb}' Target: '{result.target_noun}'")
            explicit_preds.append(asdict(result))

    # Run evaluation
    gold = [
        {"text": "add split-screen multitasking mode", "suggestion_type": "explicit"},
        {"text": "make patient portal accessible on mobile devices", "suggestion_type": "explicit"},
        {"text": "add comprehensive allergen guide to menu", "suggestion_type": "explicit"},
        {"text": "fix app freezing issue", "suggestion_type": "implicit"},
        {"text": "add verified purchase review filter", "suggestion_type": "implicit"},
        {"text": "improve packaging quality", "suggestion_type": "implicit"},
        {"text": "reduce appointment wait times", "suggestion_type": "implicit"},
    ]

    print("\n" + "="*80)
    print("  EVALUATION RESULTS")
    print("="*80)

    eval_results = UnifiedEvaluator.evaluate_all(explicit_preds, implicit_preds, gold)

    for section, metrics in eval_results.items():
        print(f"\n  [{section.upper()}]")
        for k, v in metrics.items():
            print(f"    {k:30s}: {v}")


if __name__ == "__main__":
    demo_with_dataset()
