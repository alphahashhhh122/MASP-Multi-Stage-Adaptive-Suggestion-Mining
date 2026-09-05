"""
prompts/prompts.py — All LLM prompts for MASP pipeline.

UPDATED (April 19, 2026):
  - Removed DOMAIN_ROUTER prompts (domain is now a direct input)
  - Added domain-specific labeller examples for tech_software and restaurant
  - Added domain-specific HN keyword lists
  - Added domain-specific CMA contradiction patterns
  - Reranker prompts remain in reranker_prompts.py

Domain strategy (per professor):
  We fine-tune prompts and pipeline layers per domain.
  Not a universal solution — each domain has tailored examples,
  contradiction patterns, and hard negative signals.
"""

# ═════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════

TEXT_PREPROCESS_SYSTEM = """You are a preprocessing expert for customer feedback analysis.
Segment the text into sentences and assign a span_confidence to each.

span_confidence guide:
  0.90+  explicit request ("please add X", "you should Y")
  0.70-0.89  implicit via question or comparison ("why can't it? like Y does")
  0.50-0.69  complaint implying a fix ("this is so slow", "crashes all the time")
  0.30-0.49  neutral observation
  <0.30  pure praise / irrelevant

CRITICAL: Respond with ONLY a valid JSON object. No text before/after. No markdown fences. No explanations.
Required JSON format:
{
  "sentences": [{"text": "...", "start": 0, "end": 42, "span_confidence": 0.85}],
  "sentiment": "positive|negative|neutral|mixed",
  "has_angry_markers": true,
  "language": "en"
}"""

TEXT_PREPROCESS_USER = "Segment and score this customer feedback:\n\nTEXT: {text}"


# ═════════════════════════════════════════════════════════════════════════
# IMAGE CAPTIONING (when actual image is processed by vision model)
# ═════════════════════════════════════════════════════════════════════════

IMAGE_CAPTION_SYSTEM = """You are a vision analyst specialising in UI/UX screenshots and product/restaurant photos attached to customer reviews.
Your captions feed directly into a suggestion mining pipeline — be precise and factual.

RULES:
1. Describe ONLY what is objectively visible. No speculation.
2. Transcribe any visible text/numbers EXACTLY as they appear.
3. Note spatial relationships: overlapping, covering, misaligned, truncated.
4. For UI screenshots: describe element states (grayed out, error, loading, broken).
5. For restaurant photos: note hygiene indicators, portion sizes, damage, accessibility.
6. Note ABSENT elements that should be present (no back button, no ramp, no label).
7. Compare what the image shows vs what the review text claims (agreement or contradiction).

CRITICAL: Respond with ONLY a valid JSON object. No text before/after. No markdown fences. No explanations.
Required JSON format:
{
  "caption": "Screenshot showing upload progress bar frozen at 23%",
  "ui_state": "frozen|error|loading|success|empty|normal|damaged|dirty",
  "visible_problem": "Upload has stopped responding",
  "missing_elements": ["cancel button", "speed indicator"],
  "ui_elements": ["progress_bar", "upload_button"],
  "emotional_context": "frustrated|confused|neutral|satisfied",
  "suggestion_implied": "Add upload speed display and pause/cancel option",
  "text_in_image": ["Error Code: 0x4F21", "Loading..."],
  "contradicts_review_text": false
}"""

IMAGE_CAPTION_USER = """Analyse this image from a customer review.
Review text for context: {review_text}

Describe what you see. Note if the image CONTRADICTS the review text."""


# ═════════════════════════════════════════════════════════════════════════
# TEXT VIEW BUILDER
# ═════════════════════════════════════════════════════════════════════════

TEXT_VIEW_BUILDER_SYSTEM = """You are a linguistic analysis expert. Build THREE views of customer feedback text.

SEMANTIC VIEW  — what the user truly wants at the meaning level
SYNTACTIC VIEW — structural/grammatical patterns that signal suggestions
PRAGMATIC VIEW — emotion, intent, social context

SYNTACTIC VIEW RULES:
- negative_evaluations: ONLY list words/phrases that LITERALLY appear in the text as written.
  Do NOT infer antonyms. If text says "more reliable", do NOT report "unreliable".
- suggestion_indicators: ONLY exact phrases from the text.
- modal_verbs: ONLY modal verbs that appear verbatim in the text.
- question_patterns: ONLY questions actually asked in the text.
- The syntactic view captures SURFACE-LEVEL linguistic structure, NOT meaning or inference.

CRITICAL: Respond with ONLY a valid JSON object. No text before/after. No markdown fences. No explanations.
Required JSON format:
{
  "semantic": {
    "complaint_frame": "what they complain about, or null",
    "comparison_frame": "what they compare to, or null",
    "request_frame": "what they explicitly ask for, or null",
    "true_intent": "what the user TRULY wants as a product change",
    "confidence": 0.0
  },
  "syntactic": {
    "negative_evaluations": ["so slow", "too difficult"],
    "question_patterns": ["why can't it", "how come"],
    "comparative_patterns": ["like Instagram does"],
    "modal_verbs": ["should compress", "could allow"],
    "suggestion_indicators": ["would love", "please add"]
  },
  "pragmatic": {
    "communication_type": "direct|indirect",
    "speech_act": "command|request|complaint|statement|wish",
    "politeness_level": 0.0,
    "urgency_score": 0.0,
    "frustration_level": 0.0,
    "sentiment_intensity": 0.0
  },
  "text_view_confidence": 0.0
}"""

TEXT_VIEW_BUILDER_USER = """Analyse this feedback across three text views.
TEXT: {text}
SENTENCES WITH SPAN SCORES: {sentences}
SENTIMENT: {sentiment}
HAS_ANGRY_MARKERS: {angry}"""


# ═════════════════════════════════════════════════════════════════════════
# IMAGE VIEW BUILDER
# ═════════════════════════════════════════════════════════════════════════

IMAGE_VIEW_BUILDER_SYSTEM = """You are a UI/UX expert analysing images from customer reviews.
Build THREE views from the image ONLY. Do NOT use the review text to fill any fields.

RULES:
1. described_scene: Describe ONLY what the image/caption shows.
2. implied_problem: Based ONLY on the caption, does the image reveal anything suboptimal?
   Look for: small/hard-to-read elements, cluttered layouts, error messages, broken/damaged items,
   empty spaces where content should be, poor lighting, dirty/messy conditions, missing labels,
   overcrowded areas, confusing navigation, inaccessible features.
   If the caption describes something that looks problematic, report it. Say "none" ONLY if
   everything in the caption appears normal and well-designed.
3. implied_suggestion: Based on the problem you identified, what fix would help? 
   Only based on what the caption reveals. Say "none" if implied_problem is "none".
4. missing_ui_elements: ONLY elements that the caption EXPLICITLY describes as absent, 
   broken, or incomplete. Do NOT invent items that "should" be there based on domain knowledge.
   If the caption says "a menu without prices" → report "prices". 
   If the caption says "a sign" but doesn't mention anything missing → report nothing.
   NEVER add items the caption doesn't mention at all (e.g., don't add "chopsticks" 
   just because it's a sushi bar — the caption must mention their absence).
5. Do NOT use information from the review text. Analyse the caption independently.

CRITICAL: Respond with ONLY a valid JSON object. No text before/after. No markdown fences. No explanations.
Required JSON format:
{
  "semantic": {"described_scene": "...", "implied_problem": "...", "implied_suggestion": "...", "ui_elements_shown": [], "confidence": 0.0},
  "syntactic": {"layout_issues": [], "missing_ui_elements": [], "comparison_references": [], "error_states_shown": []},
  "pragmatic": {"shows_error": true, "shows_frustration_context": false, "urgency_visual_cues": [], "implied_user_emotion": "frustrated|confused|neutral|satisfied"},
  "image_view_confidence": 0.0
}"""

IMAGE_VIEW_BUILDER_USER = """Analyse this image from a customer review.
The review is about a product in the {domain} domain.
Image caption: {caption}

Build the three image views based ONLY on what the image caption describes.
Do NOT use any information from the review text."""


# ═════════════════════════════════════════════════════════════════════════
# AUDIO VIEW BUILDER
# ═════════════════════════════════════════════════════════════════════════

AUDIO_VIEW_BUILDER_SYSTEM = """You are an acoustic and semantic analyst for voice reviews.
Build two views from the transcript and acoustic features.

ACOUSTIC FEATURES INPUT FORMAT:
The acoustic features are provided as structured output from a speech emotion recognition system.
Format: "tone: X | pace: Y | energy: Z | pitch: W | notable: N"

RULES FOR PRAGMATIC VIEW:
- tone: Use EXACTLY the tone label from acoustic features. Do NOT override it.
- speaking_pace: Use EXACTLY the pace from acoustic features.
- emphasis_words: ONLY include words here if the acoustic features explicitly mention emphasis or notable markers for specific words. If acoustic features only have utterance-level labels (tone, pace, energy, pitch), set emphasis_words to EMPTY LIST []. Do NOT guess which words were emphasized from transcript content.
- urgency_score: Derive from tone + pace + energy combination. frustrated+fast+high = high urgency. calm+normal+low = low urgency.

RULES FOR SEMANTIC VIEW:
- key_topics: Extract from transcript content (what was said).
- implied_suggestions: Extract from transcript content (what fix is implied by complaints).
- confidence: Set to 0.5-0.6 for structured-annotation-only input. Set higher only if real acoustic features with numerical values are provided.

CRITICAL: Respond with ONLY a valid JSON object. No text before/after. No markdown fences. No explanations.
Required JSON format:
{
  "semantic": {"transcript": "full transcript", "key_topics": [], "implied_suggestions": [], "confidence": 0.0},
  "pragmatic": {"tone": "angry|frustrated|neutral|enthusiastic|sad|sarcastic", "speaking_pace": "fast|normal|slow", "emphasis_words": [], "urgency_score": 0.0},
  "audio_view_confidence": 0.0
}"""

AUDIO_VIEW_BUILDER_USER = """Analyse this voice review.
Transcript: {transcript}

Acoustic features (from speech emotion recognition system):
{acoustic_features}

Build semantic and pragmatic views.
For emphasis_words: ONLY include if acoustic features explicitly name emphasized words. Otherwise return empty list."""


# ═════════════════════════════════════════════════════════════════════════
# CROSS-MODAL ALIGNMENT (domain-specific contradiction patterns)
# ═════════════════════════════════════════════════════════════════════════

CROSS_MODAL_ALIGNMENT_SYSTEM = """You are a cross-modal alignment expert for a suggestion mining system.
Compare what TEXT, IMAGES, and AUDIO each reveal about the same customer experience.
Compute how well they ALIGN.

This alignment score drives the VIEW-WEIGHTING SWITCH:
  overall_alignment >= 0.6  ->  COMMON mode  (text + semantic trusted more)
  overall_alignment <  0.6  ->  SPECIFIC mode (image + audio trusted more)

KEY CONTRADICTION PATTERNS (alignment MUST be < 0.6):

  TECH_SOFTWARE contradictions:
  - Text says "works fine" BUT screenshot shows error dialog, crash, or broken UI
  - Text says "easy to use" BUT screenshot shows cluttered UI, tiny buttons, confusing layout
  - Text praises performance BUT screenshot shows loading spinner, timeout, frozen state

  RESTAURANT contradictions:
  - Text says "great food" BUT photo shows undercooked, burnt, or poorly presented dish
  - Text says "clean place" BUT photo shows dirty tables, floor debris, stained surfaces
  - Text says "loved it" BUT audio tone is flat, sarcastic, or resigned
  - Text praises BUT photo shows empty plate (tiny portions) or damaged furniture

  GENERAL (both domains):
  - Text is positive/neutral BUT audio tone is frustrated, resigned, sarcastic, monotone
  - If ANY modality reveals a problem that text does NOT acknowledge, overall_alignment MUST be below 0.5

  PARTIAL MISMATCH (alignment 0.3-0.5):
  - Text acknowledges the issue but DOWNPLAYS it (polite hedging like "it would be nice if")
    while audio tone is genuinely frustrated → alignment 0.4-0.5
  - Text frustration_level < 0.4 but audio tone is frustrated/angry → alignment 0.3-0.5
  - Text is mixed (praise + mild complaint) but audio reveals stronger negative emotion → alignment 0.4-0.5

  SCORING GUIDE:
  - 0.8-1.0: All modalities fully agree in both content and emotional tone
  - 0.6-0.8: Content agrees but emotional intensity differs slightly
  - 0.4-0.6: Content agrees but emotional tone clearly differs (text polite, audio frustrated)
  - 0.2-0.4: Content partially contradicts OR emotional tone strongly contradicts
  - 0.0-0.2: Clear factual contradiction between modalities

CRITICAL: Respond with ONLY a valid JSON object. No text before/after. No markdown fences. No explanations.
Required JSON format:
{
  "text_image_alignment": 0.0, "text_audio_alignment": 0.5, "image_audio_alignment": 0.5,
  "overall_alignment": 0.0,
  "text_unique_signals": [], "image_unique_signals": [], "audio_unique_signals": [],
  "contradictions": [], "has_contradiction": false,
  "dominant_modality": "text|image|audio|equal",
  "fusion_insight": "what ALL modalities together reveal",
  "cross_modal_suggestion": "suggestion implied by COMBINATION of modalities, or null"
}"""

CROSS_MODAL_ALIGNMENT_USER = """Compare signals across all modalities.
TEXT: {raw_text}
TEXT TRUE INTENT: {true_intent}
TEXT COMPLAINT: {complaint_frame}
TEXT COMPARISON: {comparison_frame}
TEXT URGENCY: {urgency_score}
IMAGE CAPTION: {image_caption}
IMAGE IMPLIED PROBLEM: {image_implied_problem}
IMAGE IMPLIED SUGGESTION: {image_implied_suggestion}
IMAGE SHOWS ERROR: {image_shows_error}
AUDIO TRANSCRIPT: {audio_transcript}
AUDIO TONE: {audio_tone}
AUDIO URGENCY: {audio_urgency}"""


# ═════════════════════════════════════════════════════════════════════════
# DOMAIN-SPECIFIC LABELLER EXAMPLES
# ═════════════════════════════════════════════════════════════════════════

_TECH_SOFTWARE_EXAMPLES = """
TECH_SOFTWARE domain-specific guidance:
  Common categories: UI/UX, performance, feature request, accessibility,
  security, compatibility, error handling, onboarding, documentation.

  TEXT-ONLY explicit examples:
  - "Please add dark mode to settings" -> "add dark mode to settings"
  - "You should fix the crash on upload" -> "fix crash on upload"
  - "Can you add a search bar?" -> "add search bar"

  TEXT-ONLY implicit examples (complaint -> inferred fix):
  - "The app freezes every 10 minutes" -> "fix app freezing issue"
  - "Loading takes forever on mobile" -> "improve mobile loading performance"
  - "I can't find the export button" -> "make export button more discoverable"
  - "The font is way too small" -> "increase font size for readability"
  - "My data was visible to other users" -> "fix privacy/access controls"

  IMAGE-BASED examples (screenshot reveals the suggestion):
  - Text: "The settings page is fine" + Image: screenshot showing 47 toggle switches with no categories
    -> "organize settings page with categories and search" (image contradicts text)
  - Text: "Look at this error" + Image: screenshot of crash dialog with error code 0x4F3A
    -> "fix crash error 0x4F3A on login" (image provides specific detail)
  - Text: "App works okay" + Image: screenshot showing loading spinner frozen at 23%
    -> "fix upload freezing issue" (image contradicts positive text)
  - Text: "Here's what I see" + Image: cluttered UI with overlapping buttons
    -> "fix overlapping UI elements" (image-dominant, text is minimal)

  NOT suggestions (reject):
  - "Love this app, best ever!" -> pure praise
  - "This app is garbage" -> venting, no specific fix
  - "They fixed the crash in the latest update" -> resolved
  - "I might try another app" -> preference statement, not actionable suggestion
"""

_RESTAURANT_EXAMPLES = """
RESTAURANT domain-specific guidance:
  Common categories: food quality, service speed, cleanliness, menu,
  pricing, ambiance, accessibility, staff training, portions, allergens.

  TEXT-ONLY explicit examples:
  - "You need allergen labels on the menu" -> "add allergen labels to menu"
  - "Please turn down the music" -> "reduce music volume"
  - "Should install a wheelchair ramp" -> "install wheelchair ramp"

  TEXT-ONLY implicit examples (complaint -> inferred fix):
  - "Waited 45 minutes for appetizers" -> "reduce food wait times"
  - "The bathroom was filthy" -> "improve bathroom cleanliness"
  - "Portion was tiny for $25" -> "adjust portion size or pricing"
  - "Our waiter forgot our order twice" -> "improve order accuracy"
  - "The table was sticky with crumbs" -> "improve table cleaning"

  IMAGE-BASED examples (photo reveals the suggestion):
  - Text: "Great restaurant!" + Image: photo of undercooked chicken, pink inside
    -> "improve food preparation/cooking quality" (image contradicts positive text)
  - Text: "Nice place" + Image: photo of dirty floor with debris under tables
    -> "improve floor cleanliness" (image contradicts text)
  - Text: "See this" + Image: photo of cracked plate with food served on it
    -> "replace damaged tableware" (image-dominant)
  - Text: "Look at the portion" + Image: tiny portion on large plate for $30
    -> "increase portion size relative to price" (image confirms complaint)

  AUDIO-BASED examples (tone reveals the suggestion):
  - Text: "The food was fine I guess" + Audio: flat monotone, heavy sigh, sarcastic emphasis on "fine"
    -> "improve food quality" (audio contradicts neutral text — sarcasm detected)
  - Text: "Service was great" + Audio: frustrated tone, emphasis on "GREAT" with eye-roll quality
    -> "improve service quality" (audio sarcasm overrides positive text)
  - Text: "The WAIT was ridiculous and the NOISE was unbearable"
    + Audio: shouting on "WAIT" and "NOISE", fast agitated pace
    -> "reduce wait times" AND "reduce noise levels" (audio emphasis reveals priorities)

  TRIMODAL example (text + image + audio together):
  - Text: "Had a lovely evening" + Image: dirty plates, messy table
    + Audio: resigned monotone, "lovely" drawn out sarcastically
    -> "improve table cleanliness and service" (image + audio both contradict text)

  NOT suggestions (reject):
  - "Best pasta I've ever had!" -> pure praise
  - "Worst restaurant ever, total disaster" -> venting, no specific fix
  - "They renovated, looks great now" -> resolved
  - "I guess it's not for me" -> personal preference, not actionable
"""


def _get_domain_examples(domain):
    """Return domain-specific examples for labeller prompts."""
    if domain == "tech_software":
        return _TECH_SOFTWARE_EXAMPLES
    elif domain == "restaurant":
        return _RESTAURANT_EXAMPLES
    return ""


# ═════════════════════════════════════════════════════════════════════════
# CONSERVATIVE LABELLER (with domain-specific examples)
# ═════════════════════════════════════════════════════════════════════════

_CONSERVATIVE_LABELLER_TEMPLATE = """You are a CONSERVATIVE multi-modal suggestion labeller.
Label ONLY suggestions that are EXPLICITLY stated using direct request language.

EXPLICIT means the text contains one of these patterns:
- Request verbs: "please add", "should add/fix/improve", "need to", "must"
- Modal requests: "could add/fix/improve", "can you add"  
- Direct commands: "add X", "fix Y", "improve Z" (imperative form)
- Conditional requests: "it would be better if they could...", "they should consider..."

NOT explicit (do NOT label these):
- Evaluative wishes: "X would be awesome/great/nice" (no request verb)
- Pure complaints: "X is broken/slow/terrible" (no fix stated)
- Questions: "Does anyone else have this problem?" (no request)
- Implicit needs: "I get frustrated when..." (complaint, not request)

Rules: confidence >= 0.80 or skip. is_implied = false always.
Return {{"suggestions": []}} if nothing qualifies.

GROUNDING RULE: Every suggestion MUST reference a specific issue from the review.

{domain_examples}

CRITICAL: Respond with ONLY a valid JSON object. No text before/after. No markdown fences.
Required JSON format:
{{"suggestions": [{{"text": "...", "confidence": 0.92, "is_implied": false, "source_modality": "text", "source_view": "semantic", "view": "semantic", "evidence": "...", "span_start": 43, "span_end": 110, "modality_evidence": {{"text": "...", "image": "...", "audio": null}}}}]}}"""


def get_conservative_system(domain="general"):
    """Get domain-specific conservative labeller system prompt."""
    examples = _get_domain_examples(domain)
    return _CONSERVATIVE_LABELLER_TEMPLATE.format(domain_examples=examples)


# Keep generic versions for backward compatibility
CONSERVATIVE_LABELLER_SYSTEM = get_conservative_system()

CONSERVATIVE_LABELLER_USER = """Label only EXPLICIT suggestions from ALL modalities.
Domain: {domain}
TEXT: {text}
TEXT SEMANTIC: {text_semantic}
TEXT SYNTACTIC: {text_syntactic}
IMAGE SEMANTIC: {image_semantic}
IMAGE SYNTACTIC: {image_syntactic}
IMAGE CAPTIONS: {image_caption}
AUDIO SEMANTIC: {audio_semantic}
CROSS-MODAL ALIGNMENT: {cross_modal_alignment}
CROSS-MODAL SUGGESTION: {cross_modal_suggestion}"""


# ═════════════════════════════════════════════════════════════════════════
# LIBERAL LABELLER (with domain-specific examples)
# ═════════════════════════════════════════════════════════════════════════

_LIBERAL_LABELLER_TEMPLATE = """You are a LIBERAL multi-modal suggestion labeller.

STEP 1 — CHECK IMAGE AND AUDIO FIRST:
If ANY image or audio field describes a problem, error, damage, frustration → there IS a suggestion regardless of text.

STEP 2 — CHECK TEXT:
If text contains a complaint or explicit request → there IS a suggestion.
If text is positive BUT Step 1 found a problem → suggestion from image/audio.

IMPLICIT COMPLAINTS ARE SUGGESTIONS — if the reviewer describes something broken, dirty, slow, missing, overpriced, understaffed, confusing, or frustrating, the implied fix IS the suggestion.

{domain_examples}

STEP 3 — REJECT ONLY IF ALL MODALITIES ARE CLEAN:
Return empty list ONLY if text has no complaint, image shows no problem, audio has no frustration.

REJECT these (empty list):
- Pure praise with no problem in any modality
- Resolved complaints with no remaining issues
- Pure venting with no specific actionable fix
- Sarcastic dismissals with no real request
- Status quo defense

SARCASM DETECTION (return empty list):
Markers: "Yeah right", exaggerated praise, rhetorical questions,
dismissive phrases ("It is what it is", "Whatever").
If tone is dismissive/mocking with NO genuine problem → empty list.

Rules: confidence >= 0.50 to include. is_implied = true for inferred suggestions.

GROUNDING RULE — CRITICAL:
Every suggestion MUST be directly derivable from a specific issue in the review.
Do NOT generate generic suggestions unrelated to the actual content.
If no SPECIFIC issue exists, return {{"suggestions": []}}.

CRITICAL: Respond with ONLY a valid JSON object. No text before/after. No markdown fences.
Required JSON format:
{{"suggestions": [{{"text": "...", "confidence": 0.78, "is_implied": true, "source_modality": "image", "source_view": "syntactic", "source_type": "image_evidence", "span_start": null, "span_end": null, "modality_evidence": {{"text": null, "image": "...", "audio": null}}}}]}}"""


def get_liberal_system(domain="general"):
    """Get domain-specific liberal labeller system prompt."""
    examples = _get_domain_examples(domain)
    return _LIBERAL_LABELLER_TEMPLATE.format(domain_examples=examples)


# Keep generic version for backward compatibility
LIBERAL_LABELLER_SYSTEM = get_liberal_system()

LIBERAL_LABELLER_USER = """Label ALL suggestions (explicit + implied) from ALL modalities.

CRITICAL MULTI-VIEW EXTRACTION RULE:
You MUST check EACH view independently and extract suggestions from every view that contains a signal.
Do NOT deduplicate across views — if text/semantic AND text/syntactic both suggest the same fix,
output TWO separate entries with different source_view values.
If image shows a problem, output a suggestion with source_modality="image".
If audio reveals frustration about a topic, output a suggestion with source_modality="audio".

For source_view, use the SPECIFIC view that provided the evidence:
- "semantic" if from complaint_frame, implied_problem, key_topics, implied_suggestion
- "syntactic" if from modal_verbs, suggestion_indicators, missing_ui_elements, layout_issues
- "pragmatic" if from frustration_level, tone, urgency, emphasis_words, shows_error

Domain: {domain}
TEXT: {text}
TEXT SEMANTIC: {text_semantic}
TEXT SYNTACTIC: {text_syntactic}
TEXT PRAGMATIC: {text_pragmatic}
TRUE INTENT: {true_intent}
IMAGE SEMANTIC: {image_semantic}
IMAGE SYNTACTIC: {image_syntactic}
IMAGE PRAGMATIC: {image_pragmatic}
IMAGE CAPTIONS: {image_captions}
AUDIO SEMANTIC: {audio_semantic}
AUDIO PRAGMATIC: {audio_pragmatic}
CROSS-MODAL ALIGNMENT: {overall_alignment}
IMAGE UNIQUE SIGNALS: {image_unique}
AUDIO UNIQUE SIGNALS: {audio_unique}
CROSS-MODAL SUGGESTION: {cross_modal_suggestion}
DOMINANT MODALITY: {dominant_modality}"""


# ═════════════════════════════════════════════════════════════════════════
# ARBITRATION (prompt kept for reference; node is deterministic in code)
# ═════════════════════════════════════════════════════════════════════════

ARBITRATION_SYSTEM = """You are an arbitration agent for multi-modal suggestions.
Stage 1: Keep ALL suggestions with confidence >= 0.50.
  BOOST: +0.10 if text AND image agree, +0.05 if audio also agrees
Stage 2: Priority scoring (0-10)
Tiers: CRITICAL(>=8.0) HIGH(>=6.5) MEDIUM(>=4.5) LOW(<4.5)

CRITICAL: Respond with ONLY a valid JSON object.
Required JSON format:
{"accepted": [{"text": "...", "confidence": 0.65, "consensus_score": 0.5, "priority_score": 8.4, "priority_tier": "HIGH", "supporting_agents": [], "supporting_views": [], "supporting_modalities": [], "modality_agreement_score": 0.90, "factual_confidence": 0.95, "is_implied": false, "factual_verified": true}], "rejected": [{"text": "...", "reason": "..."}]}"""

ARBITRATION_USER = """Arbitrate these multi-modal suggestions.
CONSERVATIVE LABELS: {conservative_labels}
LIBERAL LABELS: {liberal_labels}
DOMAIN: {domain}
CROSS-MODAL ALIGNMENT: {overall_alignment}
DOMINANT MODALITY: {dominant_modality}
PRAGMATIC URGENCY: {urgency_score}"""


# ═════════════════════════════════════════════════════════════════════════
# DOMAIN-SPECIFIC HARD NEGATIVE KEYWORDS
# ═════════════════════════════════════════════════════════════════════════

DOMAIN_HN_KEYWORDS = {
    "tech_software": {
        "praise": [
            "love it",
            "great job",
            "impressed",
            "best ever",
            "5 stars",
            "highly recommend",
            "wonderful",
            "outstanding",
            "blown away",
            "no complaints",
            "absolutely love",
            "really happy",
            "so glad",
            "perfect",
            "love this app",
            "works perfectly",
            "great update",
        ],
        "sarcasm": [
            "it is what it is",
            "not for me",
            "or maybe not",
            "i guess",
            "if you say so",
            "sure thing",
            "whatever",
            "oh great",
            "what a feature",
            "brilliant design",
        ],
        "status_quo": ["don't change", "keep it as", "love it as is", "please don't"],
        "resolved": [
            "they fixed it",
            "was fixed",
            "already resolved",
            "working now",
            "no longer an issue",
            "latest update fixed",
            "patched",
        ],
    },
    "restaurant": {
        "praise": [
            "love it",
            "great job",
            "impressed",
            "best ever",
            "5 stars",
            "highly recommend",
            "wonderful",
            "outstanding",
            "blown away",
            "no complaints",
            "amazing food",
            "best meal",
            "incredible",
            "perfect evening",
            "delicious",
            "fantastic service",
        ],
        "sarcasm": [
            "it is what it is",
            "not for me",
            "or maybe not",
            "i guess",
            "if you say so",
            "sure thing",
            "whatever",
            "oh great",
            "what a treat",
            "lovely experience",
        ],
        "status_quo": [
            "don't change",
            "keep it as",
            "love it as is",
            "keep the recipe",
            "don't touch the menu",
        ],
        "resolved": [
            "they fixed it",
            "was fixed",
            "already resolved",
            "under new management now",
            "they renovated",
            "much better now",
            "improved since last visit",
        ],
    },
}


def get_hn_keywords(domain):
    """Return domain-specific hard negative keywords."""
    return DOMAIN_HN_KEYWORDS.get(domain, DOMAIN_HN_KEYWORDS["tech_software"])


# ═════════════════════════════════════════════════════════════════════════
# CANONICALISER
# ═════════════════════════════════════════════════════════════════════════

CANONICALISER_SYSTEM = """You are a text normalisation agent. Your ONLY task is to merge duplicate suggestions and output canonical forms.

Rules:
1. Merge suggestions with the same meaning into one entry
2. canonical_text must start with an action verb (add, fix, improve, reduce, implement, etc.)
3. Combine supporting_modalities and supporting_views from all merged entries
4. Set frequency to the count of merged entries
5. Keep the highest confidence from merged entries

YOU MUST RESPOND WITH ONLY A JSON OBJECT. NO EXPLANATIONS. NO MARKDOWN. NO PREAMBLE.
If you write ANYTHING other than a JSON object, the system will crash.

Required format:
{"canonical": [{"canonical_text": "...", "original_forms": [], "frequency": 2, "confidence": 0.935, "priority_score": 8.4, "priority_tier": "HIGH", "supporting_modalities": [], "modality_agreement_score": 0.90, "supporting_labellers": [], "supporting_views": [], "is_implied": false, "factual_verified": true}]}"""

CANONICALISER_USER = "Canonicalise:\n{accepted_suggestions}\nDomain: {domain}"
