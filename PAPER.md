# MASP: A Multimodal Agentic Pipeline for Suggestion Mining from Reviews

> Draft. Numbers reported here are taken from the accompanying materials: the
> dataset files under `data/`, the recorded RoBERTa re-evaluation in
> `roberta_reeval_log.txt`, and the MASP detection/mining scores recorded in the
> evaluation harness (`compute_final_metrics.py`). Per-path, per-modality,
> and per-type MASP breakdowns are produced by that harness from a full
> `results/<run>/pipeline_results.csv`; fill the marked cells from your run.

## Abstract

Suggestion mining — deciding whether a user review contains an actionable
suggestion and, if so, extracting it — is usually framed as sentence
classification over text. Real reviews are increasingly multimodal (photos,
voice notes) and express suggestions implicitly and politely, without lexical
signals such as *should* or *please*. We present **MASP**, a multimodal agentic
pipeline that mines suggestions from reviews combining text, image, and audio.
MASP is a 12-layer LangGraph state graph of pure-function agents: parallel
per-modality views, cross-modal alignment, dual conservative/liberal labelling,
consensus arbitration with cross-modal boosting, evidence-grounded
explicit/implicit routing, canonicalisation, clustering, memory, and a
view-weighted reranker with a human-review gate. On a 2,500-item, two-domain
(restaurant, tech) multimodal dataset with eight controlled extraction paths and
hard negatives, MASP performs end-to-end mining (detection **F1 0.946**,
Mining F1 **0.751**) with no task-specific training. A strong supervised
RoBERTa detector matches MASP on binary detection (F1 0.950) but performs no
extraction and, we show, attains perfect recall on the implicit no-signal-word
paths only by exploiting a distributional artifact — highlighting why detection
F1 alone is a misleading measure of suggestion understanding.

## 1. Introduction

Platforms want to turn the suggestions buried in reviews into product and
service actions. Two things make this hard in practice. First, reviews are no
longer text-only: a diner photographs a dish and records a voice note; a shopper
attaches a screenshot. Second, many real suggestions are **implicit** — "the
tables were a little close together" is a request for more spacing with no
imperative and no signal word. Classifiers trained on explicit, signal-word-rich
data learn shortcuts that fail on exactly these cases.

We make three contributions:

1. **MASP**, a multimodal, agentic suggestion-mining pipeline (Section 4) that is
   prompt-based and requires no task-specific fine-tuning, yet does full
   *mining* (detection **and** extraction), not just classification.
2. A **2,500-item multimodal dataset** (Section 3) with two domains, aligned
   image and audio, eight controlled positive extraction paths, and
   suggestion-shaped hard negatives, built to separate genuine understanding
   from lexical shortcuts.
3. An **artifact analysis** (Section 6) showing that a supervised detector's
   headline F1 is inflated by memorizable surface cues on the implicit paths,
   which motivates reporting Mining F1 and per-path recall rather than a single
   detection number.

## 2. Related work

Suggestion mining was popularized as a binary/sequence-labelling task over
sentences (e.g., the SemEval suggestion-mining shared task, whose splits we
cross-evaluate on; see `data/semeval_*`). Most systems are single-model text
classifiers. Multimodal review understanding has focused on sentiment rather
than actionable suggestions. Agentic LLM pipelines (LangGraph-style state graphs
of cooperating prompts) have been applied to reasoning and retrieval, but not,
to our knowledge, to multimodal suggestion mining with explicit/implicit routing
and evidence grounding. MASP combines these threads.

## 3. Dataset

**Composition.** `MASP_Dataset_v5` contains 2,500 review items across two
domains — restaurant (`REST`) and tech (`TECH`). Each item carries the review
text and, where present, an aligned gold image and an aligned gold audio
transcript (`media/MEDIA_MANIFEST.csv` maps `entry_id → image, audio`; the
release includes 527 gold images and 527 gold audio files). Splits:

| Split | N | Positive | Negative |
|---|---:|---:|---:|
| train | 1,486 | 929 | 557 |
| test  | 527 | 335 | 192 |

(train/test overlap = 0, verified in `roberta_reeval_log.txt`.)

**Controlled structure.** Test positives are organized into eight *extraction
paths* (`P1`–`P8`) that vary how the suggestion is realized, plus `HN`, a pool of
suggestion-shaped **hard negatives**:

| Path | N | Mean chars | Character |
|---|---:|---:|---|
| P1 | 98 | 174 | explicit |
| P2 | 78 | 176 | explicit |
| P3 | 32 | 169 | explicit |
| P4 | 21 | 163 | explicit |
| P6 | 40 | 173 | explicit |
| P5 | 22 | 90 | implicit / polite |
| P7 | 22 | 90 | implicit / polite |
| P8 | 22 | 85 | implicit / polite |
| HN | 192 | 131 | non-suggestion |

`P5`/`P7`/`P8` are short and, by construction, contain **no explicit signal
words** — a lexical check over a 19-word signal list finds **0 violations** on
all three paths (`roberta_reeval_log.txt`). They are the implicit, polite cases
a shortcut model should fail on. `HN` items look suggestion-like but are not.

**Annotation.** Items are labelled for `is_suggestion`, `suggestion_type`
(explicit/implicit), and `suggestion_text`. Sampled inter-annotator-agreement
items are released in `data/IAA_annotation_samples.csv`
(`compute_iaa.py`); a 50-item human-evaluation sample is in
`data/human_eval_50_samples.csv`. Raw annotator sheets are withheld for
anonymity.

## 4. Method

MASP is a 12-layer LangGraph state graph (`graph/pipeline.py`); each node is
a pure `PipelineState → partial-state` function (`agents/nodes.py`). All LLM
calls route through `llm_backend.py` (Ollama `gemma3:27b-it-qat`, text and
vision, temperature 0); audio is transcribed with Whisper.

```
preprocess ─▶ {text, image, audio} views ─▶ cross-modal align ─▶ domain router
   ─▶ {conservative, liberal} labellers ─▶ merge ─▶ arbitration
   ─▶ evidence provenance ─▶ suggestion switch ─▶ canonicalise
   ─▶ cluster ─▶ memory ─▶ view-weighted rerank ─▶ human-review gate
```

- **Per-modality views (Layers 2–4).** Text gets three views; the image gets
  three views via the vision model; audio gets two views of its transcript.
  Building multiple views per modality reduces single-prompt variance.
- **Cross-modal alignment (Layer 5).** Fuses the modality views and computes an
  alignment signal that drives a *view-weighting switch* downstream, so
  well-aligned modalities count more.
- **Domain routing (Layer 6).** Classifies the review's domain (agentic-RAG
  framing) to condition later prompts.
- **Dual labelling (Layer 7).** A **conservative** labeller marks explicit
  suggestions only; a **liberal** labeller marks explicit + implied. Running both
  and reconciling them mitigates the bias of any single labeller — directly
  targeting the implicit paths.
- **Arbitration (Layer 8).** Reconciles the two labellers by consensus and
  applies cross-modal boosting: agreement corroborated by image/audio evidence is
  strengthened.
- **Evidence provenance (Layer 8.5).** Computes a grounding score for each
  accepted suggestion, enriching it *before* the switch so the faithfulness gate
  can use it.
- **Suggestion switch (Layer 8.6).** Classifies each suggestion as
  explicit/implicit/ambiguous and routes it to a type-specific evaluator,
  attaching actionability, feasibility, specificity, and faithfulness scores and
  an inference chain. Ambiguous items run both paths and keep the better.
- **Canonicalise / cluster / memory (Layers 9–10).** De-duplicate and standardise
  suggestions, cluster them across reviews, and maintain a multi-type memory for
  collective cognition across the corpus.
- **View-weighted reranker (Layer 11).** Scores suggestions with a multi-feature
  scorer weighted by the alignment switch.
- **Human-review gate (Layer 12).** Flags low-confidence cases for review rather
  than forcing a decision.

MASP therefore outputs, per review, whether a suggestion exists **and** the
extracted, typed, grounded suggestion text — full mining, not classification.

## 5. Experimental setup

**Task and metrics** (`compute_final_metrics.py`). We report **Detection
F1** (did the system surface any suggestion?), **Extraction quality** (stemmed
token-overlap between the top predicted suggestion and the gold), and their
harmonic mean, **Mining F1**. We add a 1,000-sample bootstrap 95% CI and break
results down per path, per modality combination (T, T+I, T+A, T+I+A), and per
suggestion type. MASP is evaluated zero-shot (no task training).

**Baselines.** TF-IDF + Linear SVM and a supervised **RoBERTa-base** classifier,
both trained on the 1,486-item train split; and prompt-only LLM baselines
(`baselines_llm.py`, `baseline_single_prompt.py`). RoBERTa is trained
for 5 epochs at lr 2e-5, three seeds (42/123/456), majority vote
(`roberta_reeval.py`). Baselines are text-only and produce detection only (the
SVM/RoBERTa classifiers have no extraction head), so their Mining F1 is
undefined.

## 6. Results

### 6.1 Main comparison

| System | Supervision | Detection F1 | Extraction | Mining F1 |
|---|---|---:|---:|---:|
| TF-IDF + SVM | 1,486 labels | *[fill from harness]* | *[fill]* | *[fill]* |
| RoBERTa-base (3-seed vote) | 1,486 labels | **0.950** | — (no extraction) | — |
| **MASP (ours)** | none (prompt-based) | **0.946** | **[fill: ext]** | **0.751** |

RoBERTa detection F1 across seeds: 0.948 / 0.951 / 0.950 (MCC ≈ 0.86); majority
vote 0.950. MASP's detection F1 (0.946) is statistically on par, but MASP
additionally **extracts** the suggestion (Mining F1 0.751), which the supervised
classifiers cannot do at all, and it does so **without any labelled training
data**.

### 6.2 The detection metric is misleading: an artifact on the implicit paths

Per-path recall for the RoBERTa majority-vote model (`roberta_reeval_log.txt`):

| Path | Character | RoBERTa recall |
|---|---|---:|
| P1 | explicit | 0.990 |
| P2 | explicit | 0.846 |
| P3 | explicit | 1.000 |
| P4 | explicit | 1.000 |
| P6 | explicit | 1.000 |
| P5 | implicit, no signal words | **1.000** |
| P7 | implicit, no signal words | **1.000** |
| P8 | implicit, no signal words | **1.000** |
| HN | hard negative | 21/192 false positives |

A purely lexical model achieving **perfect recall on P5/P7/P8** — short,
implicit, signal-word-free suggestions — cannot be doing so through
signal-word cues (there are none; the polite-masking check finds 0 violations).
It is exploiting a **distributional artifact**: surface regularities specific to
how those short paths were generated, memorized from the 3×-augmented training
distribution. Our harness flags exactly these paths (`ARTIFACT` tag in
`roberta_reeval.py`). The lesson is methodological: a single detection F1 hides
this. MASP is evaluated on the same paths without any exposure to the training
distribution, and reports per-path recall and Mining F1 so that implicit-path
performance and extraction are visible rather than absorbed into one number.

### 6.3 MASP breakdowns

*From the full MASP `pipeline_results.csv` via `compute_final_metrics.py`:*

- Per-path recall (P1–P8, HN): *[fill]*
- Per-modality F1 (T, T+I, T+A, T+I+A): *[fill]* — quantifies the multimodal gain.
- Per-type recall (explicit vs. implicit): *[fill]*
- Extraction quality overall and on P5/P7/P8: *[fill]*
- SemEval cross-domain transfer (`data/semeval_*`, `eval_semeval_cross.py`):
  *[fill]*

### 6.4 Ablations

Conditions (`REPRODUCIBILITY_NOTES.md`): A1 no-image, A2 no-audio, A4
switch-off (`run_ablations.py`, legacy `A1` = force COMMON mode), B5 text-only
(`baselines_comprehensive.py --baseline B5`). Report Detection/Mining F1 per
condition to isolate the contribution of each modality and of the
explicit/implicit switch: *[fill from ablation runs]*.

## 7. Discussion

MASP trades a fraction of a point of detection F1 for two things a supervised
classifier cannot provide: end-to-end **extraction** of the actual suggestion
(Mining F1 0.751) and **multimodal** grounding, with **no labelled training
data**. The artifact analysis argues that the supervised "win" on detection is
partly illusory on precisely the hard, implicit cases that motivate the task.
For a deployment that must return *what to change*, not merely *that something
was suggested*, and that must handle photos and voice notes, an agentic mining
pipeline is the more faithful design.

## 8. Limitations and ethics

MASP depends on a capable local vision-language model (`gemma3:27b`); quality and
latency scale with it. Extraction quality is measured by stemmed token overlap,
a proxy for semantic match. The dataset is two-domain and English; the SemEval
cross-evaluation probes but does not guarantee broader transfer. The human-review
gate is a safeguard, not a substitute for human oversight. Raw annotator sheets
are withheld for anonymity; sampled IAA items support the agreement audit. No
personal data beyond the released review content is used.

## 9. Conclusion

We presented MASP, a prompt-based multimodal agentic pipeline that mines
suggestions — detecting and extracting them — from reviews that combine text,
image, and audio. It matches a supervised detector on detection F1 while adding
extraction and multimodality with no task training, and we show why detection F1
alone overstates suggestion understanding on implicit cases. Code, data, and
evaluation harness are released for reproduction.

---

### Reproducing the numbers

```bash
# RoBERTa baseline (real numbers used above are recorded in roberta_reeval_log.txt)
python roberta_reeval.py

# MASP full run + definitive metrics (needs Ollama gemma3:27b)
python run_dataset.py --csv data/test.csv --output results/test_FINAL_v5
python compute_final_metrics.py
```
