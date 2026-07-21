# MASP: Multimodal Agentic Suggestion Pipeline

MASP mines actionable suggestions from user reviews that combine **text, image,
and audio**. It is a multi-agent pipeline built on LangGraph: each review is
processed through parallel per-modality views, cross-modal alignment, dual
conservative/liberal labelling, arbitration, evidence grounding, an
explicit/implicit suggestion switch, canonicalisation, clustering, memory, and a
view-weighted reranker, with a final human-review gate for uncertain cases.

This is the supplementary package for the EMNLP submission: source code,
datasets, gold media, and evaluation scripts.

## Contents

| Path | Description |
|---|---|
| `code/` | Pipeline (`graph/`, `agents/`, `memory/`, `prompts/`, `media/`), baselines, ablations, and evaluation scripts |
| `data/` | Dataset splits, SemEval cross-eval sets, IAA and human-eval samples |
| `media/` | Gold images and audio + `MEDIA_MANIFEST.csv` |
| `roberta_reeval.py`, `roberta_reeval_log.txt` | Supervised RoBERTa baseline re-evaluation and its recorded log |
| `PAPER.md` | Paper draft |
| `REPRODUCIBILITY_NOTES.md` | Notes for reproducing the reported results |

## Method

The pipeline is a 12-layer LangGraph state graph (`code/graph/pipeline.py`):

```
preprocess → {text, image, audio} views → cross-modal align → domain router
→ {conservative, liberal} labellers → arbitration → evidence provenance
→ suggestion switch (explicit/implicit) → canonicalise → cluster → memory
→ view-weighted rerank → human-review gate
```

Every agent is a pure `PipelineState -> partial-state` function
(`code/agents/nodes.py`). Dual labelling (conservative = explicit only, liberal
= explicit + implied) mitigates single-model bias; arbitration applies
cross-modal boosting; the suggestion switch routes explicit vs. implicit
suggestions to type-specific evaluators.

## Backend

All LLM calls route through `code/llm_backend.py`. Default local configuration:

- Provider: Ollama HTTP API
- Text + vision model: `gemma3:27b-it-qat`
- Temperature: `0.0`
- Audio transcription: OpenAI Whisper

```bash
ollama serve
ollama pull gemma3:27b-it-qat
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# smoke checks
python code/llm_backend.py
python code/run_dataset.py --help

# small pipeline run (needs Ollama running)
python code/run_dataset.py --csv data/test.csv --output results_safe --max-samples 5

# definitive metrics (needs a full results/<run>/pipeline_results.csv)
python code/compute_final_metrics.py
```

## Evaluation

`code/compute_final_metrics.py` reports **Detection F1** (was any suggestion
found?), **Extraction quality** (stemmed token overlap of the top suggestion
with the gold), and their harmonic mean, **Mining F1**, with a 1000-sample
bootstrap 95% CI. It breaks results down per extraction path (P1–P8, HN), per
modality combination (T, T+I, T+A, T+I+A), and per suggestion type (explicit,
implicit). Baselines (TF-IDF+SVM, RoBERTa, LLM single-prompt) are evaluated on
the same splits.

## Data notes

Two review domains (restaurant `REST`, tech `TECH`). The test set has 527 items
(335 positive, 192 hard negatives). Positives span eight extraction paths;
`P5`/`P7`/`P8` are short, implicit/polite suggestions with no explicit signal
words, and `HN` are suggestion-shaped non-suggestions — both designed to defeat
lexical shortcuts. Raw annotator sheets are withheld for anonymity; sampled IAA
items are included for the audit.
