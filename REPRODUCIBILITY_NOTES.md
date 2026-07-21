# MASP Reproducibility Notes

This supplementary package contains the source code, prompts, datasets, and
gold media used for the MASP EMNLP 2026 submission.

## Core Pipeline Modules

The runnable pipeline depends on the following code directories:

- `code/graph/`
- `code/agents/`
- `code/memory/`
- `code/prompts/`

These directories are included in this package. The main entry point is
`code/run_dataset.py`, which calls `code/main.py`.

## Paper Table 5 Ablation Mapping

The paper reports four ablation conditions:

- A1 no-image input: run the pipeline on `data/test_A1_no_image.csv`.
- A2 no-audio input: run the pipeline on `data/test_A2_no_audio.csv`.
- B5 text-only pipeline: run `code/baselines_comprehensive.py --baseline B5`.
- A4 switch-off: run `code/run_ablations.py --ablation A1`.

The last command uses a legacy code-level name. In `run_ablations.py`,
legacy `A1` means "force COMMON mode", which corresponds to the paper's
A4 switch-off condition.

Optional extension: `code/run_b5_extraction.py` reruns B5 with extraction
logging and reports Detection F1, extraction overlap, and Mining F1. This
was not part of the original paper tables, but it is included so future
users can fill the B5 extraction gap directly.

## Media

The `media/` folder contains the 527 image files and 527 audio files used
for the real-input evaluation. `media/MEDIA_MANIFEST.csv` maps each test
`entry_id` to its corresponding image and audio file.
