# MASP Reproducibility Notes

This repository contains the source code, prompts, dataset splits, and media
manifest used by the MASP research prototype.

## Core Pipeline Modules

The runnable pipeline depends on the following code directories:

- `graph/`
- `agents/`
- `memory/`
- `prompts/`

The main entry point is `run_dataset.py`, which calls `main.py`.

## Paper Table 5 Ablation Mapping

The paper reports four ablation conditions:

- A1 no-image input: run the pipeline on `data/test_A1_no_image.csv`.
- A2 no-audio input: run the pipeline on `data/test_A2_no_audio.csv`.
- B5 text-only pipeline: run `baselines_comprehensive.py --baseline B5`.
- A4 switch-off: run `run_ablations.py --ablation A1`.

The last command uses a legacy code-level name. In `run_ablations.py`,
legacy `A1` means "force COMMON mode", which corresponds to the paper's
A4 switch-off condition.

Optional extension: `run_b5_extraction.py` reruns B5 with extraction
logging and reports Detection F1, extraction overlap, and Mining F1. This
was not part of the original paper tables, but it is included so future
users can fill the B5 extraction gap directly.

## Media

The `media/` folder contains the 527 image files and 527 audio files used
for the real-input evaluation. `media/MEDIA_MANIFEST.csv` maps each test
`entry_id` to its corresponding image and audio file.
