#!/usr/bin/env python3
"""Train and evaluate a RoBERTa suggestion-detection baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from datasets import Dataset

TEXT_COLUMNS = ("raw_text", "text", "review_text")


def parse_label(value: object) -> int:
    """Convert the dataset's common boolean encodings to 0 or 1."""
    return int(str(value).strip().lower() in {"true", "1", "1.0", "yes"})


def select_text_column(frame: pd.DataFrame) -> str:
    for column in TEXT_COLUMNS:
        if column in frame.columns:
            return column
    raise ValueError(f"Expected one of these text columns: {', '.join(TEXT_COLUMNS)}")


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    import pandas as pd

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path.as_posix()} and {test_path.as_posix()}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    text_column = select_text_column(train)
    if text_column not in test.columns:
        raise ValueError(f"Test split does not contain the '{text_column}' column")

    for frame in (train, test):
        frame["label"] = frame["is_suggestion"].map(parse_label)
    return train, test, text_column


def binary_metrics(gold: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold,
        predicted,
        average="binary",
        zero_division=0,
    )
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "mcc": round(float(matthews_corrcoef(gold, predicted)), 4),
        "tp": int(((predicted == 1) & (gold == 1)).sum()),
        "fp": int(((predicted == 1) & (gold == 0)).sum()),
        "fn": int(((predicted == 0) & (gold == 1)).sum()),
        "tn": int(((predicted == 0) & (gold == 0)).sum()),
    }


def per_path_metrics(
    test: pd.DataFrame, predicted: np.ndarray
) -> dict[str, dict[str, float | int]]:
    if "extraction_path" not in test.columns:
        return {}

    scored = test.copy()
    scored["predicted"] = predicted
    results: dict[str, dict[str, float | int]] = {}
    for path, group in scored.groupby("extraction_path"):
        results[str(path)] = binary_metrics(
            group["label"].to_numpy(), group["predicted"].to_numpy()
        )
        results[str(path)]["count"] = len(group)
    return results


def make_dataset(
    frame: pd.DataFrame,
    text_column: str,
    tokenizer: Any,
    max_length: int,
) -> Dataset:
    from datasets import Dataset

    dataset = Dataset.from_dict(
        {
            "text": frame[text_column].fillna("").astype(str).tolist(),
            "label": frame["label"].tolist(),
        }
    )

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return dataset.map(tokenize, batched=True)


def train_seed(
    *,
    seed: int,
    model_name: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    output_dir: Path,
    epochs: float,
) -> np.ndarray:
    import numpy as np
    from transformers import (
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir / f"seed-{seed}"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=seed,
        logging_steps=100,
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    prediction = trainer.predict(test_dataset)
    return np.argmax(prediction.predictions, axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/roberta"))
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    from transformers import AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train, test, text_column = load_splits(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_dataset = make_dataset(train, text_column, tokenizer, args.max_length)
    test_dataset = make_dataset(test, text_column, tokenizer, args.max_length)
    gold = test["label"].to_numpy()

    seed_predictions: list[np.ndarray] = []
    seed_results: dict[str, object] = {}
    report: dict[str, object] = {
        "model": args.model,
        "train_count": len(train),
        "test_count": len(test),
        "seeds": seed_results,
    }

    for seed in args.seeds:
        predicted = train_seed(
            seed=seed,
            model_name=args.model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
        )
        seed_predictions.append(predicted)
        seed_results[str(seed)] = {
            "overall": binary_metrics(gold, predicted),
            "per_path": per_path_metrics(test, predicted),
        }

    stacked = np.stack(seed_predictions)
    majority = (stacked.sum(axis=0) >= (len(seed_predictions) // 2 + 1)).astype(int)
    report["majority_vote"] = {
        "overall": binary_metrics(gold, majority),
        "per_path": per_path_metrics(test, majority),
    }

    output_path = args.output_dir / "metrics.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["majority_vote"], indent=2))
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
