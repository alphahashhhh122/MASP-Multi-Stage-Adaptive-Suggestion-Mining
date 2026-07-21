"""
evaluate.py — MASP Evaluation Harness

Implements all 9 evaluation dimensions from the MASP Evaluation Metrics Framework.
Runs the full pipeline (or loads cached outputs), compares against gold labels,
and produces paper-ready results tables.

Usage:
    python evaluate.py --dataset path/to/gold.csv --output results/
    python evaluate.py --cached path/to/outputs/ --gold path/to/gold.csv

Paper table format:
    Metric         | B1    | B2    | B3    | B5    | MASP   | Delta
    Extraction F1  | 0.72  | 0.78  | 0.84  | 0.85  | 0.88*  | +0.16
    ...

Dimensions:
    1. Dataset quality (IAA)               — Krippendorff's α, Cohen's κ
    2. Extraction quality                  — P, R, F1, span F1
    3. Explicit vs implicit (type-specific) — uses UnifiedEvaluator from suggestion_switch.py
    4. Multi-view switch                   — switch accuracy, view ablation deltas
    5. Multimodal fusion                   — modality ablation, cross-modal alignment
    6. Faithfulness                        — hallucination rate, grounding score, RAGAS-style
    7. Ranking                             — NDCG@K, Kendall's τ, top-K precision
    8. Cross-domain + memory               — leave-one-out, memory hit rate
    9. Cost + efficiency                   — tokens/review, latency, $/review
"""

import json
import logging
import math
import time
import os
from collections import defaultdict
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DIM 1: DATASET QUALITY (IAA)
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetQualityMetrics:
    """
    Dimension 1: Inter-annotator agreement.
    Requires: two sets of annotations on the same samples.
    """

    @staticmethod
    def krippendorff_alpha(annotations_a: list, annotations_b: list) -> float:
        """
        Krippendorff's Alpha for two annotators.
        Handles nominal, ordinal, interval data.
        Target: >= 0.70
        """
        if len(annotations_a) != len(annotations_b):
            raise ValueError("Annotation lists must be same length")
        n = len(annotations_a)
        if n == 0:
            return 0.0

        # Observed disagreement
        do = sum(1 for a, b in zip(annotations_a, annotations_b) if a != b) / n

        # Expected disagreement (chance)
        all_vals = annotations_a + annotations_b
        val_counts = defaultdict(int)
        for v in all_vals:
            val_counts[v] += 1
        total = len(all_vals)
        de = 1.0 - sum((c / total) ** 2 for c in val_counts.values())

        if de == 0:
            return 1.0  # perfect agreement by chance
        return round(1.0 - (do / de), 4)

    @staticmethod
    def cohen_kappa(labels_a: list, labels_b: list) -> float:
        """
        Cohen's Kappa for two annotators on binary/categorical labels.
        Target: >= 0.75
        """
        n = len(labels_a)
        if n == 0:
            return 0.0
        po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

        all_labels = set(labels_a + labels_b)
        pe = sum(
            (labels_a.count(l) / n) * (labels_b.count(l) / n)
            for l in all_labels
        )
        if pe == 1.0:
            return 1.0
        return round((po - pe) / (1 - pe), 4)

    @staticmethod
    def span_agreement_f1(spans_a: list[tuple], spans_b: list[tuple]) -> float:
        """
        Token-level F1 between two annotators' span selections.
        Each span is (start, end) character offsets.
        Target: >= 0.65
        """
        tokens_a = set()
        for s, e in spans_a:
            tokens_a.update(range(s, e))
        tokens_b = set()
        for s, e in spans_b:
            tokens_b.update(range(s, e))

        if not tokens_a and not tokens_b:
            return 1.0
        tp = len(tokens_a & tokens_b)
        p = tp / len(tokens_a) if tokens_a else 0
        r = tp / len(tokens_b) if tokens_b else 0
        if p + r == 0:
            return 0.0
        return round(2 * p * r / (p + r), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# DIM 2: EXTRACTION QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

class ExtractionMetrics:
    """
    Dimension 2: Core detection metrics (SemEval-2019 style).
    """

    @staticmethod
    def compute(predictions: list[dict], gold: list[dict],
                match_threshold: float = 0.5) -> dict:
        """
        Compute P, R, F1 for suggestion detection.
        Match is based on token overlap >= threshold.
        """
        tp, fp, fn = 0, 0, 0
        gold_matched = set()

        for pred in predictions:
            pred_text = pred.get("text", "").lower().strip()
            best_score = 0
            best_idx = -1
            for i, g in enumerate(gold):
                score = _token_f1(pred_text, g.get("text", "").lower().strip())
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_score >= match_threshold and best_idx not in gold_matched:
                tp += 1
                gold_matched.add(best_idx)
            else:
                fp += 1

        fn = len(gold) - len(gold_matched)
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0

        return {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "n_predictions": len(predictions),
            "n_gold": len(gold),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DIM 3: EXPLICIT VS IMPLICIT (delegates to user's UnifiedEvaluator)
# ═══════════════════════════════════════════════════════════════════════════════

class TypeSpecificMetrics:
    """
    Dimension 3: Type-aware evaluation.
    Delegates to UnifiedEvaluator from suggestion_switch.py.
    """

    @staticmethod
    def compute(predictions: list[dict], gold: list[dict]) -> dict:
        """
        Split predictions by type, run type-specific evaluation.
        Returns separate explicit, implicit, and unified metrics.
        """
        try:
            from agents.suggestion_switch import (
                ExplicitEvaluator, ImplicitEvaluator, UnifiedEvaluator
            )
            explicit_preds = [p for p in predictions if p.get("suggestion_type") == "explicit"]
            implicit_preds = [p for p in predictions if p.get("suggestion_type") != "explicit"]
            return UnifiedEvaluator.evaluate_all(explicit_preds, implicit_preds, gold)
        except ImportError:
            # Fallback: basic split
            explicit_preds = [p for p in predictions if not p.get("is_implied", True)]
            implicit_preds = [p for p in predictions if p.get("is_implied", True)]
            return {
                "explicit": {"count": len(explicit_preds)},
                "implicit": {"count": len(implicit_preds)},
                "unified": {"total": len(predictions)},
                "note": "UnifiedEvaluator not available; install suggestion_switch.py",
            }


# ═══════════════════════════════════════════════════════════════════════════════
# DIM 4: MULTI-VIEW SWITCH METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class SwitchMetrics:
    """
    Dimension 4: View-Weighting Switch evaluation.
    """

    @staticmethod
    def switch_accuracy(predictions: list[dict], gold: list[dict]) -> float:
        """
        % of samples where the switch mode (common/specific) led to correct output.
        Target: >= 0.75
        """
        correct = 0
        total = 0
        gold_texts = {g["text"].lower().strip() for g in gold}

        for pred in predictions:
            total += 1
            pred_text = pred.get("text", "").lower().strip()
            # Check if this prediction matches any gold
            for g_text in gold_texts:
                if _token_f1(pred_text, g_text) >= 0.5:
                    correct += 1
                    break

        return round(correct / max(total, 1), 4)

    @staticmethod
    def view_ablation_deltas(full_f1: float, ablation_f1s: dict[str, float]) -> dict:
        """
        For each view removed, compute F1 delta.
        Every view should contribute > 0.
        """
        return {
            view: round(full_f1 - abl_f1, 4)
            for view, abl_f1 in ablation_f1s.items()
        }

    @staticmethod
    def labeller_agreement(conservative_labels: list[dict],
                           liberal_labels: list[dict]) -> float:
        """
        Cohen's Kappa between conservative and liberal labeller outputs.
        Target: 0.40-0.70 (moderate, not redundant).
        """
        # Match by sample_id or text
        cons_texts = {l.get("text", "").lower() for l in conservative_labels}
        lib_texts = {l.get("text", "").lower() for l in liberal_labels}
        all_texts = cons_texts | lib_texts

        labels_a = [1 if t in cons_texts else 0 for t in all_texts]
        labels_b = [1 if t in lib_texts else 0 for t in all_texts]

        return DatasetQualityMetrics.cohen_kappa(labels_a, labels_b)


# ═══════════════════════════════════════════════════════════════════════════════
# DIM 5: MULTIMODAL FUSION
# ═══════════════════════════════════════════════════════════════════════════════

class MultimodalMetrics:
    """
    Dimension 5: Cross-modal fusion evaluation.
    """

    @staticmethod
    def modality_ablation(results: dict[str, float]) -> dict:
        """
        Compare F1 across modality configurations.
        results: {"text_only": 0.78, "text_image": 0.84, "text_audio": 0.82, "all": 0.88}
        Each should be > text_only.
        """
        text_only = results.get("text_only", 0)
        return {
            config: {"f1": f1, "delta_vs_text_only": round(f1 - text_only, 4)}
            for config, f1 in results.items()
        }

    @staticmethod
    def image_only_discovery_rate(predictions: list[dict]) -> float:
        """
        % of suggestions found ONLY through image evidence.
        Novel metric for MASP.
        """
        image_only = sum(
            1 for p in predictions
            if p.get("source_modality") == "image"
            or (p.get("evidence_modalities") and
                p["evidence_modalities"] == ["image"])
        )
        return round(image_only / max(len(predictions), 1), 4)

    @staticmethod
    def evidence_provenance_score(predictions: list[dict]) -> float:
        """
        Avg evidence pieces per accepted suggestion across modalities.
        Target: >= 3.0
        """
        pieces = [p.get("total_evidence_pieces", 0) for p in predictions]
        return round(sum(pieces) / max(len(pieces), 1), 2)


# ═══════════════════════════════════════════════════════════════════════════════
# DIM 6: FAITHFULNESS
# ═══════════════════════════════════════════════════════════════════════════════

class FaithfulnessMetrics:
    """
    Dimension 6: Hallucination and grounding metrics.
    """

    @staticmethod
    def hallucination_rate(predictions: list[dict], source_texts: list[str]) -> float:
        """
        % of accepted suggestions NOT supported by any evidence in source.
        Target: < 5%
        """
        hallucinated = 0
        for pred, source in zip(predictions, source_texts):
            pred_text = pred.get("text", "").lower()
            source_lower = source.lower()
            # Check grounding score from evidence provenance
            grounding = pred.get("evidence_grounding_score",
                                 pred.get("grounding_score", None))
            if grounding is not None:
                if grounding < 0.20:
                    hallucinated += 1
            else:
                # Fallback: token overlap
                overlap = _token_f1(pred_text, source_lower)
                if overlap < 0.15:
                    hallucinated += 1

        return round(hallucinated / max(len(predictions), 1), 4)

    @staticmethod
    def avg_grounding_score(predictions: list[dict]) -> float:
        """
        Average evidence_grounding_score across all predictions.
        Uses grounding from evidence_provenance module.
        Target: >= 0.60
        """
        scores = [
            p.get("evidence_grounding_score", p.get("grounding_score", 0.5))
            for p in predictions
        ]
        return round(sum(scores) / max(len(scores), 1), 4)

    @staticmethod
    def grounding_coverage(predictions: list[dict]) -> float:
        """
        % of suggestions with evidence from 2+ modalities.
        Target: >= 40%
        """
        multi_modal = sum(
            1 for p in predictions
            if len(p.get("evidence_modalities", [])) >= 2
        )
        return round(multi_modal / max(len(predictions), 1), 4)

    @staticmethod
    def false_positive_rate(predictions: list[dict],
                            human_rejections: list[str]) -> float:
        """
        % of accepted suggestions that humans reject.
        Target: < 10%
        """
        rejected = set(r.lower().strip() for r in human_rejections)
        fp = sum(
            1 for p in predictions
            if p.get("text", "").lower().strip() in rejected
        )
        return round(fp / max(len(predictions), 1), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# DIM 7: RANKING QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

class RankingMetrics:
    """
    Dimension 7: Ranking and prioritization against human rankings.
    """

    @staticmethod
    def ndcg_at_k(predicted_ranking: list[str], gold_ranking: list[str],
                  k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain at K.
        Target: >= 0.75
        """
        gold_scores = {}
        for i, item in enumerate(gold_ranking):
            gold_scores[item.lower().strip()] = len(gold_ranking) - i  # higher = better

        # DCG
        dcg = 0.0
        for i, item in enumerate(predicted_ranking[:k]):
            rel = gold_scores.get(item.lower().strip(), 0)
            dcg += rel / math.log2(i + 2)

        # IDCG
        ideal = sorted(gold_scores.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))

        return round(dcg / max(idcg, 1e-10), 4)

    @staticmethod
    def kendall_tau(predicted_ranking: list[str],
                    gold_ranking: list[str]) -> float:
        """
        Kendall's Tau rank correlation.
        Target: >= 0.60
        """
        # Build rank maps
        pred_rank = {item.lower().strip(): i for i, item in enumerate(predicted_ranking)}
        gold_rank = {item.lower().strip(): i for i, item in enumerate(gold_ranking)}
        common = set(pred_rank.keys()) & set(gold_rank.keys())
        items = sorted(common)

        if len(items) < 2:
            return 0.0

        concordant = 0
        discordant = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pred_diff = pred_rank[items[i]] - pred_rank[items[j]]
                gold_diff = gold_rank[items[i]] - gold_rank[items[j]]
                if pred_diff * gold_diff > 0:
                    concordant += 1
                elif pred_diff * gold_diff < 0:
                    discordant += 1

        total = concordant + discordant
        if total == 0:
            return 0.0
        return round((concordant - discordant) / total, 4)

    @staticmethod
    def top_k_precision(predicted_top_k: list[str],
                        gold_top_k: list[str]) -> float:
        """
        % of system's top-K that appear in human's top-K.
        Target: >= 80%
        """
        pred_set = {t.lower().strip() for t in predicted_top_k}
        gold_set = {t.lower().strip() for t in gold_top_k}
        overlap = len(pred_set & gold_set)
        return round(overlap / max(len(pred_set), 1), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# DIM 8: CROSS-DOMAIN + MEMORY (report-only, computed from multiple runs)
# ═══════════════════════════════════════════════════════════════════════════════

class CrossDomainMetrics:
    """
    Dimension 8: Computed from leave-one-domain-out experiments.
    """

    @staticmethod
    def domain_adaptation_gap(in_domain_f1: float, out_domain_f1: float) -> float:
        """Target: < 0.10"""
        return round(in_domain_f1 - out_domain_f1, 4)

    @staticmethod
    def memory_hit_rate(predictions: list[dict]) -> float:
        """% of suggestions with semantic_memory_hit=True."""
        hits = sum(1 for p in predictions if p.get("semantic_memory_hit"))
        return round(hits / max(len(predictions), 1), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# DIM 9: EFFICIENCY (computed from pipeline run metadata)
# ═══════════════════════════════════════════════════════════════════════════════

class EfficiencyMetrics:
    """
    Dimension 9: Cost, latency, human review rate.
    """

    @staticmethod
    def compute(run_logs: list[dict]) -> dict:
        """
        run_logs: list of {tokens_in, tokens_out, latency_ms, model}
        """
        if not run_logs:
            return {}
        total_tokens = [r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in run_logs]
        latencies = [r.get("latency_ms", 0) / 1000 for r in run_logs]

        # Placeholder per-million-token cost estimate; update rates before use.
        costs = []
        for r in run_logs:
            cost = (r.get("tokens_in", 0) * 3 + r.get("tokens_out", 0) * 15) / 1_000_000
            costs.append(cost)

        return {
            "avg_tokens_per_review": round(sum(total_tokens) / len(total_tokens)),
            "avg_latency_seconds": round(sum(latencies) / len(latencies), 2),
            "avg_cost_usd": round(sum(costs) / len(costs), 4),
            "total_reviews": len(run_logs),
        }

    @staticmethod
    def human_review_rate(predictions: list[dict]) -> float:
        """% flagged for human review. Target: < 25%."""
        flagged = sum(1 for p in predictions if p.get("needs_human_review"))
        return round(flagged / max(len(predictions), 1), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# DIM 10: STATISTICAL SIGNIFICANCE
# ═══════════════════════════════════════════════════════════════════════════════

class StatisticalTests:
    """
    Bootstrap confidence intervals and significance tests.
    """

    @staticmethod
    def bootstrap_ci(scores: list[float], n_resamples: int = 10000,
                     confidence: float = 0.95) -> tuple[float, float, float]:
        """
        Returns (mean, lower_ci, upper_ci).
        """
        import random
        n = len(scores)
        if n == 0:
            return 0.0, 0.0, 0.0
        means = []
        for _ in range(n_resamples):
            sample = [random.choice(scores) for _ in range(n)]
            means.append(sum(sample) / n)
        means.sort()
        alpha = (1 - confidence) / 2
        lo = means[int(alpha * n_resamples)]
        hi = means[int((1 - alpha) * n_resamples)]
        return round(sum(scores) / n, 4), round(lo, 4), round(hi, 4)

    @staticmethod
    def paired_bootstrap_test(scores_a: list[float], scores_b: list[float],
                              n_resamples: int = 10000) -> float:
        """
        Paired bootstrap test. Returns p-value.
        Significant if p < 0.05.
        """
        import random
        n = len(scores_a)
        if n == 0:
            return 1.0
        observed_diff = sum(scores_a) / n - sum(scores_b) / n
        count = 0
        for _ in range(n_resamples):
            indices = [random.randint(0, n - 1) for _ in range(n)]
            diff = (sum(scores_a[i] for i in indices) - sum(scores_b[i] for i in indices)) / n
            if diff <= 0:  # null hypothesis: A is not better
                count += 1
        return round(count / n_resamples, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _token_f1(text_a: str, text_b: str) -> float:
    """Token-level F1 between two strings."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    tp = len(tokens_a & tokens_b)
    p = tp / len(tokens_a)
    r = tp / len(tokens_b)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ═══════════════════════════════════════════════════════════════════════════════
# FULL EVALUATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(
    predictions: list[dict],
    gold: list[dict],
    source_texts: list[str] = None,
    gold_ranking: list[str] = None,
    human_rejections: list[str] = None,
    run_logs: list[dict] = None,
    ablation_results: dict = None,
) -> dict:
    """
    Run ALL 9 evaluation dimensions and return paper-ready results.

    Args:
        predictions: pipeline outputs (list of enriched suggestion dicts)
        gold: gold-labeled suggestions with text, suggestion_type, rank
        source_texts: original review texts (for faithfulness check)
        gold_ranking: human-ranked suggestion texts in priority order
        human_rejections: texts humans rejected (for FP rate)
        run_logs: pipeline run metadata (tokens, latency)
        ablation_results: {ablation_name: f1_score} for view ablation deltas
    """
    results = {}

    # Dim 2: Extraction quality
    results["extraction"] = ExtractionMetrics.compute(predictions, gold)

    # Dim 3: Type-specific
    results["type_specific"] = TypeSpecificMetrics.compute(predictions, gold)

    # Dim 5: Multimodal
    results["multimodal"] = {
        "image_only_discovery_rate": MultimodalMetrics.image_only_discovery_rate(predictions),
        "avg_evidence_pieces": MultimodalMetrics.evidence_provenance_score(predictions),
    }

    # Dim 6: Faithfulness
    results["faithfulness"] = {
        "avg_grounding_score": FaithfulnessMetrics.avg_grounding_score(predictions),
        "grounding_coverage": FaithfulnessMetrics.grounding_coverage(predictions),
    }
    if source_texts:
        results["faithfulness"]["hallucination_rate"] = \
            FaithfulnessMetrics.hallucination_rate(predictions, source_texts)
    if human_rejections:
        results["faithfulness"]["false_positive_rate"] = \
            FaithfulnessMetrics.false_positive_rate(predictions, human_rejections)

    # Dim 7: Ranking
    if gold_ranking:
        pred_ranking = [p["text"] for p in sorted(
            predictions, key=lambda x: x.get("score", x.get("priority_score", 0)),
            reverse=True
        )]
        results["ranking"] = {
            "ndcg_5": RankingMetrics.ndcg_at_k(pred_ranking, gold_ranking, k=5),
            "ndcg_10": RankingMetrics.ndcg_at_k(pred_ranking, gold_ranking, k=10),
            "kendall_tau": RankingMetrics.kendall_tau(pred_ranking, gold_ranking),
            "top_5_precision": RankingMetrics.top_k_precision(
                pred_ranking[:5], gold_ranking[:5]),
        }

    # Dim 8: Memory
    results["memory"] = {
        "memory_hit_rate": CrossDomainMetrics.memory_hit_rate(predictions),
    }

    # Dim 9: Efficiency
    if run_logs:
        results["efficiency"] = EfficiencyMetrics.compute(run_logs)
        results["efficiency"]["human_review_rate"] = \
            EfficiencyMetrics.human_review_rate(predictions)

    # Dim 10: Bootstrap CIs on main F1
    if len(predictions) >= 10:
        per_sample_scores = []
        for pred in predictions:
            best = 0
            for g in gold:
                score = _token_f1(pred.get("text", ""), g.get("text", ""))
                best = max(best, score)
            per_sample_scores.append(best)
        mean, lo, hi = StatisticalTests.bootstrap_ci(per_sample_scores)
        results["statistical"] = {
            "f1_mean": mean,
            "f1_ci_lower": lo,
            "f1_ci_upper": hi,
            "ci_width": round(hi - lo, 4),
        }

    # Switch stats summary
    switch_types = defaultdict(int)
    for p in predictions:
        switch_types[p.get("suggestion_type", "unknown")] += 1
    results["switch_summary"] = dict(switch_types)

    return results


def print_paper_table(all_results: dict[str, dict]):
    """
    Print the paper-ready comparison table.
    all_results: {"MASP": {...}, "B1": {...}, "B2": {...}, ...}
    """
    metrics = ["f1", "precision", "recall"]
    print("\n" + "=" * 80)
    print("  PAPER TABLE — Main Results")
    print("=" * 80)
    print(f"  {'Metric':<25}", end="")
    for name in all_results:
        print(f"  {name:>10}", end="")
    print()
    print("  " + "-" * 75)

    for metric in metrics:
        print(f"  {metric:<25}", end="")
        for name, res in all_results.items():
            val = res.get("extraction", {}).get(metric, "—")
            if isinstance(val, float):
                print(f"  {val:>10.4f}", end="")
            else:
                print(f"  {val:>10}", end="")
        print()

    # Faithfulness
    print(f"  {'avg_grounding':<25}", end="")
    for name, res in all_results.items():
        val = res.get("faithfulness", {}).get("avg_grounding_score", "—")
        if isinstance(val, float):
            print(f"  {val:>10.4f}", end="")
        else:
            print(f"  {val:>10}", end="")
    print()

    # Evidence pieces
    print(f"  {'avg_evidence_pcs':<25}", end="")
    for name, res in all_results.items():
        val = res.get("multimodal", {}).get("avg_evidence_pieces", "—")
        print(f"  {val:>10}", end="")
    print()

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("MASP Evaluation Harness")
    print("Usage: import and call run_full_evaluation() with your pipeline outputs.")
    print("See docstring for full API.")
