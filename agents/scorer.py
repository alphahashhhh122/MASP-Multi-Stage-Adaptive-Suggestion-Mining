"""
agents/scorer.py

Deterministic linear scorer (Phase 1) + GradientBoosting (Phase 2).

UPDATED: Integrates features from both:
  - evidence_provenance grounding_score
  - suggestion_switch (explicit/implicit routing) → actionability, feasibility, faithfulness, type_bonus

Feature set: weighted ranking features from agreement, memory, evidence, and type scores.

New features from pipeline_integration.py SCORER_PATCH:
  grounding_score       — evidence provenance faithfulness (0-1)
  actionability_norm    — implicit path quality gate (1-5 normalized to 0-1)
  feasibility_norm      — implicit path quality gate (1-5 normalized to 0-1)
  faithfulness_score    — switch's faithfulness check (0-1)
  type_bonus            — explicit suggestions get +0.3 (inherently more reliable)
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


DEFAULT_WEIGHTS = {
    # ── Original 10 features ──
    "view_agreement_ratio":      1.3,
    "modality_agreement_score":  1.0,
    "semantic_memory_hit":       0.7,
    "canonical_frequency_norm":  0.5,
    "past_approval_rate":        0.9,
    "cluster_size_norm":         1.5,   # most powerful signal
    "avg_confidence":            0.7,
    "has_urgency_markers":       0.4,
    "user_segment_importance":   0.5,
    "modality_alignment":        0.4,
    "explicit_bonus":            0.4,
    # ── NEW: evidence provenance (Q-S-E) ──
    "grounding_score":           1.0,   # high weight — faithfulness matters
    # ── NEW: suggestion switch type-aware features ──
    "actionability_norm":        0.7,   # implicit: 0-1 normalized from 1-5
    "feasibility_norm":          0.5,   # implicit: 0-1 normalized from 1-5
    "faithfulness_score":        0.6,   # switch faithfulness check
    "type_bonus":                0.3,   # +0.3 for explicit (inherently safer)
}

MAX_SCORE = sum(DEFAULT_WEIGHTS.values())


def _normalise_cluster_size(cluster_size: int) -> float:
    if cluster_size <= 1: return 0.0
    return min(1.0, math.log(cluster_size) / math.log(100))

def _normalise_frequency(freq: int) -> float:
    if freq <= 1: return 0.0
    return min(1.0, math.log(freq) / math.log(50))


def score_suggestion(features: dict, weights: dict = None) -> float:
    """
    Deterministic linear scoring using weighted features. Returns 0-10.
    """
    w = weights or DEFAULT_WEIGHTS

    raw = (
        w["view_agreement_ratio"]     * features.get("view_agreement_ratio", 0.0)
        + w["modality_agreement_score"] * features.get("modality_agreement_score", 0.5)
        + w["semantic_memory_hit"]      * (1 if features.get("semantic_memory_hit") else 0)
        + w["canonical_frequency_norm"] * _normalise_frequency(features.get("canonical_frequency", 1))
        + w["past_approval_rate"]       * features.get("past_approval_rate", 0.5)
        + w["cluster_size_norm"]        * _normalise_cluster_size(features.get("cluster_size", 1))
        + w["avg_confidence"]           * features.get("avg_confidence", 0.5)
        + w["has_urgency_markers"]      * (1 if features.get("has_urgency_markers") else 0)
        + w["user_segment_importance"]  * features.get("user_segment_importance", 0.7)
        + w["modality_alignment"]       * features.get("modality_alignment", 0.5)
        + w["explicit_bonus"]           * (0 if features.get("is_implied") else 1)
        # NEW: evidence provenance (reads evidence_grounding_score set by user's node)
        + w["grounding_score"]          * features.get("grounding_score",
                                            features.get("evidence_grounding_score", 0.5))
        # NEW: suggestion switch type-aware (from pipeline_integration.py SCORER_PATCH)
        + w["actionability_norm"]       * (features.get("actionability_score", 3) / 5.0)
        + w["feasibility_norm"]         * (features.get("feasibility_score", 3) / 5.0)
        + w["faithfulness_score"]       * features.get("faithfulness_score", 0.5)
        + w["type_bonus"]              * (1 if features.get("suggestion_type") == "explicit" else 0)
    )

    return round((raw / MAX_SCORE) * 10, 2)


# ─── Phase 2: Learned model ──────────────────────────────────────────────────

class LearnedScorer:
    def __init__(self):
        self._model = None
        self._is_trained = False
        self._training_data: list[dict] = []

    def add_training_sample(self, features: dict, human_score: float):
        self._training_data.append({"features": features, "score": human_score})

    def _feature_vector(self, f: dict) -> list:
        """Build 15-element feature vector."""
        return [
            f.get("view_agreement_ratio", 0),
            f.get("modality_agreement_score", 0.5),
            1 if f.get("semantic_memory_hit") else 0,
            _normalise_frequency(f.get("canonical_frequency", 1)),
            f.get("past_approval_rate", 0.5),
            _normalise_cluster_size(f.get("cluster_size", 1)),
            f.get("avg_confidence", 0.5),
            1 if f.get("has_urgency_markers") else 0,
            f.get("user_segment_importance", 0.7),
            f.get("modality_alignment", 0.5),
            0 if f.get("is_implied") else 1,
            # New
            f.get("grounding_score", f.get("evidence_grounding_score", 0.5)),
            f.get("actionability_score", 3) / 5.0,
            f.get("feasibility_score", 3) / 5.0,
            f.get("faithfulness_score", 0.5),
        ]

    def train(self, min_samples: int = 50) -> bool:
        if len(self._training_data) < min_samples:
            return False
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            import numpy as np
            X = [self._feature_vector(s["features"]) for s in self._training_data]
            y = [s["score"] for s in self._training_data]
            self._model = GradientBoostingRegressor(n_estimators=50, max_depth=3)
            self._model.fit(np.array(X), np.array(y))
            self._is_trained = True
            logger.info(f"Learned scorer trained on {len(X)} gold labels (15 features)")
            return True
        except ImportError:
            logger.warning("sklearn not installed — staying on linear scorer")
            return False

    def predict(self, features: dict) -> Optional[float]:
        if not self._is_trained or self._model is None:
            return None
        import numpy as np
        return round(float(self._model.predict([self._feature_vector(features)])[0]), 2)

    def feature_importance(self) -> Optional[dict]:
        if not self._is_trained: return None
        names = list(DEFAULT_WEIGHTS.keys())
        return dict(zip(names, self._model.feature_importances_.tolist()))

    @property
    def training_size(self) -> int:
        return len(self._training_data)


_learned_scorer: Optional[LearnedScorer] = None

def get_learned_scorer() -> LearnedScorer:
    global _learned_scorer
    if _learned_scorer is None:
        _learned_scorer = LearnedScorer()
    return _learned_scorer

def compute_final_score(features: dict) -> tuple[float, str]:
    learned = get_learned_scorer()
    ml_score = learned.predict(features)
    if ml_score is not None:
        return ml_score, "gradient_boosting"
    return score_suggestion(features), "weighted_linear"
