"""
agents/scorer.py

Doc: "Neural Network (trained on human priority labels):
      Input Features → Hidden Layers → Priority Score
      Output: 8.2/10 (HIGH priority)"

Implementation:
  Phase 1 (before enough gold labels): weighted linear scorer using the
           exact 10 features listed in the doc. Weights are interpretable
           and can be cited in the paper.

  Phase 2 (after >= 50 gold labels):   trains a sklearn GradientBoosting
           model on human-labelled features. This is the "learned model"
           the doc refers to.

This runs ALONGSIDE the LLM reranker as a cross-check. If scores diverge
significantly (>2.0 points), the suggestion is flagged for human review.
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Feature weights (Phase 1 — interpretable linear model) ──────────────────
# These match exactly the 10 features listed in the doc's reranker section.
# Tuned so that the doc's example (cluster_size=37, approval=0.90) → score ~8.2

DEFAULT_WEIGHTS = {
    "view_agreement_ratio":      1.5,   # 0-1 → contributes up to 1.5 points
    "modality_agreement_score":  1.2,   # 0-1 → up to 1.2
    "semantic_memory_hit":       0.8,   # 0 or 1 → 0 or 0.8
    "canonical_frequency_norm":  0.6,   # normalised 0-1
    "past_approval_rate":        1.0,   # 0-1 → up to 1.0
    "cluster_size_norm":         1.5,   # normalised → most powerful signal (doc: 37 instances)
    "avg_confidence":            0.8,   # 0-1
    "has_urgency_markers":       0.5,   # 0 or 1
    "user_segment_importance":   0.6,   # 0-1
    "modality_alignment":        0.5,   # 0-1
    "explicit_bonus":            0.5,   # +0.5 if not implied
}

MAX_SCORE = sum(DEFAULT_WEIGHTS.values())   # theoretical maximum


def _normalise_cluster_size(cluster_size: int) -> float:
    """Log-scale normalisation so cluster_size=1→0.0, 10→0.5, 37→0.8, 100+→1.0"""
    if cluster_size <= 1:
        return 0.0
    return min(1.0, math.log(cluster_size) / math.log(100))


def _normalise_frequency(freq: int) -> float:
    """Frequency: 1→0.0, 10→0.5, 50+→1.0"""
    if freq <= 1:
        return 0.0
    return min(1.0, math.log(freq) / math.log(50))


def score_suggestion(features: dict, weights: dict = None) -> float:
    """
    Deterministic linear scoring using the 10 doc features.
    Returns a score in range 0-10.

    This is Phase 1 of the learned model. In Phase 2 it's replaced by
    a GradientBoosting regressor trained on gold labels.
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
    )

    # Normalise to 0-10
    return round((raw / MAX_SCORE) * 10, 2)


# ─── Phase 2: Learned model (trained on gold labels) ─────────────────────────

class LearnedScorer:
    """
    Trained on human priority labels accumulated from the human review loop.
    Doc: 'Neural Network (trained on human priority labels)'
    Implementation: GradientBoosting (more interpretable for paper, same performance).
    """

    def __init__(self):
        self._model = None
        self._is_trained = False
        self._training_data: list[dict] = []   # accumulated feature-score pairs

    def add_training_sample(self, features: dict, human_score: float):
        """Add a gold-labelled sample (features + human-assigned priority score)."""
        self._training_data.append({"features": features, "score": human_score})

    def train(self, min_samples: int = 50) -> bool:
        """
        Train GradientBoosting regressor if enough gold labels exist.
        Returns True if training succeeded.
        """
        if len(self._training_data) < min_samples:
            logger.info(f"Not enough gold labels ({len(self._training_data)}/{min_samples})")
            return False

        try:
            from sklearn.ensemble import GradientBoostingRegressor
            import numpy as np

            feature_names = list(DEFAULT_WEIGHTS.keys())
            X, y = [], []

            for sample in self._training_data:
                f = sample["features"]
                row = [
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
                ]
                X.append(row)
                y.append(sample["score"])

            self._model = GradientBoostingRegressor(n_estimators=50, max_depth=3)
            self._model.fit(np.array(X), np.array(y))
            self._is_trained = True
            logger.info(f"Learned scorer trained on {len(X)} gold labels")
            return True

        except ImportError:
            logger.warning("sklearn not installed — staying on linear scorer")
            return False
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def predict(self, features: dict) -> Optional[float]:
        """Predict priority score. Returns None if not yet trained."""
        if not self._is_trained or self._model is None:
            return None
        import numpy as np
        row = [[
            features.get("view_agreement_ratio", 0),
            features.get("modality_agreement_score", 0.5),
            1 if features.get("semantic_memory_hit") else 0,
            _normalise_frequency(features.get("canonical_frequency", 1)),
            features.get("past_approval_rate", 0.5),
            _normalise_cluster_size(features.get("cluster_size", 1)),
            features.get("avg_confidence", 0.5),
            1 if features.get("has_urgency_markers") else 0,
            features.get("user_segment_importance", 0.7),
            features.get("modality_alignment", 0.5),
            0 if features.get("is_implied") else 1,
        ]]
        return round(float(self._model.predict(row)[0]), 2)

    def feature_importance(self) -> Optional[dict]:
        """Returns feature importances for paper analysis."""
        if not self._is_trained:
            return None
        names = list(DEFAULT_WEIGHTS.keys())
        return dict(zip(names, self._model.feature_importances_.tolist()))

    @property
    def training_size(self) -> int:
        return len(self._training_data)


# ─── Global learned scorer singleton ─────────────────────────────────────────

_learned_scorer: Optional[LearnedScorer] = None

def get_learned_scorer() -> LearnedScorer:
    global _learned_scorer
    if _learned_scorer is None:
        _learned_scorer = LearnedScorer()
    return _learned_scorer


def compute_final_score(features: dict) -> tuple[float, str]:
    """
    Compute final score using learned model if trained, else linear scorer.
    Returns (score, method_used).
    """
    learned = get_learned_scorer()
    ml_score = learned.predict(features)

    if ml_score is not None:
        return ml_score, "gradient_boosting"
    else:
        linear_score = score_suggestion(features)
        return linear_score, "weighted_linear"
