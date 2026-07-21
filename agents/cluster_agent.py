"""
agents/cluster_agent.py

Cluster Agent — groups canonical suggestions from MULTIPLE reviews into clusters.
cluster_size (how many reviews mentioned this) flows to reranker as the most powerful signal.
"""

import json
import logging
import uuid
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SuggestionCluster:
    def __init__(self, cluster_id: str, seed_text: str, domain: str):
        self.cluster_id = cluster_id
        self.seed_text = seed_text
        self.domain = domain
        self.members: list[dict] = []
        self.centroid_text = seed_text
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at

    @property
    def size(self) -> int:
        return len(self.members)

    def add_member(self, suggestion_text: str, sample_id: str, confidence: float):
        self.members.append({
            "text": suggestion_text, "sample_id": sample_id,
            "confidence": confidence, "added_at": datetime.utcnow().isoformat()
        })
        self.updated_at = datetime.utcnow().isoformat()
        self.centroid_text = max(self.members, key=lambda m: m["confidence"])["text"]

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id, "centroid_text": self.centroid_text,
            "domain": self.domain, "size": self.size, "members": self.members,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }


class ClusterStore:
    def __init__(self, similarity_threshold: float = 0.72):
        self._clusters: dict[str, SuggestionCluster] = {}
        self.similarity_threshold = similarity_threshold
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                self._embedder = "jaccard"
        return self._embedder

    def _similarity(self, text_a: str, text_b: str) -> float:
        embedder = self._get_embedder()
        if embedder == "jaccard":
            a = set(text_a.lower().split())
            b = set(text_b.lower().split())
            return len(a & b) / len(a | b) if (a | b) else 0.0
        import numpy as np
        embs = embedder.encode([text_a, text_b])
        dot = np.dot(embs[0], embs[1])
        norms = np.linalg.norm(embs[0]) * np.linalg.norm(embs[1])
        return float(dot / norms) if norms > 0 else 0.0

    def assign(self, canonical_text: str, sample_id: str, domain: str, confidence: float) -> dict:
        domain_clusters = [c for c in self._clusters.values() if c.domain == domain]
        best_cluster, best_sim = None, 0.0
        for cluster in domain_clusters:
            sim = self._similarity(canonical_text, cluster.centroid_text)
            if sim > best_sim:
                best_sim, best_cluster = sim, cluster

        if best_cluster and best_sim >= self.similarity_threshold:
            best_cluster.add_member(canonical_text, sample_id, confidence)
            return {"cluster_id": best_cluster.cluster_id, "cluster_size": best_cluster.size}
        else:
            cid = f"cluster_{uuid.uuid4().hex[:8]}"
            cluster = SuggestionCluster(cid, canonical_text, domain)
            cluster.add_member(canonical_text, sample_id, confidence)
            self._clusters[cid] = cluster
            return {"cluster_id": cid, "cluster_size": 1}

    def get_cluster(self, cluster_id: str) -> Optional[SuggestionCluster]:
        """Return a single cluster by ID."""
        return self._clusters.get(cluster_id)

    def get_all_clusters(self, domain: str = None) -> list[dict]:
        clusters = self._clusters.values()
        if domain:
            clusters = [c for c in clusters if c.domain == domain]
        return [c.to_dict() for c in sorted(clusters, key=lambda c: c.size, reverse=True)]

    def top_clusters(self, n: int = 10, domain: str = None) -> list[dict]:
        return self.get_all_clusters(domain)[:n]

    def stats(self) -> dict:
        sizes = [c.size for c in self._clusters.values()]
        return {
            "total_clusters": len(self._clusters),
            "total_assignments": sum(sizes),
            "avg_cluster_size": round(sum(sizes) / len(sizes), 2) if sizes else 0,
            "max_cluster_size": max(sizes) if sizes else 0,
            "singleton_clusters": sum(1 for s in sizes if s == 1),
        }


_cluster_store: Optional[ClusterStore] = None

def get_cluster_store() -> ClusterStore:
    global _cluster_store
    if _cluster_store is None:
        _cluster_store = ClusterStore()
    return _cluster_store


def cluster_node(state) -> dict:
    logger.info(f"[cluster_node] {state['sample_id']}")
    store = get_cluster_store()
    domain = state.get("domain", "general")
    enriched = []
    for s in state.get("canonical_suggestions", []):
        text = s.get("canonical_text") or s.get("text", "")
        result = store.assign(text, state["sample_id"], domain, s.get("confidence", 0.5))
        enriched.append({**s, "cluster_id": result["cluster_id"], "cluster_size": result["cluster_size"]})
    return {"canonical_suggestions": enriched, "cluster_stats": store.stats()}
