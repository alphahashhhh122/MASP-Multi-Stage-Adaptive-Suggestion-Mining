"""
memory/store.py

COLLECTIVE COGNITION MEMORY SYSTEM (MAS Memory framework, TechRxiv 2025)

Implements the transition from individual to collective cognition described in
"Memory in LLM-based Multi-agent Systems" — shared memories transform individual
agent knowledge into a team knowledge base.

5 Memory Types (each with short-term + long-term):
  1. SemanticMemory       — canonical form embeddings (agentic RAG pattern, A-RAG 2026)
  2. EpisodicMemory       — past labelling decisions + approval rates (temporal tracking)
  3. HumanEditMemory      — corrections made by human reviewers (active learning loop)
  4. ModalityAlignMemory  — which text+image alignment patterns led to accepted suggestions
  5. CrossDomainMemory    — suggestion patterns spanning multiple product domains
                            (collective cognition: "reduce wait time" transfers across
                             healthcare, restaurant, e-commerce)

The MemoryManager.lookup() actively BOOSTS confidence based on approval rate:
  confidence 0.75 -> boosted to 0.85 because 18/20 past decisions accepted.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """Stores every labelling decision (accept/reject) with human flag."""
    def __init__(self):
        self._short_term: list[dict] = []
        self._long_term: list[dict] = []

    def add(self, text: str, decision: str, human_approved: bool = False):
        entry = {"id": str(uuid.uuid4())[:8], "text": text, "decision": decision,
                 "human_approved": human_approved, "ts": datetime.utcnow().isoformat()}
        self._short_term.append(entry)
        self._long_term.append(entry)
        cutoff = datetime.utcnow() - timedelta(days=90)
        self._short_term = [e for e in self._short_term if datetime.fromisoformat(e["ts"]) >= cutoff]

    def approval_rate(self, text: str, use_long_term: bool = False) -> float:
        pool = self._long_term if use_long_term else self._short_term
        relevant = [e for e in pool if self._sim(text, e["text"]) > 0.55]
        if not relevant: return 0.5
        accepted = sum(1 for e in relevant if e["decision"] == "accept")
        return round(accepted / len(relevant), 3)

    def boost_confidence(self, text: str, base_confidence: float) -> float:
        rate = self.approval_rate(text)
        boost = (rate - 0.5) * 0.20
        return min(0.99, round(base_confidence + boost, 3))

    def gold_labels(self) -> list[dict]:
        return [e for e in self._long_term if e["human_approved"]]

    def recent_decisions(self, n: int = 20) -> list[dict]:
        return self._long_term[-n:]

    @staticmethod
    def _sim(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0

    def stats(self) -> dict:
        return {"short_term_count": len(self._short_term), "long_term_count": len(self._long_term),
                "gold_label_count": len(self.gold_labels())}


class SemanticMemory:
    """Stores canonical suggestion embeddings for similarity lookup (agentic RAG pattern)."""
    def __init__(self, collection_name: str = "suggestion_canonical"):
        self._short_term: dict[str, dict] = {}
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            self._client = chromadb.Client()
            self._ef = embedding_functions.DefaultEmbeddingFunction()
            self._col = self._client.get_or_create_collection(name=collection_name, embedding_function=self._ef)
            self._available = True
        except Exception:
            self._available = False
            self._long_term_fallback: dict[str, dict] = {}

    def add(self, canonical_text: str, metadata: dict, persist: bool = True):
        doc_id = str(uuid.uuid4())
        self._short_term[doc_id] = {"text": canonical_text, **metadata}
        if persist:
            if self._available:
                self._col.add(ids=[doc_id], documents=[canonical_text], metadatas=[metadata])
            else:
                self._long_term_fallback[doc_id] = {"text": canonical_text, **metadata}

    def query(self, text: str, top_k: int = 3) -> list[dict]:
        results = []
        for entry in self._short_term.values():
            sim = self._jaccard(text, entry["text"])
            if sim > 0.4:
                results.append({"canonical_text": entry["text"], "similarity": sim, **entry})
        if self._available and self._col.count() > 0:
            try:
                r = self._col.query(query_texts=[text], n_results=min(top_k, self._col.count()))
                for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
                    results.append({"canonical_text": doc, "similarity": 1.0 - dist, **meta})
            except Exception: pass
        else:
            for entry in getattr(self, '_long_term_fallback', {}).values():
                sim = self._jaccard(text, entry["text"])
                results.append({"canonical_text": entry["text"], "similarity": sim, **entry})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        seen, deduped = set(), []
        for r in results:
            if r["canonical_text"] not in seen:
                seen.add(r["canonical_text"]); deduped.append(r)
        return deduped[:top_k]

    def count(self) -> int:
        base = self._col.count() if self._available else len(getattr(self, '_long_term_fallback', {}))
        return base + len(self._short_term)

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0


class HumanEditMemory:
    """Stores human-corrected suggestion texts (active learning loop, ACL 2025 survey)."""
    def __init__(self):
        self._short_term: list[dict] = []
        self._long_term: list[dict] = []

    def add_correction(self, original: str, corrected: str, reviewer_id: str = "human"):
        entry = {"original": original, "corrected": corrected, "reviewer_id": reviewer_id,
                 "ts": datetime.utcnow().isoformat()}
        self._short_term.append(entry); self._long_term.append(entry)

    def apply_corrections(self, text: str) -> str:
        ta = set(text.lower().split())
        best_sim, best_corrected = 0.0, None
        for entry in self._long_term + self._short_term:
            tb = set(entry["original"].lower().split())
            sim = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
            if sim > 0.75 and sim > best_sim:
                best_sim, best_corrected = sim, entry["corrected"]
        return best_corrected if best_corrected else text

    def all_corrections(self) -> list[dict]:
        """Return all stored human corrections (for analysis/paper reporting)."""
        return self._long_term

    def stats(self) -> dict:
        return {"short_term_edits": len(self._short_term), "long_term_edits": len(self._long_term)}


class ModalityAlignMemory:
    """Stores which cross-modal alignment patterns led to accepted vs rejected suggestions."""
    def __init__(self):
        self._short_term: list[dict] = []
        self._long_term: list[dict] = []

    def record(self, alignment_score: float, dominant_modality: str, domain: str, decision: str):
        entry = {"alignment_score": alignment_score, "dominant_modality": dominant_modality,
                 "domain": domain, "decision": decision, "ts": datetime.utcnow().isoformat()}
        self._short_term.append(entry); self._long_term.append(entry)
        cutoff = datetime.utcnow() - timedelta(days=30)
        self._short_term = [e for e in self._short_term if datetime.fromisoformat(e["ts"]) >= cutoff]

    def expected_acceptance(self, alignment_score: float, dominant_modality: str, domain: str) -> float:
        similar = [e for e in self._long_term
                   if abs(e["alignment_score"] - alignment_score) < 0.15
                   and e["dominant_modality"] == dominant_modality and e["domain"] == domain]
        if len(similar) < 3: return 0.5
        return round(sum(1 for e in similar if e["decision"] == "accept") / len(similar), 3)

    def stats(self) -> dict:
        return {"short_term_patterns": len(self._short_term), "long_term_patterns": len(self._long_term)}


class CrossDomainMemory:
    """
    Collective Cognition Memory (MAS Memory framework, TechRxiv 2025).
    Stores suggestion patterns spanning multiple product domains.
    Example: "Add dark mode" appears in mobile_app + saas + gaming -> universal pattern.
    """
    def __init__(self):
        self._patterns: dict[str, dict] = {}

    def record(self, canonical_text: str, domain: str, accepted: bool):
        key = canonical_text.lower().strip()
        if key not in self._patterns:
            self._patterns[key] = {"canonical_text": canonical_text, "domains": set(), "count": 0, "accepted_count": 0}
        self._patterns[key]["domains"].add(domain)
        self._patterns[key]["count"] += 1
        if accepted: self._patterns[key]["accepted_count"] += 1

    def is_cross_domain(self, canonical_text: str, min_domains: int = 2) -> bool:
        key = canonical_text.lower().strip()
        pattern = self._patterns.get(key)
        if not pattern:
            ta = set(canonical_text.lower().split())
            for k, p in self._patterns.items():
                tb = set(k.split())
                if len(ta & tb) / len(ta | tb) > 0.7: return len(p["domains"]) >= min_domains
        return pattern and len(pattern["domains"]) >= min_domains

    def cross_domain_boost(self, canonical_text: str) -> float:
        key = canonical_text.lower().strip()
        pattern = self._patterns.get(key)
        if not pattern: return 0.0
        return min(0.10, (len(pattern["domains"]) - 1) * 0.03)

    def top_universal_patterns(self, n: int = 10) -> list[dict]:
        patterns = [{**v, "domains": list(v["domains"]), "domain_count": len(v["domains"]),
                     "acceptance_rate": round(v["accepted_count"] / v["count"], 3) if v["count"] else 0}
                    for v in self._patterns.values()]
        return sorted(patterns, key=lambda p: p["domain_count"], reverse=True)[:n]

    def stats(self) -> dict:
        return {"unique_patterns": len(self._patterns),
                "cross_domain_count": sum(1 for p in self._patterns.values() if len(p["domains"]) >= 2)}


class MemoryManager:
    """Unified interface for all 5 memory types (collective cognition framework)."""
    def __init__(self):
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory()
        self.human_edit = HumanEditMemory()
        self.modality = ModalityAlignMemory()
        self.cross_domain = CrossDomainMemory()

    def lookup(self, suggestions, alignment_score=0.5, dominant_modality="text", domain="general"):
        """Collective memory retrieval with active confidence boosting."""
        results = {}
        for s in suggestions:
            text = s.get("canonical_text") or s.get("text", "")
            base_conf = s.get("confidence", 0.5)
            similar = self.semantic.query(text, top_k=3)
            memory_hit = any(h["similarity"] > 0.75 for h in similar)
            approval_rate = self.episodic.approval_rate(text)
            boosted = self.episodic.boost_confidence(text, base_conf)
            corrected = self.human_edit.apply_corrections(text)
            xd_boost = self.cross_domain.cross_domain_boost(text)
            boosted = min(0.99, boosted + xd_boost)
            modal_expected = self.modality.expected_acceptance(alignment_score, dominant_modality, domain)
            results[text] = {"past_approval_rate": approval_rate, "semantic_memory_hit": memory_hit,
                             "memory_hit": memory_hit, "similar_canonical_forms": similar,
                             "boosted_confidence": boosted, "corrected_text": corrected,
                             "is_human_corrected": corrected != text,
                             "cross_domain": self.cross_domain.is_cross_domain(text),
                             "cross_domain_boost": xd_boost, "modal_expected_acceptance": modal_expected}
        return results

    def store_canonical(self, suggestions, domain="general", accepted=True):
        for s in suggestions:
            text = s.get("canonical_text") or s.get("text", "")
            meta = {"priority_tier": s.get("priority_tier", "MEDIUM"), "domain": domain,
                    "frequency": str(s.get("frequency", 1))}
            self.semantic.add(text, meta)
            self.cross_domain.record(text, domain, accepted)

    def record_decision(self, text, decision, human=False, alignment_score=0.5,
                        dominant_modality="text", domain="general"):
        self.episodic.add(text, decision, human_approved=human)
        self.modality.record(alignment_score, dominant_modality, domain, decision)
        self.cross_domain.record(text, domain, accepted=(decision == "accept"))

    def record_human_edit(self, original, corrected, reviewer="human"):
        self.human_edit.add_correction(original, corrected, reviewer)
        self.episodic.add(corrected, "accept", human_approved=True)

    def apply_memory_to_suggestions(self, suggestions, alignment_score=0.5,
                                    dominant_modality="text", domain="general"):
        hits = self.lookup(suggestions, alignment_score, dominant_modality, domain)
        enriched = []
        for s in suggestions:
            text = s.get("canonical_text") or s.get("text", "")
            mem = hits.get(text, {})
            enriched.append({**s, "canonical_text": mem.get("corrected_text", text),
                             "confidence": mem.get("boosted_confidence", s.get("confidence", 0.5)),
                             "semantic_memory_hit": mem.get("semantic_memory_hit", False),
                             "past_approval_rate": mem.get("past_approval_rate", 0.5),
                             "cross_domain": mem.get("cross_domain", False),
                             "modal_alignment_score": mem.get("modal_expected_acceptance", 0.5),
                             "similar_forms": mem.get("similar_canonical_forms", [])})
        return enriched

    def stats(self) -> dict:
        return {"semantic": {"count": self.semantic.count()}, "episodic": self.episodic.stats(),
                "human_edit": self.human_edit.stats(), "modality": self.modality.stats(),
                "cross_domain": self.cross_domain.stats()}


_memory_manager: Optional[MemoryManager] = None

def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None: _memory_manager = MemoryManager()
    return _memory_manager
