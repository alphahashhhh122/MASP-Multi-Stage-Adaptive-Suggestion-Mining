"""
graph/pipeline.py - LangGraph state graph.

Full topology:

         START
           │
   [preprocess_node]           Layer 1: text+image+audio
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
[text_  [image_ [audio_        Layers 2/3/4: parallel views
 views]  views]  views]
     └─────┼─────┘
           │
  [cross_modal_align]          Layer 5: alignment → View-Weighting Switch
           │
  [domain_router]              Layer 6: domain classification (agentic RAG)
           │
     ┌─────┼─────┐
     ▼           ▼
[conservative] [liberal]       Layer 7: MVP-aware dual labellers
     └─────┼─────┘
           │
   [merge_labels]              fan-in
           │
   [arbitration]               Layer 8: consensus + cross-modal boosting
           │
    ┌──────┼──────┐
    │no accepted  │accepted
    ▼             ▼
   END   [evidence_provenance] Layer 8.5: evidence provenance
                  │
          [suggestion_switch]  Layer 8.6: explicit/implicit routing + type eval
                  │
          [canonicaliser]      Layer 9: de-dup + standardise
                  │
           [cluster_node]      Layer 9b: cluster across reviews
                  │
          [memory_node]        Layer 10: collective cognition (MAS Memory)
                  │
          [reranker_node]      Layer 11: View-Weighting Switch + view-weighted scoring
                  │
    [human_review_gate]        Layer 12: flag uncertain cases
                  │
                 END

Integration notes:
  - evidence_provenance grounding_score enriches accepted_suggestions BEFORE
    the switch, so the switch's faithfulness gate can use it.
  - suggestion_switch uses the FULL user version (1233 lines) with:
    * SuggestionType classifier (explicit/implicit/ambiguous)
    * ExplicitResult / ImplicitResult dataclasses
    * UnifiedEvaluator for type-specific metrics
    * Ambiguous case handling (run both paths, pick winner)
  - Switch enriches suggestions with: suggestion_type, actionability_score,
    feasibility_score, specificity_score, faithfulness_score, inference_chain,
    type_adjusted_score, switch_eval_metrics
  - These flow to reranker as additional scoring features.
"""

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from graph.state import PipelineState
from agents.nodes import (
    preprocess_node,
    text_view_builder_node,
    image_view_builder_node,
    audio_view_builder_node,
    cross_modal_align_node,
    domain_router_node,
    conservative_labeller_node,
    liberal_labeller_node,
    merge_labels_node,
    arbitration_node,
    canonicaliser_node,
    memory_node,
    reranker_node,
    human_review_gate_node,
)
from agents.cluster_agent import cluster_node
from agents.evidence_provenance import evidence_provenance_node
from agents.suggestion_switch import suggestion_switch_node


def _route_after_arbitration(state: PipelineState) -> str:
    if state.get("error"):
        return "end"
    if not state.get("accepted_suggestions"):
        return "end"
    return "evidence_provenance"


def build_graph(checkpointer=None):
    graph = StateGraph(PipelineState)

    # Register graph nodes.
    graph.add_node("preprocess",            preprocess_node)
    graph.add_node("text_views",            text_view_builder_node)
    graph.add_node("image_views",           image_view_builder_node)
    graph.add_node("audio_views",           audio_view_builder_node)
    graph.add_node("cross_modal_align",     cross_modal_align_node)
    graph.add_node("domain_router",         domain_router_node)
    graph.add_node("conservative_labeller", conservative_labeller_node)
    graph.add_node("liberal_labeller",      liberal_labeller_node)
    graph.add_node("merge_labels",          merge_labels_node)
    graph.add_node("arbitration",           arbitration_node)
    graph.add_node("evidence_provenance",   evidence_provenance_node)   # NEW
    graph.add_node("suggestion_switch",     suggestion_switch_node)     # NEW
    graph.add_node("canonicaliser",         canonicaliser_node)
    graph.add_node("cluster_agent",         cluster_node)
    graph.add_node("memory_agent",          memory_node)
    graph.add_node("reranker",              reranker_node)
    graph.add_node("human_review_gate",     human_review_gate_node)

    # ── Edges ────────────────────────────────────────────────────────────────

    graph.add_edge(START, "preprocess")

    # Parallel view building
    graph.add_edge("preprocess", "text_views")
    graph.add_edge("preprocess", "image_views")
    graph.add_edge("preprocess", "audio_views")

    # Fan-in to cross-modal
    graph.add_edge("text_views",  "cross_modal_align")
    graph.add_edge("image_views", "cross_modal_align")
    graph.add_edge("audio_views", "cross_modal_align")

    graph.add_edge("cross_modal_align", "domain_router")

    # Parallel labelling (MVP-aware)
    graph.add_edge("domain_router", "conservative_labeller")
    graph.add_edge("domain_router", "liberal_labeller")

    # Fan-in to arbitration
    graph.add_edge("conservative_labeller", "merge_labels")
    graph.add_edge("liberal_labeller",      "merge_labels")
    graph.add_edge("merge_labels",          "arbitration")

    # Conditional: accepted → evidence + switch chain, else END
    graph.add_conditional_edges(
        "arbitration",
        _route_after_arbitration,
        {"evidence_provenance": "evidence_provenance", "end": END}
    )

    # NEW: evidence → switch → canonicaliser
    graph.add_edge("evidence_provenance", "suggestion_switch")
    graph.add_edge("suggestion_switch",   "canonicaliser")

    # Continue existing chain
    graph.add_edge("canonicaliser",     "cluster_agent")
    graph.add_edge("cluster_agent",     "memory_agent")
    graph.add_edge("memory_agent",      "reranker")
    graph.add_edge("reranker",          "human_review_gate")
    graph.add_edge("human_review_gate", END)

    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


def build_graph_with_memory():
    return build_graph(checkpointer=MemorySaver())
