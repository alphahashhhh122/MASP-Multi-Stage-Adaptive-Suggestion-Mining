"""
graph/pipeline.py  —  LangGraph state graph with ALL gaps fixed.

Full topology:

         START
           |
   [preprocess_node]           Layer 1: text+image+audio + discourse markers + span similarity
           |
     +-----+-----+-----+
     v     v     v
[text_  [image_ [audio_        Layers 2/3/4: parallel per-modality view building
 views]  views]  views]
     +-----+-----+-----+
           |
  [cross_modal_align]          Layer 5: alignment score -> drives View-Weighting Switch
           |
  [domain_router]              Layer 6: domain classification
           |
     +-----+-----+
     v           v
[conservative] [liberal]       Layer 7: parallel labellers, see ALL modality views
     +-----+-----+
           |
   [merge_labels]              fan-in
           |
   [arbitration]               Layer 8: consensus + cross-modal confidence boosting
           |
    +------+------+
    |no accepted  |accepted
    v             v
   END    [canonicaliser]      Layer 9:  de-dup + standardise (embedding-based intent)
                  |
           [cluster_node]      Layer 9b: GAP 1 FIX - cluster across reviews, adds cluster_size
                  |
          [memory_node]        Layer 10: semantic + episodic memory enrichment
                  |
          [reranker_node]      Layer 11: VIEW-WEIGHTING SWITCH + full 9-feature scoring
                  |
    [human_review_gate]        Layer 12: flag uncertain cases
                  |
                 END
"""

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from .state import PipelineState
from ..agents.nodes import (
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
from ..agents.cluster_agent import cluster_node


def _route_after_arbitration(state: PipelineState) -> str:
    if state.get("error"):
        return "end"
    if not state.get("accepted_suggestions"):
        return "end"
    return "canonicaliser"


def build_graph(checkpointer=None):
    graph = StateGraph(PipelineState)

    # ── Register nodes ───────────────────────────────────────────────────────
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
    graph.add_node("canonicaliser",         canonicaliser_node)
    graph.add_node("cluster_agent",         cluster_node)        # GAP 1 FIX
    graph.add_node("memory_agent",          memory_node)
    graph.add_node("reranker",              reranker_node)
    graph.add_node("human_review_gate",     human_review_gate_node)

    # ── Edges ────────────────────────────────────────────────────────────────

    graph.add_edge(START, "preprocess")

    # Fan-out: preprocess -> all 3 view builders (parallel)
    graph.add_edge("preprocess", "text_views")
    graph.add_edge("preprocess", "image_views")
    graph.add_edge("preprocess", "audio_views")

    # Fan-in: all 3 views -> cross-modal alignment
    graph.add_edge("text_views",  "cross_modal_align")
    graph.add_edge("image_views", "cross_modal_align")
    graph.add_edge("audio_views", "cross_modal_align")

    graph.add_edge("cross_modal_align", "domain_router")

    # Fan-out: domain_router -> both labellers (parallel)
    graph.add_edge("domain_router", "conservative_labeller")
    graph.add_edge("domain_router", "liberal_labeller")

    # Fan-in: both labellers -> merge -> arbitration
    graph.add_edge("conservative_labeller", "merge_labels")
    graph.add_edge("liberal_labeller",      "merge_labels")
    graph.add_edge("merge_labels",          "arbitration")

    # Conditional routing after arbitration
    graph.add_conditional_edges(
        "arbitration",
        _route_after_arbitration,
        {"canonicaliser": "canonicaliser", "end": END}
    )

    # Linear chain — cluster_agent now sits between canonicaliser and memory
    graph.add_edge("canonicaliser",     "cluster_agent")   # GAP 1 FIX
    graph.add_edge("cluster_agent",     "memory_agent")
    graph.add_edge("memory_agent",      "reranker")
    graph.add_edge("reranker",          "human_review_gate")
    graph.add_edge("human_review_gate", END)

    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


def build_graph_with_memory():
    return build_graph(checkpointer=MemorySaver())
