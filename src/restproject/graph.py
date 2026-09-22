from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    cook_node,
    end_node,
    llm_node,
    order_confirm_node,
    order_retry_node,
    serve_node,
)
from .state import RestaurantState

# ──────────────────────────────────────────────────────────────────────────────
#  Conditional edge functions
# ──────────────────────────────────────────────────────────────────────────────

def route_after_llm(state: RestaurantState) -> str:
    """After the LLM extracts an order (or flags off-topic), decide next step."""
    status = state["status"]
    if status == "off_topic":
        return "stop"          # off-topic → stop graph, main loop re-prompts
    if status == "pending":
        return "order_confirm"  # valid order → confirm against menu
    return "stop"


def route_after_confirm(state: RestaurantState) -> str:
    """After order_confirm, decide: cook, retry, or end."""
    status = state["status"]
    if status == "confirm":
        return "cook"
    if status in ("partial", "unavailable"):
        return "order_retry"
    return "end"


def route_after_retry(state: RestaurantState) -> str:
    """After order_retry, decide: cook (partial accepted), stop for new input, or end."""
    status = state["status"]
    if status == "confirm":
        return "cook"       # user accepted partial qty → proceed to kitchen
    if status == "pending":
        return "stop"       # needs new user input → stop graph, main loop handles
    if status == "apology":
        return "end"
    return "end"


def route_after_cook(state: RestaurantState) -> str:
    """After cook_node, decide: serve, retry cook, or end."""
    status = state["status"]
    if status == "cook_done":
        return "serve"
    if status == "cook_failed":
        retries = state["cook_retries"]
        if retries > 0:
            return "cook"   # retry cooking
        else:
            return "end"
    if status == "apology":
        return "end"
    return "end"


def route_after_serve(state: RestaurantState) -> str:
    """After serve_node: success → end, fail → back to cook (re-cook), apology → end."""
    status = state["status"]
    if status == "complete":
        return "end"
    if status == "serve_failed":
        # Serve failure sends order back to kitchen (cook retries)
        return "cook"
    if status == "apology":
        return "end"
    return "end"


# ──────────────────────────────────────────────────────────────────────────────
#  Build the graph
# ──────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(RestaurantState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("llm", llm_node)
    builder.add_node("order_confirm", order_confirm_node)
    builder.add_node("order_retry", order_retry_node)
    builder.add_node("cook", cook_node)
    builder.add_node("serve", serve_node)
    builder.add_node("end", end_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.add_edge(START, "llm")

    # ── Conditional edges ─────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "llm",
        route_after_llm,
        {
            "order_confirm": "order_confirm",
            "stop": END,           # off-topic or unrecognised → stop graph
        },
    )

    builder.add_conditional_edges(
        "order_confirm",
        route_after_confirm,
        {
            "cook": "cook",
            "order_retry": "order_retry",
            "end": "end",
        },
    )

    builder.add_conditional_edges(
        "order_retry",
        route_after_retry,
        {
            "cook": "cook",
            "stop": END,           # pending new user input → stop, main loop prompts
            "end": "end",
        },
    )

    builder.add_conditional_edges(
        "cook",
        route_after_cook,
        {
            "serve": "serve",
            "cook": "cook",
            "end": "end",
        },
    )

    builder.add_conditional_edges(
        "serve",
        route_after_serve,
        {
            "end": "end",
            "cook": "cook",        # serve fail → back to kitchen for re-cook
        },
    )

    # ── Terminal edge ─────────────────────────────────────────────────────────
    builder.add_edge("end", END)

    return builder.compile()


# Singleton graph (compiled once at import time)
graph = build_graph()
