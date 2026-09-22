from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    cook_node,
    create_order_node,
    end_node,
    inventory_check_node,
    llm_node,
    menu_validator_node,
    order_retry_node,
    serve_node,
)
from .state import RestaurantState

# ──────────────────────────────────────────────────────────────────────────────
#  Conditional edge (routing) functions
#
#  New architecture (matches the diagram):
#
#  START → llm → menu_validator
#                  ├─(menu_invalid)─→ END        [free re-prompt, no retry cost]
#                  └─(menu_valid)──→ inventory_check
#                                       ├─(partial)──→ order_retry
#                                       │                ├─(confirm)─→ create_order
#                                       │                ├─(pending)─→ END
#                                       │                └─(apology)─→ end
#                                       └─(confirm)─→ create_order
#                                                        └──────────→ cook
#                                                                       ├─(cook_done)───→ serve
#                                                                       ├─(cook_failed)─→ cook
#                                                                       └─(apology)─────→ end
#                                                              serve
#                                                                ├─(complete)────→ end
#                                                                ├─(serve_failed)→ cook  [re-cook]
#                                                                └─(apology)─────→ end
# ──────────────────────────────────────────────────────────────────────────────


def route_after_llm(state: RestaurantState) -> str:
    """LLM parsed the input — route to menu_validator or stop (off-topic)."""
    status = state["status"]
    if status == "pending":
        return "menu_validator"   # valid JSON order → check menu
    return "stop"                 # off_topic or unparseable → main loop re-prompts


def route_after_menu_validator(state: RestaurantState) -> str:
    """Dish found → inventory_check; not found → END (free re-prompt)."""
    status = state["status"]
    if status == "menu_valid":
        return "inventory_check"
    return "stop"                 # menu_invalid → stop graph, main loop re-prompts


def route_after_inventory_check(state: RestaurantState) -> str:
    """Sufficient stock → create_order; insufficient → order_retry."""
    status = state["status"]
    if status == "confirm":
        return "create_order"
    if status == "partial":
        return "order_retry"
    return "end"


def route_after_retry(state: RestaurantState) -> str:
    """User responded to partial/unavailable offer."""
    status = state["status"]
    if status == "confirm":
        return "create_order"   # user accepted partial → create order
    if status == "pending":
        return "stop"           # user wants new order → stop, main loop asks again
    if status == "apology":
        return "end"
    return "end"


def route_after_cook(state: RestaurantState) -> str:
    """Cook succeeded → serve; failed with retries → cook again; exhausted → end."""
    status = state["status"]
    if status == "cook_done":
        return "serve"
    if status == "cook_failed":
        return "cook" if state["cook_retries"] > 0 else "end"
    if status == "apology":
        return "end"
    return "end"


def route_after_serve(state: RestaurantState) -> str:
    """Served → end; more items → cook next; serve failed → re-cook; apology → end."""
    status = state["status"]
    if status == "complete":
        return "end"
    if status == "next_item":
        return "cook"    # advance to next item in the queue
    if status == "serve_failed":
        return "cook"    # re-cook same item
    if status == "apology":
        return "end"
    return "end"


# ──────────────────────────────────────────────────────────────────────────────
#  Build the graph
# ──────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(RestaurantState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("llm",              llm_node)
    builder.add_node("menu_validator",   menu_validator_node)
    builder.add_node("inventory_check",  inventory_check_node)
    builder.add_node("create_order",     create_order_node)
    builder.add_node("order_retry",      order_retry_node)
    builder.add_node("cook",             cook_node)
    builder.add_node("serve",            serve_node)
    builder.add_node("end",              end_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.add_edge(START, "llm")

    # ── Conditional edges ─────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "llm",
        route_after_llm,
        {"menu_validator": "menu_validator", "stop": END},
    )

    builder.add_conditional_edges(
        "menu_validator",
        route_after_menu_validator,
        {"inventory_check": "inventory_check", "stop": END},
    )

    builder.add_conditional_edges(
        "inventory_check",
        route_after_inventory_check,
        {"create_order": "create_order", "order_retry": "order_retry", "end": "end"},
    )

    # create_order always feeds straight into cook (no branching)
    builder.add_edge("create_order", "cook")

    builder.add_conditional_edges(
        "order_retry",
        route_after_retry,
        {"create_order": "create_order", "stop": END, "end": "end"},
    )

    builder.add_conditional_edges(
        "cook",
        route_after_cook,
        {"serve": "serve", "cook": "cook", "end": "end"},
    )

    builder.add_conditional_edges(
        "serve",
        route_after_serve,
        {"end": "end", "cook": "cook"},
    )

    # ── Terminal edge ─────────────────────────────────────────────────────────
    builder.add_edge("end", END)

    return builder.compile()


# Singleton graph (compiled once at import time)
graph = build_graph()
