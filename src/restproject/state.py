from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


# ──────────────────────────────────────────────────────────────────────────────
#  LangGraph State
# ──────────────────────────────────────────────────────────────────────────────

class RestaurantState(TypedDict):
    # ── Conversation history (LLM + user turns) ──────────────────────────────
    messages: Annotated[list, add_messages]

    # ── Extracted order details ───────────────────────────────────────────────
    dish_name: str          # Extracted dish name (normalized to menu key) — current item
    required_qty: int       # Quantity the user asked for — current item
    available_qty: int      # Quantity available in the menu — current item

    # ── Multi-item order lists ────────────────────────────────────────────────
    order_items: list       # Raw items from LLM: [{"dish": str, "qty": int}]
    item_queue: list        # Validated items waiting for the kitchen: [{"dish": str, "qty": int}]
    served_items: list      # Successfully served items (for final summary): [{"dish": str, "qty": int}]

    # ── Order tracking ────────────────────────────────────────────────────────
    order_id: str           # e.g. "ORD-4821"; set by create_order_node

    # ── Pipeline status ───────────────────────────────────────────────────────
    # Possible values (in flow order):
    #   "pending"        – initial / waiting for LLM to extract order
    #   "off_topic"      – user asked a non-food question → re-prompt
    #   "menu_valid"     – dish found on menu → proceed to inventory_check
    #   "menu_invalid"   – dish not on menu → ask customer (free re-prompt)
    #   "confirm"        – dish + qty fully available → create_order
    #   "partial"        – dish available but qty insufficient → order_retry
    #   "order_created"  – order ID assigned → cook
    #   "cook_done"      – cook succeeded → serve
    #   "cook_failed"    – cook failed (retries remain) → cook again
    #   "serve_failed"   – serve failed → re-cook
    #   "complete"       – order fully completed and delivered → end
    #   "apology"        – retries exhausted; session ending → end
    status: str

    # ── Retry counters ────────────────────────────────────────────────────────
    order_retries: int   # starts at 3 – decrements on each partial/unavail
    cook_retries: int    # starts at 2 – decrements on each cook failure / re-cook
    serve_retries: int   # starts at 2 – decrements on each serve failure

    # ── Final result ──────────────────────────────────────────────────────────
    final_result: str    # "success" | "failed" | ""
