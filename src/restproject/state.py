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
    dish_name: str          # Extracted dish name
    required_qty: int       # Quantity the user asked for
    available_qty: int      # Quantity available in the menu (0 = not in menu)

    # ── Pipeline status ───────────────────────────────────────────────────────
    # Possible values:
    #   "pending"        – initial / waiting for LLM to extract order
    #   "confirm"        – dish + qty fully available
    #   "partial"        – dish available but qty insufficient
    #   "unavailable"    – dish not on menu / qty = 0
    #   "cooking"        – cook node is running
    #   "cook_done"      – cook succeeded
    #   "cook_failed"    – cook failed this attempt
    #   "serving"        – serve node is running
    #   "serve_done"     – serve succeeded
    #   "serve_failed"   – serve failed this attempt
    #   "complete"       – order fully completed and delivered
    #   "apology"        – something went wrong; session ending
    #   "off_topic"      – user asked a non-food question
    status: str

    # ── Retry counters ────────────────────────────────────────────────────────
    order_retries: int   # starts at 3 – decrements on each partial/unavail
    cook_retries: int    # starts at 2 – decrements on each cook failure
    serve_retries: int   # starts at 2 – decrements on each serve failure

    # ── Final result ──────────────────────────────────────────────────────────
    final_result: str    # "success" | "failed" | ""
