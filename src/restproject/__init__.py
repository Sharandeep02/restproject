"""
Restaurant Order Management Agent – Entry Point
================================================
Run with:
    uv run restproject
or:
    python -m restproject
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# ── Load .env from project root (walk up from this file) ─────────────────────
_here = Path(__file__).resolve()
for _parent in [_here.parent, _here.parent.parent, _here.parent.parent.parent]:
    _env_file = _parent / ".env"
    if _env_file.exists():
        load_dotenv(dotenv_path=_env_file, override=True)
        break

from .graph import graph
from .state import RestaurantState


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    print("=" * 60)
    print("  🍽️   Restaurant Order Management AI Agent")
    print("=" * 60)
    print("Type your food order (e.g. 'I want 3 burgers')")
    print("Type 'quit' or 'exit' to leave.\n")


def _step_label(i: int, total: int, content: str) -> str:
    """Pick an emoji label based on message content keywords."""
    c = content.lower()
    if "offtopic" in c or "can only help" in c or "couldn't understand" in c:
        return "🤖"
    if "check availability" in c or "noted your order" in c:
        return "📋 Order Received"
    if "not on our menu" in c or "is not on our menu" in c:
        return "❌ Menu Check"
    if "✓" in c and "menu" in c:
        return "📋 Menu ✓"
    if "only have" in c or "please tell me" in c or "would you like" in c:
        return "⚠️  Partial Stock"
    if "all items served" in c or "🎉" in c:
        return "🎉 All Served"
    if "inventory check" in c or "📦" in c:
        return "📦 Inventory ✓"
    if "order" in c and "created" in c:
        return "🧾 Order Created"
    if "has been served" in c or "enjoy your meal" in c:
        return "🛎️  Served"
    if "order complete" in c:
        return "🎉 Complete"
    if "served" in c or "enjoy your meal" in c:
        return "🛎️  Served"
    if "order complete" in c:
        return "🎉 Complete"
    if "apologize" in c or "session has ended" in c or "session ended" in c:
        return "❌ Failed"
    if "sending back to the kitchen" in c:
        return "🔄 Re-cooking"
    return "🤖"


def _print_order_status(state: RestaurantState, prev_msg_count: int) -> None:
    """Print every NEW AI message added since the last graph.invoke()."""
    from langchain_core.messages import AIMessage
    all_msgs = state["messages"]
    new_msgs  = all_msgs[prev_msg_count:]   # only messages added this invocation

    ai_new = [m for m in new_msgs if isinstance(m, AIMessage)]
    if not ai_new:
        return

    print()
    for i, msg in enumerate(ai_new):
        label = _step_label(i, len(ai_new), msg.content)
        # Print each step on its own clearly-labelled line
        print(f"  [{label}]")
        print(f"  {msg.content}")
        if i < len(ai_new) - 1:
            print(f"  {'·' * 56}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
#  Main interactive loop
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _print_banner()

    # ── Initial state ─────────────────────────────────────────────────────────
    state: RestaurantState = {
        "messages": [],
        "dish_name": "",
        "required_qty": 0,
        "available_qty": 0,
        "order_items": [],
        "item_queue": [],
        "served_items": [],
        "order_id": "",
        "status": "pending",
        "order_retries": 3,
        "cook_retries": 2,
        "serve_retries": 2,
        "final_result": "",
    }

    # ── Session loop ──────────────────────────────────────────────────────────
    session_active = True
    while session_active:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break

        if not user_input:
            continue

        # Append the user's message to state
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Record how many messages exist before invoke (to detect new ones after)
        prev_msg_count = len(state["messages"])

        # Invoke the graph
        state = graph.invoke(state)   # type: ignore[assignment]

        # Print all NEW AI messages added during this invocation (order status trail)
        _print_order_status(state, prev_msg_count)

        # ── Check terminal conditions ─────────────────────────────────────────
        terminal_statuses = {"complete", "apology", "off_topic", "menu_invalid"}
        current_status = state.get("status", "")

        if current_status in terminal_statuses:
            if current_status in ("off_topic", "menu_invalid"):
                # Soft stop — re-prompt without consuming an order_retry slot
                print("[System] Please place a food order from our menu, or type 'exit' to quit.\n")
                state["status"] = "pending"
                state["messages"] = []
            else:
                session_active = False

    print("\nThank you for using the Restaurant Agent! 🍽️")


if __name__ == "__main__":
    main()
