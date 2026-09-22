from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

# ── Load .env and inject GROQ_API_KEY into env immediately ───────────────────
# This must happen before ChatGroq is ever constructed.
try:
    from dotenv import load_dotenv as _load_dotenv
    _here = Path(__file__).resolve()
    for _p in [_here.parent, _here.parent.parent, _here.parent.parent.parent]:
        _ef = _p / ".env"
        if _ef.exists():
            _load_dotenv(dotenv_path=_ef, override=True)
            break
except ImportError:
    pass  # dotenv not installed – rely on shell env

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from .menu import MENU
from .state import RestaurantState

# ──────────────────────────────────────────────────────────────────────────────
#  Lazy LLM getter  (initialised on first use so import doesn't fail without key)
# ──────────────────────────────────────────────────────────────────────────────
_llm_instance: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    return _llm_instance

# ──────────────────────────────────────────────────────────────────────────────
#  System prompt for the LLM node
# ──────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a restaurant order-taking assistant.
Your ONLY job is to help customers place food orders.

Rules:
1. If the user's message is NOT about placing a food order, reply EXACTLY:
   "OFFTOPIC"
   (nothing else)

2. If the message IS about placing an order, extract the dish name and quantity.
   Reply EXACTLY in this JSON format (no markdown, no extra text):
   {"dish": "<dish_name_lowercase>", "qty": <integer>}

3. Do NOT handle greetings, general chat, weather, recipes, or anything else.
"""


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 1 – llm_node
# ──────────────────────────────────────────────────────────────────────────────
def llm_node(state: RestaurantState) -> dict[str, Any]:
    """
    Calls the LLM to either:
      • Extract dish + qty from the latest user message, OR
      • Inform the user it cannot help with off-topic queries.

    Also handles the retry-prompt: if order_retries > 0 and status is
    'partial' or 'unavailable', it presents options to the user and waits
    for the next human input.
    """
    # ── Build message list for the LLM ───────────────────────────────────────
    system = SystemMessage(content=_SYSTEM_PROMPT)

    # The last message in state.messages is the most recent user turn
    last_msg = state["messages"][-1]

    # If we're in a re-prompt cycle, the last message is already the user's
    # decision; pass the whole history for context
    response = _get_llm().invoke([system] + state["messages"])
    raw = response.content.strip()

    # ── Parse LLM response ────────────────────────────────────────────────────
    if raw == "OFFTOPIC":
        ai_msg = AIMessage(
            content=(
                "I'm a restaurant ordering assistant. "
                "I can only help you place a food order. "
                "Please tell me what you'd like to order!"
            )
        )
        return {
            "messages": [ai_msg],
            "status": "off_topic",
        }

    # Try to parse as JSON order
    try:
        import json
        order = json.loads(raw)
        dish = order["dish"].strip().lower()
        qty = int(order["qty"])
        ai_msg = AIMessage(
            content=f"Got it! I've noted your order: {qty}x {dish.title()}. Let me check availability…"
        )
        return {
            "messages": [ai_msg],
            "dish_name": dish,
            "required_qty": qty,
            "status": "pending",
        }
    except Exception:
        # Fallback – treat as off-topic / unclear
        ai_msg = AIMessage(
            content=(
                "I couldn't understand your order. "
                "Please specify the dish name and quantity. "
                "Example: 'I want 2 burgers'"
            )
        )
        return {
            "messages": [ai_msg],
            "status": "off_topic",
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Helper – dish name normalizer (shared by menu_validator_node)
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_dish(raw: str) -> str:
    """
    Normalize dish name to match menu keys.
    Handles plurals ('burgers' → 'burger', 'sandwiches' → 'sandwich') and
    extra whitespace.  Falls back to fuzzy prefix match if exact match fails.
    """
    raw = raw.strip().lower()
    if raw in MENU:
        return raw

    # Try stripping common plural suffixes
    for suffix in ("es", "s"):
        candidate = raw[: -len(suffix)] if raw.endswith(suffix) else raw
        if candidate in MENU:
            return candidate

    # Fuzzy: find any menu key that is a prefix of the input (or vice-versa)
    for key in MENU:
        if raw.startswith(key) or key.startswith(raw):
            return key

    return raw   # return as-is; menu_validator will handle the unavailable case


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 2 – menu_validator_node
#  Checks whether the requested dish exists on the menu.
#  • Found     → status='menu_valid'    → inventory_check_node
#  • Not found → status='menu_invalid'  → END (main loop re-prompts for free)
# ──────────────────────────────────────────────────────────────────────────────
def menu_validator_node(state: RestaurantState) -> dict[str, Any]:
    dish = _normalize_dish(state["dish_name"])
    if dish not in MENU:
        msg = AIMessage(
            content=(
                f"❌ '{state['dish_name'].title()}' is not on our menu.\n"
                f"Available items: {', '.join(d.title() for d in MENU)}.\n"
                "Please order one of the items above."
            )
        )
        return {
            "messages": [msg],
            "dish_name": dish,
            "available_qty": 0,
            "status": "menu_invalid",
        }

    msg = AIMessage(
        content=f"✓ {dish.title()} is on our menu! Checking inventory…"
    )
    return {
        "messages": [msg],
        "dish_name": dish,   # persist normalized name
        "status": "menu_valid",
    }


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 3 – inventory_check_node
#  Verifies that sufficient quantity is available.
#  • Sufficient → status='confirm'   → create_order_node
#  • Partial    → status='partial'   → order_retry_node
# ──────────────────────────────────────────────────────────────────────────────
def inventory_check_node(state: RestaurantState) -> dict[str, Any]:
    dish     = state["dish_name"]
    required = state["required_qty"]
    avail    = MENU[dish]

    if avail >= required:
        msg = AIMessage(
            content=(
                f"✓ Inventory confirmed: {avail} portion(s) of {dish.title()} in stock.\n"
                f"  Your order of {required} is available!"
            )
        )
        return {
            "messages": [msg],
            "available_qty": avail,
            "status": "confirm",
        }
    else:
        msg = AIMessage(
            content=(
                f"⚠️  We only have {avail} portion(s) of {dish.title()} "
                f"(you asked for {required}).\n"
                f"Would you like to:\n"
                f"  1. Proceed with {avail} portion(s) (partial order)\n"
                f"  2. Place a new order for a different dish or quantity\n"
                "Please type your choice."
            )
        )
        return {
            "messages": [msg],
            "available_qty": avail,
            "status": "partial",
        }


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 4 – create_order_node
#  Formalises the order: generates a unique order ID and prints a receipt.
#  Always routes to cook_node (status='order_created').
# ──────────────────────────────────────────────────────────────────────────────
def create_order_node(state: RestaurantState) -> dict[str, Any]:
    order_id = f"ORD-{random.randint(1000, 9999)}"
    dish     = state["dish_name"]
    qty      = state["required_qty"]
    msg = AIMessage(
        content=(
            f"🧾 Order {order_id} created!\n"
            f"   • Item  : {qty}x {dish.title()}\n"
            f"   • Status: Confirmed ✅ — sending to the kitchen now!"
        )
    )
    return {
        "messages": [msg],
        "order_id": order_id,
        "status": "order_created",
    }




# ──────────────────────────────────────────────────────────────────────────────
#  Retry-prompt helper – used when order is partial/unavailable
# ──────────────────────────────────────────────────────────────────────────────
def order_retry_node(state: RestaurantState) -> dict[str, Any]:
    """
    Handles the user's response after a partial/unavailable status.
    Reads the last user message and either:
      • Lets the user proceed with partial qty  →  status='confirm'
      • Sends back to llm_node for a new order  →  status='pending'
      • Ends session when retries are exhausted  →  status='apology'
    """
    retries_left = state["order_retries"] - 1
    last_user_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content.lower()
            break

    if retries_left <= 0:
        apology = AIMessage(
            content=(
                "We're sorry, but we couldn't accommodate your order after multiple attempts. "
                "We apologize for the inconvenience. Please visit us again!"
            )
        )
        return {
            "messages": [apology],
            "order_retries": 0,
            "status": "apology",
            "final_result": "failed",
        }

    # User wants to proceed with partial
    if any(kw in last_user_msg for kw in ["1", "partial", "proceed", "yes", "ok", "sure", "fine"]):
        avail = state["available_qty"]
        msg = AIMessage(
            content=(
                f"Proceeding with {avail} portion(s) of {state['dish_name'].title()}. "
                "Sending to the kitchen!"
            )
        )
        return {
            "messages": [msg],
            "required_qty": avail,   # adjust qty to what's available
            "order_retries": retries_left,
            "status": "confirm",
        }

    # User wants a new order
    prompt_msg = AIMessage(
        content=(
            f"Sure! Please tell me your new order. "
            f"(Attempts remaining: {retries_left})"
        )
    )
    return {
        "messages": [prompt_msg],
        "order_retries": retries_left,
        "status": "pending",
    }


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 3 – cook_node
# ──────────────────────────────────────────────────────────────────────────────
def cook_node(state: RestaurantState) -> dict[str, Any]:
    """
    Simulates cooking – handles BOTH initial cooking and re-cook after serve failure.

    Re-cook rule (TC3): each re-cook call (i.e. when serve has already failed at
    least once, detected by serve_retries < 2) costs one cook_retry REGARDLESS
    of cook success/failure.  This means cook_retries can exhaust even if cook
    succeeds, preventing an infinite serve→re-cook loop.

    • cook_retries == 0 on entry  → immediate apology
    • 40% fail                    → cook_retries -= 1, status='cook_failed'
    • 60% success                 → if re-cook: cook_retries -= 1; status='cook_done'
    """
    retries_left = state["cook_retries"]
    is_recook = state["serve_retries"] < 2   # True after ≥1 serve failure

    if retries_left <= 0:
        apology = AIMessage(
            content=(
                "We sincerely apologize. Our kitchen has exhausted all retry attempts "
                "and cannot prepare your order. Your session has ended."
            )
        )
        return {
            "messages": [apology],
            "status": "apology",
            "final_result": "failed",
        }

    # 40% failure probability
    failed = random.random() < 0.4

    if not failed:
        # Success – if this is a re-cook, still burn one retry slot
        new_retries = retries_left - 1 if is_recook else retries_left
        msg = AIMessage(
            content=(
                f"Your {state['dish_name'].title()} has been prepared successfully! "
                "Now sending it your way…"
            )
        )
        return {
            "messages": [msg],
            "cook_retries": new_retries,
            "status": "cook_done",
        }
    else:
        new_retries = retries_left - 1   # failure always costs a retry
        if new_retries > 0:
            msg = AIMessage(
                content=(
                    f"Oops! The kitchen had a hiccup preparing your {state['dish_name'].title()}. "
                    f"Trying again… (Cook attempts remaining: {new_retries})"
                )
            )
            return {
                "messages": [msg],
                "cook_retries": new_retries,
                "status": "cook_failed",   # still retries left → route back to cook
            }
        else:
            # Retries exhausted → apology immediately so routing + main loop both see it
            apology = AIMessage(
                content=(
                    f"We're so sorry — our kitchen exhausted all attempts preparing your "
                    f"{state['dish_name'].title()}. Your session has ended."
                )
            )
            return {
                "messages": [apology],
                "cook_retries": 0,
                "status": "apology",        # ← was 'cook_failed', caused main loop not to exit
                "final_result": "failed",
            }


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 4 – serve_node
# ──────────────────────────────────────────────────────────────────────────────
def serve_node(state: RestaurantState) -> dict[str, Any]:
    """
    Simulates serving.
    • 60% chance: success  → status='serve_done'  →  final_result='success'
    • 40% chance: failure  → status='serve_failed', serve_retries decremented
    If serve_retries is 0, issue an apology.
    """
    retries_left = state["serve_retries"]

    if retries_left <= 0:
        apology = AIMessage(
            content=(
                "We sincerely apologize. Our staff failed to serve your order correctly "
                "after multiple attempts. Your session has ended."
            )
        )
        return {
            "messages": [apology],
            "status": "apology",
            "final_result": "failed",
        }

    failed = random.random() < 0.4

    if not failed:
        msg = AIMessage(
            content=(
                f"🍽️  Your {state['required_qty']}x {state['dish_name'].title()} "
                "has been served! Enjoy your meal! 😊"
            )
        )
        return {
            "messages": [msg],
            "status": "complete",
            "final_result": "success",
        }
    else:
        new_retries = retries_left - 1
        msg = AIMessage(
            content=(
                f"Apologies! There was an issue serving your {state['dish_name'].title()}. "
                f"Sending back to the kitchen for another attempt… "
                f"(Serve retries remaining: {new_retries})"
            )
        )
        return {
            "messages": [msg],
            "serve_retries": new_retries,
            "status": "serve_failed",
        }


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 5 – end_node
# ──────────────────────────────────────────────────────────────────────────────
def end_node(state: RestaurantState) -> dict[str, Any]:
    """Terminal node – always sets a clean terminal status for the main loop."""
    result = state.get("final_result", "")
    if result == "success":
        msg = AIMessage(
            content=(
                "✅ Order Complete! Thank you for dining with us. "
                "We hope to see you again soon!"
            )
        )
        return {"messages": [msg], "status": "complete", "final_result": "success"}
    else:
        msg = AIMessage(
            content=(
                "❌ Session Ended. We apologize for the trouble. "
                "Please try again later or contact our support."
            )
        )
        # Ensure status=apology + final_result=failed so the main loop always exits
        return {"messages": [msg], "status": "apology", "final_result": "failed"}
    return {"messages": [msg]}
