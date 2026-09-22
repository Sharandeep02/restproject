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

2. If the message IS about placing an order, extract ALL dish names and
   quantities (there may be more than one item).
   Reply EXACTLY as a JSON array — even for a single item:
   [{"dish": "<dish_name_lowercase>", "qty": <integer>}, ...]

   Examples:
     "I want 2 burgers and a soda"
       → [{"dish": "burger", "qty": 2}, {"dish": "soda", "qty": 1}]
     "3 pizzas please"
       → [{"dish": "pizza", "qty": 3}]

3. Always use singular lowercase dish names (burger not burgers).
4. Do NOT handle greetings, general chat, weather, recipes, or anything else.
"""


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 1 – llm_node
# ──────────────────────────────────────────────────────────────────────────────
def llm_node(state: RestaurantState) -> dict[str, Any]:
    """
    Calls the LLM to extract a list of {dish, qty} items from the user message,
    or flags OFFTOPIC.  Stores the full list in order_items; sets dish_name /
    required_qty to the first item so the subsequent nodes can process them
    one-at-a-time through the pipeline.
    """
    import json

    system  = SystemMessage(content=_SYSTEM_PROMPT)
    response = _get_llm().invoke([system] + state["messages"])
    raw = response.content.strip()

    # ── Off-topic ─────────────────────────────────────────────────────────────
    if raw == "OFFTOPIC":
        ai_msg = AIMessage(
            content=(
                "I'm a restaurant ordering assistant. "
                "I can only help you place a food order. "
                "Please tell me what you'd like to order!"
            )
        )
        return {"messages": [ai_msg], "status": "off_topic"}

    # ── Parse JSON array ──────────────────────────────────────────────────────
    try:
        parsed = json.loads(raw)

        # Accept both array and legacy single-object from older model outputs
        if isinstance(parsed, dict):
            parsed = [parsed]

        items = [
            {"dish": str(it["dish"]).strip().lower(), "qty": int(it["qty"])}
            for it in parsed
            if it.get("dish") and int(it.get("qty", 0)) > 0
        ]
        if not items:
            raise ValueError("empty item list")

        # ── Confirmation message listing all requested items ──────────────────
        item_lines = ", ".join(f"{it['qty']}x {it['dish'].title()}" for it in items)
        ai_msg = AIMessage(
            content=(
                f"Got it! I've noted your order: {item_lines}. "
                "Let me check availability…"
            )
        )
        # Use the first item as the "current" item for the pipeline
        first = items[0]
        return {
            "messages":    [ai_msg],
            "order_items": items,
            "dish_name":   first["dish"],
            "required_qty": first["qty"],
            "status":      "pending",
        }

    except Exception:
        ai_msg = AIMessage(
            content=(
                "I couldn't understand your order. "
                "Please specify dish name(s) and quantity. "
                "Example: 'I want 2 burgers and a soda'"
            )
        )
        return {"messages": [ai_msg], "status": "off_topic"}



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
#  Validates ALL items in order_items in one pass.
#  • All valid   → status='menu_valid'    → inventory_check_node
#  • Some valid  → skips unknowns, continues with valid subset
#  • All invalid → status='menu_invalid'  → END (main loop re-prompts for free)
# ──────────────────────────────────────────────────────────────────────────────
def menu_validator_node(state: RestaurantState) -> dict[str, Any]:
    items   = state.get("order_items") or [{"dish": state["dish_name"], "qty": state["required_qty"]}]
    valid   = []
    invalid = []

    for item in items:
        normalized = _normalize_dish(item["dish"])
        if normalized in MENU:
            valid.append({"dish": normalized, "qty": item["qty"]})
        else:
            invalid.append(item["dish"].title())

    # Build informative message
    lines = []
    if invalid:
        lines.append(f"⚠️  Skipping items not on our menu: {', '.join(invalid)}.")
    if valid:
        valid_names = ", ".join(f"{it['qty']}x {it['dish'].title()}" for it in valid)
        lines.append(f"✓ Menu check passed for: {valid_names}. Checking inventory…")

    if not valid:
        lines.append(
            f"Available items: {', '.join(d.title() for d in MENU)}.\n"
            "Please order from the menu above."
        )
        msg = AIMessage(content="\n".join(lines))
        return {"messages": [msg], "order_items": [], "status": "menu_invalid"}

    msg = AIMessage(content="\n".join(lines))
    first = valid[0]
    return {
        "messages":    [msg],
        "order_items": valid,
        "dish_name":   first["dish"],
        "required_qty": first["qty"],
        "status":      "menu_valid",
    }


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 3 – inventory_check_node
#  Checks ALL order_items for availability.
#
#  Single-item order:
#    • Sufficient → status='confirm'  → create_order
#    • Partial    → status='partial'  → order_retry  (user negotiates)
#
#  Multi-item order (>1 item):
#    • Auto-adjusts each item to available qty; skips zero-stock items.
#    • If at least one item survives → status='confirm' → create_order
#    • If all items are zero stock  → status='partial'  → order_retry
# ──────────────────────────────────────────────────────────────────────────────
def inventory_check_node(state: RestaurantState) -> dict[str, Any]:
    items = state.get("order_items") or [{"dish": state["dish_name"], "qty": state["required_qty"]}]

    # ── Single-item: keep original retry-negotiation behaviour ────────────────
    if len(items) == 1:
        dish     = items[0]["dish"]
        required = items[0]["qty"]
        avail    = MENU[dish]
        if avail >= required:
            msg = AIMessage(
                content=(
                    f"✓ Inventory confirmed: {avail} portion(s) of {dish.title()} in stock.\n"
                    f"  Your order of {required} is available!"
                )
            )
            return {"messages": [msg], "available_qty": avail, "status": "confirm"}
        else:
            msg = AIMessage(
                content=(
                    f"⚠️  We only have {avail} portion(s) of {dish.title()} "
                    f"(you asked for {required}).\n"
                    "Would you like to:\n"
                    f"  1. Proceed with {avail} portion(s) (partial order)\n"
                    "  2. Place a new order for a different dish or quantity\n"
                    "Please type your choice."
                )
            )
            return {"messages": [msg], "available_qty": avail, "status": "partial"}

    # ── Multi-item: auto-adjust and report ────────────────────────────────────
    confirmed  = []
    notes      = []
    for item in items:
        dish     = item["dish"]
        required = item["qty"]
        avail    = MENU[dish]
        if avail <= 0:
            notes.append(f"  ❌ {dish.title()}: out of stock — skipped")
        elif avail < required:
            notes.append(
                f"  ⚠️  {dish.title()}: only {avail} available (adjusted from {required})"
            )
            confirmed.append({"dish": dish, "qty": avail})
        else:
            notes.append(f"  ✓ {dish.title()}: {required}x in stock")
            confirmed.append({"dish": dish, "qty": required})

    if not confirmed:
        msg = AIMessage(
            content="⚠️  None of your items are currently in stock.\n" + "\n".join(notes)
        )
        return {"messages": [msg], "available_qty": 0, "status": "partial"}

    summary = "📦 Inventory check:\n" + "\n".join(notes)
    msg = AIMessage(content=summary)
    first = confirmed[0]
    return {
        "messages":    [msg],
        "order_items": confirmed,
        "dish_name":   first["dish"],
        "required_qty": first["qty"],
        "available_qty": MENU[first["dish"]],
        "status":      "confirm",
    }


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 4 – create_order_node
#  Formalises the order: generates a unique order ID, prints a full receipt,
#  and populates item_queue with remaining items (all except the first, which
#  is sent to the kitchen immediately as dish_name/required_qty).
# ──────────────────────────────────────────────────────────────────────────────
def create_order_node(state: RestaurantState) -> dict[str, Any]:
    order_id = f"ORD-{random.randint(1000, 9999)}"
    items    = state.get("order_items") or [{"dish": state["dish_name"], "qty": state["required_qty"]}]

    # Build itemised receipt
    item_lines = "\n".join(f"   • {it['qty']}x {it['dish'].title()}" for it in items)
    msg = AIMessage(
        content=(
            f"🧾 Order {order_id} created!\n"
            f"{item_lines}\n"
            f"   Status: Confirmed ✅ — sending to the kitchen now!"
        )
    )

    # First item goes to cook immediately; rest queued
    first      = items[0]
    item_queue = items[1:]   # remaining items

    return {
        "messages":     [msg],
        "order_id":     order_id,
        "order_items":  items,
        "item_queue":   item_queue,
        "served_items": [],        # reset for this order
        "dish_name":    first["dish"],
        "required_qty": first["qty"],
        "cook_retries":  2,        # fresh budget for first item
        "serve_retries": 2,
        "status":       "order_created",
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
    • 60% chance: success → check item_queue:
        - more items → pop next, reset retries, status='next_item' → cook
        - no more    → build summary, status='complete' → end
    • 40% chance: failure → status='serve_failed' → re-cook
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
        return {"messages": [apology], "status": "apology", "final_result": "failed"}

    failed = random.random() < 0.4

    if not failed:
        # ── Record this item as served ────────────────────────────────────────
        served = list(state.get("served_items") or [])
        served.append({"dish": state["dish_name"], "qty": state["required_qty"]})

        success_line = (
            f"🍽️  Your {state['required_qty']}x {state['dish_name'].title()} "
            "has been served! Enjoy your meal! 😊"
        )

        # ── Check if more items are queued ────────────────────────────────────
        queue = list(state.get("item_queue") or [])
        if queue:
            next_item = queue.pop(0)
            msg = AIMessage(content=success_line)
            return {
                "messages":     [msg],
                "served_items": served,
                "item_queue":   queue,
                "dish_name":    next_item["dish"],
                "required_qty": next_item["qty"],
                "cook_retries":  2,    # fresh budget for next item
                "serve_retries": 2,
                "status":       "next_item",   # routes back to cook
            }
        else:
            # ── All items served — build full summary ─────────────────────────
            summary_lines = "\n".join(
                f"  ✅ {it['qty']}x {it['dish'].title()}" for it in served
            )
            msg = AIMessage(
                content=(
                    f"{success_line}\n\n"
                    f"🎉 All items served!\n{summary_lines}"
                )
            )
            return {
                "messages":     [msg],
                "served_items": served,
                "status":       "complete",
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
        return {"messages": [msg], "serve_retries": new_retries, "status": "serve_failed"}



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
