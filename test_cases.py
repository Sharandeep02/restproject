#!/usr/bin/env python3
"""
Deterministic Test Runner for Restaurant Order Management Agent
==============================================================
Tests TC1, TC2, TC3 from prompt.md by mocking the LLM and random.random()
so every step is predictable regardless of API availability.

Run with:
    uv run python test_cases.py
"""
from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, patch

# ── Make the package importable ───────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from langchain_core.messages import HumanMessage, AIMessage

from restproject.graph import graph
from restproject.state import RestaurantState


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_state() -> RestaurantState:
    return {
        "messages":       [],
        "dish_name":      "",
        "required_qty":   0,
        "available_qty":  0,
        "order_items":    [],
        "item_queue":     [],
        "served_items":   [],
        "order_id":       "",
        "status":         "pending",
        "order_retries":  3,
        "cook_retries":   2,
        "serve_retries":  2,
        "final_result":   "",
    }


def _last_ai_message(state: RestaurantState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            return m.content
    return ""


def _run_session(
    user_inputs:    list[str],
    llm_responses:  list[str],
    random_values:  list[float],
    label:          str = "",
) -> RestaurantState:
    """
    Simulates the main() loop deterministically.

    • llm_responses  : pre-defined LLM replies consumed in order
    • random_values  : values returned by random.random() in order
    • Each user_input triggers one graph.invoke(); state persists between calls.
    """
    state = _make_state()

    # Build a mock LLM whose .invoke() returns responses in sequence
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        MagicMock(content=r) for r in llm_responses
    ]

    input_iter  = iter(user_inputs)
    random_iter = iter(random_values)
    step        = 0

    print(f"\n{'─'*62}")
    print(f"  SESSION START  {label}")
    print(f"{'─'*62}")

    with patch("restproject.nodes._get_llm", return_value=mock_llm), \
         patch("restproject.nodes.random") as mock_rand:

        mock_rand.random.side_effect = list(random_values)

        session_active = True
        while session_active:
            # ── Get next user input ───────────────────────────────────────
            try:
                user_input = next(input_iter)
            except StopIteration:
                print("  [no more inputs – ending session]")
                break

            step += 1
            print(f"\n  Step {step}: User → \"{user_input}\"")

            # Append user message and invoke the graph
            state["messages"] = list(state["messages"]) + [
                HumanMessage(content=user_input)
            ]
            state = graph.invoke(state)

            ai_reply = _last_ai_message(state)
            print(f"           Bot  → \"{ai_reply[:90]}{'…' if len(ai_reply)>90 else ''}\"")
            print(f"           [status={state['status']}  "
                  f"order_retries={state['order_retries']}  "
                  f"cook_retries={state['cook_retries']}  "
                  f"serve_retries={state['serve_retries']}  "
                  f"final_result='{state['final_result']}']")

            current_status = state.get("status", "")

            # ── Main-loop routing (mirrors __init__.py main()) ────────────
            if current_status in ("off_topic", "menu_invalid"):
                # Free re-prompt: reset for new order attempt
                print(f"  [{ current_status }: resetting for new order attempt]")
                state["status"]   = "pending"
                state["messages"] = []

            elif current_status in ("complete", "apology"):
                # Terminal: session ends
                session_active = False

            # else status == "pending"  →  loop for next user input ✓

    print(f"\n  SESSION END — final_result='{state.get('final_result','')}'\n")
    return state


# ─────────────────────────────────────────────────────────────────────────────
#  TEST CASE 1
#  Flow: off-topic → partial order → reject+unavailable → unavailable
#  Expected: order retries exhausted → final_result='failed'
# ─────────────────────────────────────────────────────────────────────────────

def test_tc1() -> bool:
    print("\n" + "═"*62)
    print("  TEST CASE 1 (updated for new architecture)")
    print("  i.  off-topic question → free re-prompt")
    print("  ii. unknown dish (sushi) → menu_invalid → free re-prompt")
    print("  iii.partial order (burger, need 20, only 10) → retry cost")
    print("  iv. partial again → retry cost")
    print("  v.  partial again → retries=0 → apology")
    print("  Expected: order retries exhausted → FAIL")
    print("═"*62)

    # In the new architecture:
    #   off_topic    → free re-prompt (order_retries unchanged)
    #   menu_invalid → free re-prompt (order_retries unchanged)
    #   partial      → burns one order_retry slot
    # So we exhaust retries via 3 partial rejections.

    user_inputs = [
        "What is the weather today?",  # i.  off-topic → free
        "I want sushi",                # ii. menu_invalid → free
        "I want 20 burgers",           # iii.partial (10 avail) → retries 3→2
        "No, I want 20 burgers",       # iv. partial again     → retries 2→1
        "Still want 20 burgers",       # v.  partial again     → retries 1→0 → apology
    ]

    llm_responses = [
        "OFFTOPIC",                          # i.
        '{"dish": "sushi",  "qty": 1}',      # ii.  → menu_invalid
        '{"dish": "burger", "qty": 20}',     # iii. → partial
        '{"dish": "burger", "qty": 20}',     # iv.  → partial
        '{"dish": "burger", "qty": 20}',     # v.   → apology
    ]

    state = _run_session(user_inputs, llm_responses, [], "TC1")

    passed = state.get("final_result") == "failed"
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}  "
          f"(expected final_result='failed', got '{state.get('final_result')}')")
    return passed



# ─────────────────────────────────────────────────────────────────────────────
#  TEST CASE 2
#  Flow: order available → cook fail → cook retry success →
#        serve fail → re-cook success → serve success
#  Expected: final_result='success'
# ─────────────────────────────────────────────────────────────────────────────

def test_tc2() -> bool:
    print("\n" + "═"*62)
    print("  TEST CASE 2")
    print("  - Order available (2 burgers, 10 in stock)")
    print("  - Cook fails (40%) → retries → succeeds")
    print("  - Serve fails → sends back to kitchen → re-cook succeeds")
    print("  - Serve succeeds")
    print("  Expected: overall SUCCESS")
    print("═"*62)

    user_inputs   = ["I want 2 burgers"]
    llm_responses = ['{"dish": "burger", "qty": 2}']

    #  random.random() calls in order:
    #  0.30 → cook attempt 1:  FAIL   (< 0.4)   cook_retries: 2→1
    #  0.70 → cook attempt 2:  SUCCESS (> 0.4)
    #  0.30 → serve attempt 1: FAIL   (< 0.4)   serve_retries: 2→1  → re-cook
    #  0.70 → re-cook 1:       SUCCESS (> 0.4)  cook_retries: 1→0 (is_recook)
    #  0.70 → serve attempt 2: SUCCESS (> 0.4)
    random_values = [0.30, 0.70, 0.30, 0.70, 0.70]

    state = _run_session(user_inputs, llm_responses, random_values, "TC2")

    passed = state.get("final_result") == "success"
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}  "
          f"(expected final_result='success', got '{state.get('final_result')}')")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
#  TEST CASE 3
#  Flow: partial → reject → full order → cook fail/retry/success →
#        serve fail → re-cook success (cook_retry 2→1) →
#        serve fail → re-cook success (cook_retry 1→0) →
#        serve fail → cook_retries=0 → apology
#  Expected: cook retries exhausted → final_result='failed'
# ─────────────────────────────────────────────────────────────────────────────

def test_tc3() -> bool:
    print("\n" + "═"*62)
    print("  TEST CASE 3")
    print("  - Partial order (20 burgers, only 10) → reject")
    print("  - New order: 3 pizzas (5 in stock) → confirmed")
    print("  - Cook fails → retries → succeeds")
    print("  - Serve fails → re-cook 1 succeeds (cook_retries: 2→1)")
    print("  - Serve fails → re-cook 2 succeeds (cook_retries: 1→0)")
    print("  - Serve fails → cook_retries=0 → apology (cook exhausted)")
    print("  Expected: cook retries exhausted → FAIL")
    print("═"*62)

    user_inputs = [
        "I want 20 burgers",     # partial (10 avail) → order_retry → pending
        "No, I want 3 pizzas",   # new order (pizza: 5 avail, need 3 → confirm)
    ]

    llm_responses = [
        '{"dish": "burger","qty": 20}',
        '{"dish": "pizza", "qty": 3}',
    ]

    #  random.random() calls in order:
    #  0.30 → cook 1 (initial):  FAIL   cook_retries: 2→1
    #  0.70 → cook 2 (retry):    SUCCESS
    #  0.30 → serve 1:           FAIL   serve_retries: 2→1  → re-cook (is_recook=True)
    #  0.70 → re-cook 1:         SUCCESS cook_retries: 1→0 (burned by is_recook)
    #  0.30 → serve 2:           FAIL   serve_retries: 1→0  → re-cook
    #       → cook node: cook_retries=0 → apology (no random call)
    random_values = [0.30, 0.70, 0.30, 0.70, 0.30]

    state = _run_session(user_inputs, llm_responses, random_values, "TC3")

    passed = state.get("final_result") == "failed"
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}  "
          f"(expected final_result='failed', got '{state.get('final_result')}')")
    return passed



# ─────────────────────────────────────────────────────────────────────────────
#  TEST CASE 4
#  Multi-item order: 2 burgers + 1 soda, both succeed on first cook+serve.
#  Expected: overall SUCCESS, served_items has both items.
# ─────────────────────────────────────────────────────────────────────────────

def test_tc4() -> bool:
    print("\n" + "═"*62)
    print("  TEST CASE 4  — Multi-item order")
    print("  User: 'I want 2 burgers and a soda'")
    print("  LLM extracts: [{burger,2}, {soda,1}]")
    print("  Cook burger: success; Serve burger: success → pop soda")
    print("  Cook soda:   success; Serve soda:   success → complete")
    print("  Expected: final_result='success', 2 items in served_items")
    print("═"*62)

    user_inputs   = ["I want 2 burgers and a soda"]
    llm_responses = ['[{"dish": "burger", "qty": 2}, {"dish": "soda", "qty": 1}]']

    # All random calls succeed (0.9 > 0.4 threshold)
    # cook burger (1 call) + serve burger (1 call) + cook soda (1 call) + serve soda (1 call)
    random_values = [0.9, 0.9, 0.9, 0.9]

    state = _run_session(user_inputs, llm_responses, random_values, "TC4")

    served   = state.get("served_items", [])
    passed   = (
        state.get("final_result") == "success"
        and len(served) == 2
        and any(it["dish"] == "burger" for it in served)
        and any(it["dish"] == "soda"   for it in served)
    )
    print(f"  served_items: {served}")
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}  "
          f"(expected final_result='success' + 2 served items, "
          f"got '{state.get('final_result')}' + {len(served)} items)")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
#  Main  (re-registered to include TC4)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = [
        test_tc1(),
        test_tc2(),
        test_tc3(),
        test_tc4(),
    ]

    total  = len(results)
    passed = sum(results)

    print("\n" + "═"*62)
    print(f"  FINAL RESULTS: {passed}/{total} tests PASSED")
    print("═"*62)
    for i, r in enumerate(results, 1):
        print(f"  TC{i}: {'✅ PASS' if r else '❌ FAIL'}")
    print("═"*62 + "\n")

    sys.exit(0 if all(results) else 1)
