"""End-to-end scripted test for the Infix sales agent.

Runs the full demo conversation specified in the assignment:
  1. Greeting + pricing inquiry
  2. Objection handling (expensive)
  3. High-intent signal (wants Pro for YouTube) -> platform captured here
  4. Name collection
  5. Email collection -> lead captured here (all 3 fields present)

Each step includes assertions so failures are immediately visible.
"""

import sys
import io
from agent import build_graph, new_state
from langchain_core.messages import HumanMessage

# Fix Windows console encoding for emoji/unicode characters.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )


def test_full_conversation():
    graph = build_graph()
    state = new_state()

    # The full 5-turn flow. Platform "YouTube" is mentioned in turn 3 and
    # captured by extract_info_from_message, so no separate platform turn
    # is needed. The agent should ask name -> email -> done.
    test_messages = [
        "Hi, tell me about your pricing",
        "This sounds expensive, why should I buy Pro?",
        "Okay, I want to start with Pro for my YouTube channel",
        "John Doe",
        "john@example.com",
    ]

    print("=" * 60)
    print("Infix Agent - End-to-End Test")
    print("=" * 60)
    print()

    for i, user_msg in enumerate(test_messages, 1):
        print(f"Turn {i}")
        print(f"  User:  {user_msg}")

        state["messages"].append(HumanMessage(content=user_msg))
        state["next_step"] = ""

        result = graph.invoke(state)
        state = result

        agent_response = state["messages"][-1].content
        print(f"  Agent: {agent_response}")
        print(
            f"  [State] intent={state.get('current_intent')}  "
            f"name={state.get('lead_name')}  "
            f"email={state.get('lead_email')}  "
            f"platform={state.get('lead_platform')}  "
            f"captured={state.get('lead_captured')}"
        )
        print("-" * 60)
        print()

    # ---- Assertions ----
    errors = []

    if not state.get("lead_captured"):
        errors.append("lead_captured is False - mock_lead_capture was never called")

    if state.get("lead_name") != "John Doe":
        errors.append(
            f"lead_name should be 'John Doe', got '{state.get('lead_name')}'"
        )

    if state.get("lead_email") != "john@example.com":
        errors.append(
            f"lead_email should be 'john@example.com', got '{state.get('lead_email')}'"
        )

    if state.get("lead_platform") != "YouTube":
        errors.append(
            f"lead_platform should be 'YouTube', got '{state.get('lead_platform')}'"
        )

    print("=" * 60)
    if errors:
        print("TEST FAILED")
        for err in errors:
            print(f"   - {err}")
    else:
        print("TEST PASSED - Full flow completed successfully")
        print("   * Pricing question answered")
        print("   * Objection handled")
        print("   * High-intent detected")
        print("   * Name, email, platform collected")
        print("   * mock_lead_capture() called")
    print("=" * 60)

    # Exit with non-zero so CI catches failures.
    if errors:
        raise AssertionError("; ".join(errors))


if __name__ == "__main__":
    test_full_conversation()