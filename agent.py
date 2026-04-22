from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import os
import operator
import re
import time
from pathlib import Path
from dotenv import load_dotenv

AGENT_NAME = "Infix"
PRODUCT_NAME = "AutoStream"

load_dotenv(override=True)

_LLM: Optional[ChatGoogleGenerativeAI] = None
_LLM_MODEL_NAME: Optional[str] = None


def _is_retryable_llm_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = [
        "503",
        "unavailable",
        "429",
        "resource_exhausted",
        "quota",
        "404",
        "not_found",
        "not found",
        "504",
        "deadline_exceeded",
        "deadline exceeded",
        "timed out",
        "timeout",
    ]
    return any(marker in text for marker in markers)


def _configured_model_candidates() -> list[str]:
    preferred = os.getenv("LLM_MODEL", "gemini-flash-latest").strip()
    raw = os.getenv(
        "LLM_FALLBACK_MODELS",
        "gemini-flash-latest,gemini-2.0-flash,gemini-2.5-flash",
    )
    extra = [m.strip() for m in raw.split(",") if m.strip()]

    # Keep order stable and unique while ensuring preferred is first.
    seen = set()
    ordered = [preferred] + extra
    result = []
    for model in ordered:
        if model not in seen:
            seen.add(model)
            result.append(model)
    return result


def _build_llm(model_name: str) -> ChatGoogleGenerativeAI:
    timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "25"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )


def _invoke_llm_safely(llm: ChatGoogleGenerativeAI, messages: list):
    """Invoke LLM in strict mode; retry with configured Gemini alternatives on provider errors."""
    global _LLM, _LLM_MODEL_NAME
    attempts_per_model = int(os.getenv("LLM_ATTEMPTS_PER_MODEL", "2"))
    backoff_seconds = float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", "0.75"))

    ordered_models = []
    if _LLM_MODEL_NAME:
        ordered_models.append(_LLM_MODEL_NAME)
    for model_name in _configured_model_candidates():
        if model_name not in ordered_models:
            ordered_models.append(model_name)

    last_exc: Optional[Exception] = None
    for model_name in ordered_models:
        model_client = llm if model_name == _LLM_MODEL_NAME else _build_llm(model_name)

        for attempt_idx in range(attempts_per_model):
            try:
                reply = model_client.invoke(messages)
                _LLM = model_client
                _LLM_MODEL_NAME = model_name
                return reply
            except Exception as exc:
                last_exc = exc
                if not _is_retryable_llm_error(exc):
                    raise
                if attempt_idx < attempts_per_model - 1:
                    sleep_for = backoff_seconds * (2 ** attempt_idx)
                    time.sleep(sleep_for)

    if last_exc is not None:
        raise RuntimeError(
            "All configured LLM models failed after retries. "
            "Check provider status/quota and try again."
        ) from last_exc

    raise RuntimeError("LLM invocation failed before any model call was attempted.")


def _content_to_text(content) -> str:
    """Normalize LangChain/Gemini message content blocks to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        return json.dumps(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            text = _content_to_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(content)


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    current_intent: Optional[str]
    sales_stage: Optional[str]
    qualification_in_progress: bool
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_plan: Optional[str]
    objection_count: int
    lead_captured: bool
    next_step: str


# ---------------------------------------------------------------------------
# Knowledge base & objection library (loaded once at import time)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_json_file(filepath: Path, fallback: dict) -> dict:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


DEFAULT_KNOWLEDGE_BASE = {
    "pricing": {
        "basic": {
            "price": "$29/month",
            "videos_limit": "10 videos/month",
            "resolution": "720p",
        },
        "pro": {
            "price": "$79/month",
            "videos_limit": "Unlimited videos",
            "resolution": "4K",
            "features": ["AI captions"],
        },
    },
    "policies": {
        "refund": "No refunds after 7 days",
        "support": "24/7 support available only on Pro plan",
    },
}

DEFAULT_OBJECTION_LIBRARY = {
    "objections": [
        {
            "type": "price_concern",
            "keywords": ["expensive", "too high", "costly", "price"],
            "response": (
                "I understand pricing is an important consideration. "
                "Basic starts at $29/month, and Pro at $79/month includes "
                "unlimited videos, 4K exports, and AI captions."
            ),
        }
    ]
}

KNOWLEDGE_BASE = load_json_file(DATA_DIR / "knowledge_base.json", DEFAULT_KNOWLEDGE_BASE)
OBJECTION_LIBRARY = load_json_file(DATA_DIR / "objection_library.json", DEFAULT_OBJECTION_LIBRARY)


# ---------------------------------------------------------------------------
# LLM singleton
# ---------------------------------------------------------------------------

def get_llm() -> ChatGoogleGenerativeAI:
    global _LLM, _LLM_MODEL_NAME
    if _LLM is not None:
        return _LLM
    llm_enabled = os.getenv("ENABLE_LLM", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not llm_enabled:
        raise RuntimeError("LLM is mandatory. Set ENABLE_LLM=true in .env.")
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("LLM is mandatory. GOOGLE_API_KEY is missing in .env.")

    _LLM_MODEL_NAME = os.getenv("LLM_MODEL", "gemini-flash-latest").strip()
    _LLM = _build_llm(_LLM_MODEL_NAME)
    return _LLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_json_object(text) -> Optional[dict]:
    """Extract the first JSON object from *text*."""
    text = _content_to_text(text).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def mock_lead_capture(name: str, email: str, platform: str):
    print(f"Lead captured successfully: {name}, {email}, {platform}")


def pricing_snapshot(pricing: dict) -> str:
    """Render a compact pricing summary from the knowledge base dict."""
    basic_line = (
        f"Basic Plan: {pricing['basic']['price']} - "
        f"{pricing['basic']['videos_limit']}, {pricing['basic']['resolution']}"
    )
    pro_features = ", ".join(pricing["pro"].get("features", ["AI captions"]))
    pro_line = (
        f"Pro Plan: {pricing['pro']['price']} - "
        f"{pricing['pro']['videos_limit']}, {pricing['pro']['resolution']}, {pro_features}"
    )
    lines = [basic_line, pro_line]
    if "enterprise" in pricing:
        ent = pricing["enterprise"]
        ent_features = ", ".join(ent.get("features", []))
        ent_line = (
            f"Enterprise Plan: {ent['price']} - "
            f"{ent['videos_limit']}, {ent['resolution']}"
        )
        if ent_features:
            ent_line += f", {ent_features}"
        lines.append(ent_line)
    return "\n".join(lines)


def build_rag_context(state: dict) -> str:
    """Serialize the full knowledge base + current state into a context string
    that is injected into every LLM call as the RAG source-of-truth."""
    context = {
        "product": KNOWLEDGE_BASE.get("product", {}),
        "pricing": KNOWLEDGE_BASE.get("pricing", {}),
        "policies": KNOWLEDGE_BASE.get("policies", {}),
        "objection_library": OBJECTION_LIBRARY.get("objections", []),
        "current_sales_stage": state.get("sales_stage"),
        "lead_plan": state.get("lead_plan"),
    }
    return json.dumps(context, indent=2)


# ---------------------------------------------------------------------------
# Info extraction (regex — used once per turn in the appropriate node)
# ---------------------------------------------------------------------------

def extract_info_from_message(
    message: str,
    state: dict,
    allow_loose_name: bool = False,
    allow_llm_name: bool = False,
) -> dict:
    """Parse structured lead details (plan, platform, email, name) from a
    single user message.  Returns a partial-state dict with only the keys
    that were newly discovered."""
    updates: dict = {}
    message_lower = message.lower()

    # --- plan ---
    if not state.get("lead_plan"):
        if "basic" in message_lower:
            updates["lead_plan"] = "Basic"
        elif "pro" in message_lower:
            updates["lead_plan"] = "Pro"
        elif "enterprise" in message_lower:
            updates["lead_plan"] = "Enterprise"

    # --- platform ---
    if not state.get("lead_platform"):
        platforms = {
            "youtube": "YouTube",
            "instagram": "Instagram",
            "tiktok": "TikTok",
            "twitch": "Twitch",
        }
        for key, value in platforms.items():
            if key in message_lower:
                updates["lead_platform"] = value
                break

    # --- email ---
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    emails = re.findall(email_pattern, message)
    if emails and not state.get("lead_email"):
        updates["lead_email"] = emails[0]

    # --- name (explicit patterns first) ---
    if not state.get("lead_name") and "lead_name" not in updates:
        explicit_name_patterns = [
            r"\bmyself\s+([A-Za-z][A-Za-z'-]{1,30})(?:\b|$)",
            r"\bmy name is\s+([A-Za-z][A-Za-z\s'-]{1,40})$",
            r"\bi am\s+([A-Za-z][A-Za-z\s'-]{1,40})$",
        ]
        disallowed_name_tokens = {
            "interested", "buy", "buying", "price", "pricing", "overpriced",
            "expensive", "cost", "costly", "plan", "basic", "pro",
            "enterprise", "youtube", "instagram", "tiktok", "twitch",
            "help", "support", "refund", "policy", "feature", "features",
        }
        for pattern in explicit_name_patterns:
            match = re.search(pattern, message_lower)
            if match:
                parsed_name = match.group(1).strip(" .,!?")
                name_parts = [p for p in parsed_name.split() if p]
                if 1 <= len(name_parts) <= 3 and not any(
                    p in disallowed_name_tokens for p in name_parts
                ):
                    updates["lead_name"] = " ".join(
                        part.capitalize() for part in name_parts
                    )
                    break

    # --- name (loose heuristic — short messages during qualification) ---
    if (
        allow_loose_name
        and not state.get("lead_name")
        and "lead_name" not in updates
        and "@" not in message
    ):
        words = message.split()
        stop_words = [
            "hi", "hello", "yes", "no", "ok", "sure", "i", "my", "is",
            "the", "this", "bit", "over", "priced", "price", "pricing",
            "buy", "buying", "plan", "why", "what", "which",
        ]
        if 1 <= len(words) <= 4 and not any(
            word.lower() in stop_words for word in words
        ):
            updates["lead_name"] = message.strip()

    # --- name (LLM extraction — last resort) ---
    if (
        allow_llm_name
        and not state.get("lead_name")
        and "lead_name" not in updates
        and _looks_like_name_candidate(message)
    ):
        llm_name = _extract_name_with_llm(message)
        if llm_name:
            updates["lead_name"] = llm_name

    return updates


def _extract_name_with_llm(message: str) -> Optional[str]:
    llm = get_llm()
    if not llm:
        return None
    system = (
        "Extract only a human first/last name from the user message. "
        "If no clear person name is provided, return null. "
        'Respond with strict JSON: {"name": string|null}.'
    )
    try:
        reply = _invoke_llm_safely(
            llm,
            [
                SystemMessage(content=system),
                HumanMessage(content=f"Message: {message}"),
            ]
        )
        parsed = parse_json_object(reply.content)
        if not parsed or not parsed.get("name"):
            return None
        name = str(parsed["name"]).strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z\s'-]{0,48}", name):
            return None
        return " ".join(part.capitalize() for part in name.split())
    except Exception:
        return None


def _looks_like_name_candidate(message: str) -> bool:
    """Return True when a message plausibly contains only a person name."""
    stripped = message.strip()
    if not stripped or "@" in stripped:
        return False

    tokens = stripped.split()
    if not (1 <= len(tokens) <= 4):
        return False

    return bool(re.fullmatch(r"[A-Za-z][A-Za-z\s'\-]{0,48}", stripped))


# ---------------------------------------------------------------------------
# LLM-powered intent classification
# ---------------------------------------------------------------------------

_INTENT_SYSTEM_PROMPT = """\
You are an intent classifier for a SaaS sales agent.

Classify the user's message into exactly ONE of these intents:
- "greeting"             → casual hello / hi / hey with no product question
- "product_or_pricing"   → asking about pricing, features, plans, refunds, support, comparisons, or objections
- "high_intent_lead"     → user wants to sign up, subscribe, buy, start, try a plan, or get started

Respond with strict JSON only:
{"intent": "greeting" | "product_or_pricing" | "high_intent_lead"}
"""


def _classify_intent_with_llm(user_message: str) -> str:
    """Ask the LLM to classify intent. Returns one of the three canonical
    intent strings, or None on failure."""
    llm = get_llm()
    reply = _invoke_llm_safely(
        llm,
        [
            SystemMessage(content=_INTENT_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
    )
    parsed = parse_json_object(reply.content)
    if parsed and parsed.get("intent") in (
        "greeting",
        "product_or_pricing",
        "high_intent_lead",
    ):
        return parsed["intent"]
    raise ValueError("LLM returned invalid intent JSON. Expected one of greeting/product_or_pricing/high_intent_lead.")


_INTENT_TO_NEXT = {
    "greeting": "respond_greeting",
    "product_or_pricing": "retrieve_knowledge",
    "high_intent_lead": "qualify_lead",
}


# ---------------------------------------------------------------------------
# NODE: classify_intent
# ---------------------------------------------------------------------------

def classify_intent(state: AgentState) -> dict:
    """Entry-point node.  Classifies intent, extracts structured info, and
    returns a *partial* state dict that LangGraph merges."""

    user_message = state["messages"][-1].content

    # If we are mid-qualification, skip reclassification.
    if state.get("qualification_in_progress") and not state.get("lead_captured"):
        return {"current_intent": "high_intent_lead", "next_step": "qualify_lead"}

    # Strict LLM-only intent classification (no regex fallback).
    intent = _classify_intent_with_llm(user_message)

    # Extract structured info only once per turn (no LLM name extraction
    # here — that happens inside qualify_lead when needed).
    extracted = extract_info_from_message(
        user_message, state, allow_loose_name=False, allow_llm_name=False
    )

    result: dict = {**extracted, "current_intent": intent}

    if intent == "high_intent_lead":
        result["next_step"] = "qualify_lead"
        result["qualification_in_progress"] = True
    else:
        result["next_step"] = _INTENT_TO_NEXT.get(intent, "retrieve_knowledge")

    return result


# ---------------------------------------------------------------------------
# NODE: respond_greeting
# ---------------------------------------------------------------------------

def respond_greeting(state: AgentState) -> dict:
    lead_name = state.get("lead_name")
    lead_platform = state.get("lead_platform")

    if lead_name and lead_platform:
        response = (
            f"Great to meet you {lead_name}! Awesome that you create on "
            f"{lead_platform}. I am {AGENT_NAME}, and I can help you choose "
            f"the right {PRODUCT_NAME} plan for your workflow. "
            "Do you want a quick recommendation based on your monthly posting volume?"
        )
    elif lead_platform:
        response = (
            f"Great, thanks for sharing that you create on {lead_platform}. "
            f"I am {AGENT_NAME}, and I can help you choose the right "
            f"{PRODUCT_NAME} plan. Do you want a quick recommendation based "
            "on your monthly posting volume?"
        )
    else:
        response = (
            f"Hello! I am {AGENT_NAME}, and I can help you pick the right "
            f"{PRODUCT_NAME} plan and get started quickly. "
            "Are you creating for YouTube, Instagram, TikTok, or Twitch?"
        )

    return {
        "messages": [AIMessage(content=response)],
        "sales_stage": "discovery",
        "next_step": "end",
    }


# ---------------------------------------------------------------------------
# NODE: retrieve_knowledge  (LLM-first, with RAG context)
# ---------------------------------------------------------------------------

_KNOWLEDGE_SYSTEM_PROMPT = f"""\
You are {AGENT_NAME}, a friendly and persuasive sales assistant for {PRODUCT_NAME}, \
an AI-powered automated video editing tool for content creators.

RULES:
1. Answer ONLY using the knowledge context provided below — never invent facts.
2. Be concise (2-4 sentences), warm, and persuasive.
3. When discussing pricing, always mention exact dollar amounts from the knowledge base.
4. Handle objections empathetically — acknowledge the concern, then reframe with value.
5. End every response with one short call-to-action question that nudges the user \
   toward trying or signing up.
6. Do NOT ask for name / email / platform here — that happens only after the user \
   decides to sign up.
"""


def retrieve_knowledge(state: AgentState) -> dict:
    """Answers product / pricing / policy questions.  The LLM generates the
    response using the full knowledge base as RAG context. Strict LLM-only mode."""

    user_message = state["messages"][-1].content
    user_message_lower = user_message.lower()
    objection_hit = _is_objection_message(user_message_lower)

    rag_context = build_rag_context(state)

    llm = get_llm()
    llm_reply = _invoke_llm_safely(
        llm,
        [
            SystemMessage(
                content=(
                    _KNOWLEDGE_SYSTEM_PROMPT
                    + f"\n\nKNOWLEDGE CONTEXT:\n{rag_context}"
                )
            ),
            # Include recent conversation history so the LLM has
            # multi-turn context (up to last 10 messages).
            *state["messages"][-10:],
        ]
    )
    response = _content_to_text(llm_reply.content).strip()
    if not response:
        raise ValueError("LLM returned empty response content.")

    updates = {
        "messages": [AIMessage(content=response)],
        "sales_stage": "consideration",
        "next_step": "end",
    }
    if objection_hit:
        updates["objection_count"] = state.get("objection_count", 0) + 1

    return updates


def _is_objection_message(message_lower: str) -> bool:
    for objection in OBJECTION_LIBRARY.get("objections", []):
        for keyword in objection.get("keywords", []):
            if keyword.lower() in message_lower:
                return True
    return False




# ---------------------------------------------------------------------------
# NODE: qualify_lead
# ---------------------------------------------------------------------------

def qualify_lead(state: AgentState) -> dict:
    """Collects name → email → platform one field at a time.  Routes to
    execute_tool only when all three are present."""

    last_message = state["messages"][-1].content

    # Extract info with loose-name and LLM-name enabled (only call site).
    extracted = extract_info_from_message(


        last_message, state, allow_loose_name=True, allow_llm_name=True
    )

    # Build a merged view so we can check completeness.
    merged = {**state, **extracted}



    has_name = bool(merged.get("lead_name"))
    has_email = bool(merged.get("lead_email"))
    has_platform = bool(merged.get("lead_platform"))

    if has_name and has_email and has_platform:
        return {
            **extracted,
            "sales_stage": "closing",
            "next_step": "execute_tool",


        }



    # Ask for the next missing field.
    if not has_name:
        plan = merged.get("lead_plan")
        if plan:
            response = (
                f"Great choice with the {plan} plan. "
                "To get you started, what's your name?"
            )
        else:
            response = "Great. To get started, what's your name?"
    elif not has_email:


        response = f"Thanks {merged['lead_name']}! What's your email address?"
    else:
        response = (
            "Perfect! Which platform do you create content for? "
            "(YouTube, Instagram, TikTok, or Twitch)"
        )

    return {
        **extracted,
        "messages": [AIMessage(content=response)],
        "next_step": "end",
    }


# ---------------------------------------------------------------------------
# NODE: execute_tool
# ---------------------------------------------------------------------------



def execute_tool(state: AgentState) -> dict:
    if state.get("lead_captured"):
        return {
            "messages": [AIMessage(content="You're all set! We'll be in touch soon.")],
            "next_step": "end",
        }

    name = state.get("lead_name")
    email = state.get("lead_email")
    platform = state.get("lead_platform")

    if name and email and platform:
        mock_lead_capture(name=name, email=email, platform=platform)
        plan = state.get("lead_plan")
        if plan:
            response = (
                f"Perfect! I've captured your details for the {plan} plan. "
                f"Our team will reach out to {email} shortly to get you "
                f"started with {PRODUCT_NAME}."
            )
        else:
            response = (
                f"Perfect! I've captured your details. Our team will reach "
                f"out to {email} shortly to get you started with "
                f"{PRODUCT_NAME}."
            )
        return {
            "messages": [AIMessage(content=response)],
            "lead_captured": True,
            "qualification_in_progress": False,
            "sales_stage": "won",
            "next_step": "end",
        }

    # Should not normally reach here — qualify_lead guards this.
    return {"next_step": "end"}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------

def route_next(state: AgentState) -> str:
    return state["next_step"]


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("respond_greeting", respond_greeting)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge)
    workflow.add_node("qualify_lead", qualify_lead)
    workflow.add_node("execute_tool", execute_tool)

    workflow.set_entry_point("classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        route_next,
        {
            "respond_greeting": "respond_greeting",
            "retrieve_knowledge": "retrieve_knowledge",
            "qualify_lead": "qualify_lead",
        },
    )

    workflow.add_conditional_edges(
        "qualify_lead",
        route_next,
        {
            "execute_tool": "execute_tool",
            "end": END,
        },
    )

    workflow.add_edge("respond_greeting", END)
    workflow.add_edge("retrieve_knowledge", END)
    workflow.add_edge("execute_tool", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

def new_state() -> dict:
    """Create a fresh initial state dict."""
    return {
        "messages": [],
        "current_intent": None,
        "sales_stage": None,
        "qualification_in_progress": False,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_plan": None,
        "objection_count": 0,
        "lead_captured": False,
        "next_step": "",
    }


def run_agent():
    graph = build_graph()
    state = new_state()

    print(f"{AGENT_NAME} Agent (type 'quit' to exit)\n")

    # Validate required LLM configuration early in strict mode.
    try:
        get_llm()
    except Exception as exc:
        print(f"Agent startup error: {exc}")
        print("Set ENABLE_LLM=true, provide GOOGLE_API_KEY, and use a supported LLM_MODEL.")
        return

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break

        state["messages"].append(HumanMessage(content=user_input))
        state["next_step"] = ""

        try:
            result = graph.invoke(state)
        except Exception as exc:
            print(f"Agent error: {exc}")
            message = str(exc).lower()
            if "deadline" in message or "timeout" in message or "unavailable" in message:
                print("No fallback is enabled. Provider timeout/capacity issue detected; please retry.")
            else:
                print("No fallback is enabled. Fix LLM_MODEL / GOOGLE_API_KEY and retry.")
            continue
        state = result

        last_response = state["messages"][-1].content
        print(f"Agent: {last_response}\n")


if __name__ == "__main__":
    run_agent()