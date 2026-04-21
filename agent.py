from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import os
import operator
import re
from pathlib import Path
from dotenv import load_dotenv

AGENT_NAME = "Infix"
PRODUCT_NAME = "AutoStream"

load_dotenv()

_LLM: Optional[ChatOpenAI] = None

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

DEFAULT_KNOWLEDGE_BASE = {
    "pricing": {
        "basic": {
            "price": "$29/month",
            "videos_limit": "10 videos/month",
            "resolution": "720p"
        },
        "pro": {
            "price": "$79/month",
            "videos_limit": "Unlimited videos",
            "resolution": "4K",
            "features": ["AI captions"]
        }
    },
    "policies": {
        "refund": "No refunds after 7 days",
        "support": "24/7 support available only on Pro plan"
    }
}

DEFAULT_OBJECTION_LIBRARY = {
    "objections": [
        {
            "type": "price_concern",
            "keywords": ["expensive", "too high", "costly", "price"],
            "response": (
                "I understand pricing is an important consideration. "
                "Basic starts at $29/month, and Pro at $79/month includes unlimited videos, 4K exports, and AI captions."
            ),
        }
    ]
}

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_json_file(filepath: Path, fallback: dict) -> dict:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


KNOWLEDGE_BASE = load_json_file(DATA_DIR / "knowledge_base.json", DEFAULT_KNOWLEDGE_BASE)
OBJECTION_LIBRARY = load_json_file(DATA_DIR / "objection_library.json", DEFAULT_OBJECTION_LIBRARY)


def get_llm() -> Optional[ChatOpenAI]:
    global _LLM
    if _LLM is not None:
        return _LLM
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        _LLM = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            timeout=8,
        )
    except Exception:
        _LLM = None
    return _LLM


def parse_json_object(text: str) -> Optional[dict]:
    text = text.strip()
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


def recommend_plan(state: AgentState, text: str) -> tuple[str, str]:
    text_lower = text.lower()
    selected_plan = state.get("lead_plan")
    if selected_plan:
        return selected_plan, f"You already mentioned {selected_plan}, so we can optimize onboarding for that plan."

    if any(x in text_lower for x in ["team", "agency", "4k", "captions", "unlimited", "scale"]):
        return "Pro", "Pro is best for high output creators because it includes unlimited videos, 4K exports, and AI captions."

    return "Basic", "Basic is ideal if you are starting out and want affordable automation at $29/month."


def pricing_snapshot(pricing: dict) -> str:
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
        enterprise_features = ", ".join(pricing["enterprise"].get("features", []))
        enterprise_line = (
            f"Enterprise Plan: {pricing['enterprise']['price']} - "
            f"{pricing['enterprise']['videos_limit']}, {pricing['enterprise']['resolution']}"
        )
        if enterprise_features:
            enterprise_line += f", {enterprise_features}"
        lines.append(enterprise_line)
    return "\n".join(lines)


def extract_name_with_llm(message: str) -> Optional[str]:
    llm = get_llm()
    if not llm:
        return None

    system = (
        "Extract only a human first/last name from the user message. "
        "If no clear person name is provided, return null. "
        "Respond with strict JSON: {\"name\": string|null}."
    )
    try:
        reply = llm.invoke(
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


def extract_info_from_message(
    message: str,
    state: AgentState,
    allow_loose_name: bool = False,
    allow_llm_name: bool = False,
) -> dict:
    updates: dict = {}
    message_lower = message.lower()

    if not state.get("lead_plan"):
        if "basic" in message_lower:
            updates["lead_plan"] = "Basic"
        elif "pro" in message_lower:
            updates["lead_plan"] = "Pro"
        elif "enterprise" in message_lower:
            updates["lead_plan"] = "Enterprise"
    
    if not state.get("lead_platform"):
        platforms = {
            "youtube": "YouTube",
            "instagram": "Instagram",
            "tiktok": "TikTok",
            "twitch": "Twitch"
        }
        for key, value in platforms.items():
            if key in message_lower:
                updates["lead_platform"] = value
                break
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, message)
    if emails and not state.get("lead_email"):
        updates["lead_email"] = emails[0]
    
    if not state.get("lead_name") and "lead_name" not in updates:
        explicit_name_patterns = [
            r"\bmyself\s+([A-Za-z][A-Za-z'-]{1,30})(?:\b|$)",
            r"\bmy name is\s+([A-Za-z][A-Za-z\s'-]{1,40})$",
            r"\bi am\s+([A-Za-z][A-Za-z\s'-]{1,40})$",
        ]
        disallowed_name_tokens = {
            "interested", "buy", "buying", "price", "pricing", "overpriced", "expensive",
            "cost", "costly", "plan", "basic", "pro", "enterprise", "youtube", "instagram",
            "tiktok", "twitch", "help", "support", "refund", "policy", "feature", "features"
        }
        for pattern in explicit_name_patterns:
            match = re.search(pattern, message_lower)
            if match:
                parsed_name = match.group(1).strip(" .,!?")
                name_parts = [p for p in parsed_name.split() if p]
                if 1 <= len(name_parts) <= 3 and not any(p in disallowed_name_tokens for p in name_parts):
                    updates["lead_name"] = " ".join(part.capitalize() for part in name_parts)
                    break

    if allow_loose_name and not state.get("lead_name") and "lead_name" not in updates and "@" not in message:
        words = message.split()
        stop_words = [
            "hi", "hello", "yes", "no", "ok", "sure", "i", "my", "is", "the", "this", "bit",
            "over", "priced", "price", "pricing", "buy", "buying", "plan", "why", "what", "which"
        ]
        if 1 <= len(words) <= 4 and not any(word.lower() in stop_words for word in words):
            updates["lead_name"] = message.strip()

    if allow_llm_name and not state.get("lead_name") and "lead_name" not in updates:
        llm_name = extract_name_with_llm(message)
        if llm_name:
            updates["lead_name"] = llm_name

    return updates

def classify_intent(state: AgentState) -> dict:
    text = state["messages"][-1].content.lower()
    extracted_updates = extract_info_from_message(
        state["messages"][-1].content,
        state,
        allow_loose_name=False,
        allow_llm_name=False,
    )

    has_greeting = bool(re.search(r"\b(hi|hello|hey)\b", text))

    has_pricing_concern = bool(
        re.search(r"\b(expensive|overpriced|over priced|too high|costly|price|pricing|cost)\b", text)
    )
    has_buy_signal = bool(
        re.search(
            r"\b(try|signup|sign up|get started|buy|buying|purchase|interested|subscribe)\b",
            text,
        )
    )
    has_doubt_signal = bool(re.search(r"\b(why|any reason|at all|worth it)\b", text))
    mentions_plan = bool(re.search(r"\b(plan|basic|pro|enterprise)\b", text))
    has_product_query = has_pricing_concern or bool(re.search(r"\b(refund|support|policy|features?|details|difference|compare)\b", text))

    # Keep collecting lead details only when qualification was explicitly started.
    if state.get("qualification_in_progress") and not state.get("lead_captured"):
        return {**extracted_updates, "current_intent": "high_intent_lead", "next_step": "qualify_lead"}

    if has_greeting and not has_product_query and not has_buy_signal:
        return {**extracted_updates, "current_intent": "greeting", "next_step": "respond_greeting"}

    if has_pricing_concern:
        return {**extracted_updates, "current_intent": "product_or_pricing", "next_step": "retrieve_knowledge"}

    if re.search(r"\b(refund|support|policy|features?)\b", text):
        return {**extracted_updates, "current_intent": "product_or_pricing", "next_step": "retrieve_knowledge"}

    if has_buy_signal and not has_doubt_signal:
        return {
            **extracted_updates,
            "current_intent": "high_intent_lead",
            "next_step": "qualify_lead",
            "qualification_in_progress": True,
        }

    if mentions_plan and not has_doubt_signal and "?" not in text:
        return {
            **extracted_updates,
            "current_intent": "high_intent_lead",
            "next_step": "qualify_lead",
            "qualification_in_progress": True,
        }

    return {**extracted_updates, "current_intent": "product_or_pricing", "next_step": "retrieve_knowledge"}

def respond_greeting(state: AgentState) -> AgentState:
    lead_name = state.get("lead_name")
    lead_platform = state.get("lead_platform")

    if lead_name and lead_platform:
        response = (
            f"Great to meet you {lead_name}! Awesome that you create on {lead_platform}. "
            f"I am {AGENT_NAME}, and I can help you choose the right {PRODUCT_NAME} plan for your workflow. "
            "Do you want a quick recommendation based on your monthly posting volume?"
        )
    elif lead_platform:
        response = (
            f"Great, thanks for sharing that you create on {lead_platform}. "
            f"I am {AGENT_NAME}, and I can help you choose the right {PRODUCT_NAME} plan. "
            "Do you want a quick recommendation based on your monthly posting volume?"
        )
    else:
        response = (
            f"Hello! I am {AGENT_NAME}, and I can help you pick the right {PRODUCT_NAME} plan and get started quickly. "
            "Are you creating for YouTube, Instagram, TikTok, or Twitch?"
        )
    state["messages"].append(AIMessage(content=response))
    state["sales_stage"] = "discovery"
    state["next_step"] = "end"
    return state

def retrieve_knowledge(state: AgentState) -> AgentState:
    text = state["messages"][-1].content.lower()
    pricing = KNOWLEDGE_BASE["pricing"]
    policies = KNOWLEDGE_BASE.get("policies", {})
    objection_count = state.get("objection_count", 0)
    recommended_plan, recommendation_reason = recommend_plan(state, text)

    objection_response = None
    for objection in OBJECTION_LIBRARY.get("objections", []):
        if any(keyword in text for keyword in objection.get("keywords", [])):
            objection_response = objection.get("response")
            break

    if "why should i buy" in text or "worth it" in text:
        response = (
            "Great question. The real value is saving editing time while keeping output quality consistent. "
            f"Based on what you've shared, I recommend {recommended_plan}. {recommendation_reason} "
            "If you'd like, I can get your onboarding started now in under a minute."
        )
        state["sales_stage"] = "value_pitch"
    elif "compare" in text or "difference" in text:
        enterprise_line = ""
        if "enterprise" in pricing:
            enterprise_line = (
                f"\n- Enterprise: {pricing['enterprise']['price']}, "
                f"{pricing['enterprise']['videos_limit']}, {pricing['enterprise']['resolution']}"
            )
        response = (
            "Here is the fastest comparison:\n"
            f"- Basic: {pricing['basic']['price']}, {pricing['basic']['videos_limit']}, {pricing['basic']['resolution']}\n"
            f"- Pro: {pricing['pro']['price']}, {pricing['pro']['videos_limit']}, {pricing['pro']['resolution']}\n"
            f"{enterprise_line}"
            "If you publish frequently or need 4K + AI captions, Pro is usually the better fit."
        )
        state["sales_stage"] = "recommendation"
    elif "which plan" in text or "recommend" in text:
        response = (
            f"I recommend the {recommended_plan} plan. {recommendation_reason} "
            "Want me to start your signup and reserve this plan for you?"
        )
        state["lead_plan"] = recommended_plan
        state["sales_stage"] = "recommendation"
    elif objection_response:
        state["objection_count"] = objection_count + 1
        response = (
            f"{objection_response} "
            "If you tell me your monthly video volume and platform, I can suggest the best-value plan for you."
        )
        state["sales_stage"] = "objection_handling"
    elif "refund" in text:
        response = f"Refund policy: {policies.get('refund', 'No refund policy found.')}"
        state["sales_stage"] = "objection_handling"
    elif "support" in text:
        response = f"Support policy: {policies.get('support', 'No support policy found.')}"
        state["sales_stage"] = "objection_handling"
    elif "feature" in text:
        basic_features = ", ".join(pricing["basic"].get("features", []))
        pro_features = ", ".join(pricing["pro"].get("features", []))
        response = (
            f"Basic includes: {basic_features or 'core editing features'}.\n"
            f"Pro includes: {pro_features or 'AI captions and advanced capabilities'}.\n"
            "If you want, I can help you choose in 30 seconds based on your posting volume."
        )
        state["sales_stage"] = "discovery"
    elif any(x in text for x in ["expensive", "too high", "costly"]):
        response = (
            "I understand pricing is an important consideration.\n\n"
            f"Our Basic plan at {pricing['basic']['price']} is ideal for creators getting started, "
            f"while the Pro plan at {pricing['pro']['price']} is designed for serious creators who need "
            "unlimited videos, 4K exports, and AI captions.\n\n"
            "Would you like help choosing the right plan for your content?"
        )
        state["sales_stage"] = "objection_handling"
    else:
        llm = get_llm()
        if llm:
            rag_context = {
                "product": KNOWLEDGE_BASE.get("product", {}),
                "pricing": KNOWLEDGE_BASE.get("pricing", {}),
                "policies": KNOWLEDGE_BASE.get("policies", {}),
                "objection_hint": objection_response,
                "recommended_plan": recommended_plan,
                "recommendation_reason": recommendation_reason,
                "sales_stage": state.get("sales_stage"),
            }
            try:
                llm_response = llm.invoke(
                    [
                        SystemMessage(
                            content=(
                                f"You are {AGENT_NAME}, a sales assistant for {PRODUCT_NAME}. "
                                "Answer using only the provided knowledge context. "
                                "Be concise, persuasive, and truthful. End with one short CTA question."
                            )
                        ),
                        HumanMessage(
                            content=(
                                f"User message: {state['messages'][-1].content}\n"
                                f"Knowledge context: {json.dumps(rag_context)}"
                            )
                        ),
                    ]
                )
                response = llm_response.content.strip()
            except Exception:
                response = (
                    f"{pricing_snapshot(pricing)}\n"
                    "Would you prefer Basic (budget-friendly) or Pro (unlimited + 4K + AI captions)?"
                )
        else:
            response = (
                f"{pricing_snapshot(pricing)}\n"
                "Would you prefer Basic (budget-friendly) or Pro (unlimited + 4K + AI captions)?"
            )
        state["sales_stage"] = "consideration"

    state["messages"].append(AIMessage(content=response))
    state["next_step"] = "end"
    return state


def qualify_lead(state: AgentState) -> AgentState:
    last_message = state["messages"][-1].content
    extracted_updates = extract_info_from_message(
        last_message,
        state,
        allow_loose_name=True,
        allow_llm_name=True,
    )
    state.update(extracted_updates)
    
    has_name = bool(state.get("lead_name"))
    has_email = bool(state.get("lead_email"))
    has_platform = bool(state.get("lead_platform"))
    
    if has_name and has_email and has_platform:
        state["sales_stage"] = "closing"
        state["next_step"] = "execute_tool"
        return state
    
    if not has_name:
        if state.get("lead_plan"):
            response = f"Great choice with the {state['lead_plan']} plan. To get you started, what's your name?"
        else:
            response = "Great. To get started, what's your name?"
        state["messages"].append(AIMessage(content=response))
        state["next_step"] = "end"
        return state
    
    if not has_email:
        response = f"Thanks {state['lead_name']}! What's your email address?"
        state["messages"].append(AIMessage(content=response))
        state["next_step"] = "end"
        return state
    
    if not has_platform:
        response = "Perfect! Which platform do you create content for? (YouTube, Instagram, TikTok, or Twitch)"
        state["messages"].append(AIMessage(content=response))
        state["next_step"] = "end"
        return state
    
    state["next_step"] = "execute_tool"
    return state

def execute_tool(state: AgentState) -> AgentState:
    if state.get("lead_captured"):
        state["messages"].append(AIMessage(content="You're all set! We'll be in touch soon."))
        state["next_step"] = "end"
        return state
    
    name = state.get("lead_name")
    email = state.get("lead_email")
    platform = state.get("lead_platform")
    
    if name and email and platform:
        mock_lead_capture(name=name, email=email, platform=platform)
        state["lead_captured"] = True
        state["qualification_in_progress"] = False
        if state.get("lead_plan"):
            response = (
                f"Perfect! I've captured your details for the {state['lead_plan']} plan. "
                f"Our team will reach out to {email} shortly to get you started with {PRODUCT_NAME}."
            )
        else:
            response = f"Perfect! I've captured your details. Our team will reach out to {email} shortly to get you started with {PRODUCT_NAME}."
        state["messages"].append(AIMessage(content=response))
        state["sales_stage"] = "won"
    
    state["next_step"] = "end"
    return state

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
            "qualify_lead": "qualify_lead"
        }
    )
    
    workflow.add_conditional_edges(
        "qualify_lead",
        route_next,
        {
            "execute_tool": "execute_tool",
            "end": END
        }
    )
    
    workflow.add_edge("respond_greeting", END)
    workflow.add_edge("retrieve_knowledge", END)
    workflow.add_edge("execute_tool", END)
    
    return workflow.compile()

def run_agent():
    graph = build_graph()
    
    state = {
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
        "next_step": ""
    }
    
    print(f"{AGENT_NAME} Agent (type 'quit' to exit)\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
        
        state["messages"].append(HumanMessage(content=user_input))
        state["next_step"] = ""
        
        result = graph.invoke(state)
        state = result
        
        last_response = state["messages"][-1].content
        print(f"Agent: {last_response}\n")

if __name__ == "__main__":
    run_agent()