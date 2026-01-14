from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import operator
import re

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    current_intent: Optional[str]
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    next_step: str

KNOWLEDGE_BASE = {
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

def mock_lead_capture(name: str, email: str, platform: str):
    print(f"Lead captured successfully: {name}, {email}, {platform}")

def extract_info_from_message(message: str, state: AgentState) -> AgentState:
    message_lower = message.lower()
    
    if not state.get("lead_platform"):
        platforms = {
            "youtube": "YouTube",
            "instagram": "Instagram",
            "tiktok": "TikTok",
            "twitch": "Twitch"
        }
        for key, value in platforms.items():
            if key in message_lower:
                state["lead_platform"] = value
                break
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, message)
    if emails and not state.get("lead_email"):
        state["lead_email"] = emails[0]
    
    if not state.get("lead_name") and not state.get("lead_email"):
        words = message.split()
        if len(words) <= 4 and not any(word.lower() in ["hi", "hello", "yes", "no", "ok", "sure", "i", "my", "is", "the"] for word in words):
            if "@" not in message:
                state["lead_name"] = message.strip()
    
    return state

def classify_intent(state: AgentState) -> dict:
    text = state["messages"][-1].content.lower()

    if re.search(r"\b(hi|hello|hey)\b", text):
        return {"current_intent": "greeting", "next_step": "respond_greeting"}

    if re.search(r"\b(expensive|too high|price|pricing|cost)\b", text):
        return {"current_intent": "product_or_pricing", "next_step": "retrieve_knowledge"}

    if re.search(r"\b(try|signup|sign up|get started|pro)\b", text):
        return {"current_intent": "high_intent_lead", "next_step": "qualify_lead"}

    return {"current_intent": "product_or_pricing", "next_step": "retrieve_knowledge"}

def respond_greeting(state: AgentState) -> AgentState:
    response = "Hello! I'm here to help you learn about AutoStream. What would you like to know?"
    state["messages"].append(AIMessage(content=response))
    state["next_step"] = "end"
    return state

def retrieve_knowledge(state: AgentState) -> AgentState:
    text = state["messages"][-1].content.lower()
    pricing = KNOWLEDGE_BASE["pricing"]

    if any(x in text for x in ["expensive", "too high", "costly"]):
        response = (
            "I understand pricing is an important consideration.\n\n"
            f"Our Basic plan at {pricing['basic']['price']} is ideal for creators getting started, "
            f"while the Pro plan at {pricing['pro']['price']} is designed for serious creators who need "
            "unlimited videos, 4K exports, and AI captions.\n\n"
            "Would you like help choosing the right plan for your content?"
        )
    else:
        response = (
            f"Basic Plan: {pricing['basic']['price']} – {pricing['basic']['videos_limit']}, {pricing['basic']['resolution']}\n"
            f"Pro Plan: {pricing['pro']['price']} – {pricing['pro']['videos_limit']}, {pricing['pro']['resolution']}, AI captions"
        )

    state["messages"].append(AIMessage(content=response))
    state["next_step"] = "end"
    return state


def qualify_lead(state: AgentState) -> AgentState:
    last_message = state["messages"][-1].content
    state = extract_info_from_message(last_message, state)
    
    has_name = bool(state.get("lead_name"))
    has_email = bool(state.get("lead_email"))
    has_platform = bool(state.get("lead_platform"))
    
    if has_name and has_email and has_platform:
        state["next_step"] = "execute_tool"
        return state
    
    if not has_name:
        response = "Great! To get you started, what's your name?"
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
        response = f"Perfect! I've captured your details. Our team will reach out to {email} shortly to get you started with AutoStream."
        state["messages"].append(AIMessage(content=response))
    
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
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "next_step": ""
    }
    
    print("AutoStream Agent (type 'quit' to exit)\n")
    
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