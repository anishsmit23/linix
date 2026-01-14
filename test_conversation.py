from agent import build_graph
from langchain_core.messages import HumanMessage

def test_full_conversation():
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
    
    test_messages = [
        "Hi, tell me about your pricing",
        "That sounds good, I want to try the Pro plan for my YouTube channel",
        "John Doe",
        "john@example.com"
    ]
    
    print("=" * 60)
    print("AutoStream Agent - Test Conversation")
    print("=" * 60)
    print()
    
    for user_msg in test_messages:
        print(f"User: {user_msg}")
        
        state["messages"].append(HumanMessage(content=user_msg))
        state["next_step"] = ""
        
        result = graph.invoke(state)
        state = result
        
        agent_response = state["messages"][-1].content
        print(f"Agent: {agent_response}")
        print()
        
        print(f"[State] Intent: {state.get('current_intent')}, Name: {state.get('lead_name')}, Email: {state.get('lead_email')}, Platform: {state.get('lead_platform')}")
        print("-" * 60)
        print()
    
    print("=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_full_conversation()