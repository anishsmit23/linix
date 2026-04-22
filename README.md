<div align="center">

<br/>

```
██╗███╗   ██╗███████╗██╗██╗  ██╗
██║████╗  ██║██╔════╝██║╚██╗██╔╝
██║██╔██╗ ██║█████╗  ██║ ╚███╔╝ 
██║██║╚██╗██║██╔══╝  ██║ ██╔██╗ 
██║██║ ╚████║██║     ██║██╔╝ ██╗
╚═╝╚═╝  ╚═══╝╚═╝     ╚═╝╚═╝  ╚═╝
```

### Social-to-Lead Agentic Workflow
**A LangGraph-powered conversational sales agent for AutoStream**

*ServiceHive · ML Intern Assignment Project*

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.28-1C3A5E?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-2E8B57?style=flat-square)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Gemini--1.5--Flash-Powered-4285F4?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-F7DF1E?style=flat-square)](LICENSE)

<br/>

> Infix detects buyer intent, retrieves accurate pricing from a local knowledge base,  
> qualifies high-intent leads step-by-step, and fires a lead-capture tool —  
> all within a single stateful, multi-turn conversation.

<br/>

</div>

---

## 📑 Table of Contents

- [Demo](#-demo)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Running the Agent](#-running-the-agent)
- [WhatsApp Deployment](#-whatsapp-deployment-via-webhook)
- [Evaluation Coverage](#-evaluation-coverage)

---

## 🎬 Demo

Below is a real conversation captured from the running agent:

```
You:    Hi, tell me about your pricing
Infix:  Hello! AutoStream has two main plans:
        • Basic — $29/month for 10 videos at 720p
        • Pro — $79/month for unlimited videos, 4K resolution, and AI captions
        Would you like to know which plan fits your workflow best?

You:    This sounds expensive, why should I buy Pro?
Infix:  I understand pricing matters! Pro at $79/month pays for itself quickly —
        unlimited videos means no per-video cost, you get 4K exports for
        professional quality, plus AI captions save hours of manual work.
        Would you like to try Pro and see the difference?

You:    Okay, I want to start with Pro for my YouTube channel
Infix:  Great choice with the Pro plan. To get you started, what's your name?

You:    John Doe
Infix:  Thanks John Doe! What's your email address?

You:    john@example.com
Infix:  Perfect! Which platform do you create content for?
        (YouTube, Instagram, TikTok, or Twitch)

You:    YouTube
        >> Lead captured successfully: John Doe, john@example.com, YouTube

Infix:  Perfect! I've captured your details for the Pro plan.
        Our team will reach out to john@example.com shortly to get you
        started with AutoStream.
```

---

## ✨ Features

| # | Capability | How it works |
|---|---|---|
| 🧠 | **LLM-Powered Intent Classification** | Every user message is classified by the LLM into `greeting`, `product_or_pricing`, or `high_intent_lead` using structured JSON output — handles natural language variations that regex would miss |
| 📚 | **RAG-Powered Knowledge Retrieval** | The full knowledge base (`data/knowledge_base.json`) is injected as context into every LLM response call, ensuring answers are grounded in real product data |
| 🤖 | **LLM as Primary Brain** | `gpt-4o-mini` generates all product/pricing/objection responses — no hardcoded response templates for the demo flow |
| 🪤 | **Gated Lead Qualification** | Name → Email → Platform collected one field at a time; the capture tool never fires until all three are confirmed |
| 🔧 | **Tool Execution** | `mock_lead_capture(name, email, platform)` is invoked only at the `execute_tool` graph node, never prematurely |
| 🔁 | **Full Conversation Memory** | `AgentState` carries all message history and lead fields across every turn with no external store required |
| 💬 | **Objection Handling** | Price objections and feature questions are naturally handled by the LLM using the objection library as additional context |
| 🧩 | **Graceful Regex Fallback** | If no API key is set, intent classification and responses fall back to rule-based logic — no crashes, no degraded UX |

---

## 🏗 Architecture

### Why LangGraph?

LangGraph was chosen over plain LangChain chains or AutoGen for three concrete reasons:

**1. Explicit, auditable state machine**  
The agent's logic is a directed graph with named nodes. Every routing decision — when to qualify a lead, when to fire the tool — is visible in the graph topology, not buried in a prompt. This makes the "no premature tool calls" requirement enforceable by structure rather than by prompt wording alone.

**2. Typed persistent state**  
`AgentState` is a `TypedDict` that carries the full conversation, all lead fields, and the current sales stage across every node invocation. No external memory store or vector database is needed to satisfy the 5–6 turn retention requirement.

**3. Conditional routing co-located with decisions**  
`add_conditional_edges` lets each node signal its own successor via the `next_step` field. Routing logic lives next to the code that makes the decision — easier to trace and test than a central dispatcher pattern.

---

### Graph Topology

```
                        ┌─────────────────┐
         User Input ──► │ classify_intent │  (LLM classifies intent)
                        └────────┬────────┘
                                 │
              ┌──────────────────┼───────────────────┐
              │                  │                   │
              ▼                  ▼                   ▼
    ┌──────────────────┐ ┌───────────────────┐ ┌──────────────┐
    │ respond_greeting │ │ retrieve_knowledge│ │ qualify_lead │
    └────────┬─────────┘ └─────────┬─────────┘ └──────┬───────┘
             │                     │                   │
             │              (LLM + RAG context)   ┌────┴──────────────┐
             │                     │              │                   │
             ▼                     ▼              ▼                   ▼
            END                   END       execute_tool             END
                                                  │           (still collecting)
                                                  ▼
                                                 END
```

---

### LLM-First Architecture

```
User message
     │
     ▼
classify_intent
     │
     ├── LLM available?
     │     └── YES → LLM classifies intent via structured JSON
     │                {"intent": "greeting|product_or_pricing|high_intent_lead"}
     │
     └── NO  → regex fallback (keyword matching)
     │
     ▼
retrieve_knowledge  (for product_or_pricing intent)
     │
     ├── LLM available?
     │     └── YES → LLM generates response with full knowledge base as RAG context
     │               System prompt + knowledge JSON + conversation history
     │
     └── NO  → static pricing snapshot fallback
```

**The LLM drives both decisions (intent) and responses (knowledge retrieval).** Regex and templates are only used as graceful fallbacks when no API key is configured.

---

### State Management

Every turn passes through one shared `AgentState` dict. All nodes return **partial dicts** that LangGraph merges — no node mutates and returns the full state, which prevents message duplication with the `Annotated[list, operator.add]` reducer.

```python
class AgentState(TypedDict):
    messages:                  list        # Full HumanMessage / AIMessage history
    current_intent:            str | None  # greeting | product_or_pricing | high_intent_lead
    sales_stage:               str | None  # discovery → consideration → closing → won
    qualification_in_progress: bool        # Guards re-classification during lead collection
    lead_name:                 str | None  # Collected — step 1
    lead_email:                str | None  # Collected — step 2
    lead_platform:             str | None  # Collected — step 3
    lead_plan:                 str | None  # Basic | Pro | Enterprise
    objection_count:           int         # Tracks how many objections have been raised
    lead_captured:             bool        # True after mock_lead_capture() fires
    next_step:                 str         # Routing signal consumed by conditional edges
```

---

## 📁 Project Structure

```
linix/
│
├── agent.py                       # LangGraph graph, all nodes, CLI loop
│
├── data/
│   ├── knowledge_base.json        # Product info, pricing tiers, policies  ← RAG source
│   ├── objection_library.json     # Objection types, trigger keywords, responses
│   └── conversation_examples.json # Reference examples for intent types
│
├── validate_data.py               # Validates JSON schema and field completeness
├── test_conversation.py           # Scripted end-to-end test with assertions
├── install.sh                     # One-command dependency installer
├── requirements.txt               # Pinned dependency versions
├── .env.example                   # Environment variable template
└── .gitignore
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python **3.9** or higher
- pip
- An OpenAI API key *(optional — agent runs without one in rule-based fallback mode)*

---

### 1 · Clone the repository

```bash
git clone https://github.com/anishsmit23/linix.git
cd linix
```

### 2 · Install dependencies

```bash
pip install -r requirements.txt
```

Or use the bundled helper:

```bash
bash install.sh
```

### 3 · Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your key:

```env
ENABLE_LLM=true
GOOGLE_API_KEY=AIza...
LLM_MODEL=gemini-1.5-flash
LLM_TIMEOUT_SECONDS=10
LLM_MAX_RETRIES=1
```

> **LLM is opt-in for reliability.** Keep `ENABLE_LLM=false` to force deterministic rule-based behavior.
> Set `ENABLE_LLM=true` (with a valid `GOOGLE_API_KEY`) to enable Gemini-powered intent and response generation.

---

## 🚀 Running the Agent

**Start a conversation**

```bash
python agent.py
```

**Validate knowledge base and objection library**

```bash
python validate_data.py
```

**Run the scripted end-to-end test suite**

```bash
python test_conversation.py
```

---

## 📲 WhatsApp Deployment via Webhook

To deploy Infix on WhatsApp, the recommended approach uses the **WhatsApp Business Cloud API** (Meta) with a stateless webhook receiver backed by Redis.

### Architecture

```
WhatsApp User
      │  (sends message)
      ▼
Meta Cloud API ──POST /webhook──► FastAPI Server
                                        │
                                  Load state from Redis
                                  (keyed by phone number)
                                        │
                                  graph.invoke(state)
                                        │
                                  Save updated state to Redis
                                        │
                              ◄── WhatsApp Send Message API
      │  (receives reply)
WhatsApp User
```

### Step-by-step

**1. Register a Meta App**  
Create a WhatsApp Business app at [developers.facebook.com](https://developers.facebook.com). Obtain a `phone_number_id` and a permanent `access_token`.

**2. Wrap the agent in a webhook server**

```python
from fastapi import FastAPI, Request
import redis, json
from agent import build_graph, new_state
from langchain_core.messages import HumanMessage

app   = FastAPI()
graph = build_graph()
r     = redis.Redis()

@app.get("/webhook")          # Meta verification handshake
async def verify(hub_challenge: str):
    return int(hub_challenge)

@app.post("/webhook")         # Incoming user messages
async def receive(request: Request):
    body  = await request.json()
    phone = body["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
    text  = body["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]

    raw   = r.get(phone)
    state = json.loads(raw) if raw else new_state()

    state["messages"].append(HumanMessage(content=text))
    state["next_step"] = ""
    state = graph.invoke(state)

    r.set(phone, json.dumps(state, default=str))
    reply = state["messages"][-1].content
    send_whatsapp_message(phone, reply)   # call Meta Send Message API
```

**3. Expose via HTTPS**  
Use [ngrok](https://ngrok.com) locally, or deploy to Railway / Render / AWS in production. Meta requires a valid TLS endpoint.

**4. Register your webhook URL**  
In the Meta App Dashboard, set the webhook URL and subscribe to the `messages` topic.

**The only code change needed:** replace the in-memory `AgentState` dict with Redis storage keyed by `from_phone_number`. The `agent.py` graph itself requires no modifications.

---

## ✅ Evaluation Coverage

| Criterion | Implementation | Status |
|---|---|---|
| Intent — greeting | LLM classifies via structured JSON `{"intent": "greeting"}`, regex fallback | ✅ |
| Intent — product / pricing | LLM classifies via structured JSON, regex fallback for `price`, `plan`, `refund`, etc. | ✅ |
| Intent — high-intent lead | LLM classifies buy signals naturally; regex fallback: `subscribe`, `buy`, `sign up`, etc. | ✅ |
| RAG from local knowledge base | Full `knowledge_base.json` injected as LLM context for every response | ✅ |
| Lead gated before tool call | `qualify_lead` collects all 3 fields before routing to `execute_tool` | ✅ |
| `mock_lead_capture()` tool | Fires inside `execute_tool` node only | ✅ |
| State retained across 5–6 turns | `AgentState` TypedDict persists across all graph node invocations | ✅ |
| Objection handling | Objection library injected as LLM context; LLM generates empathetic responses | ✅ |
| LLM as primary brain | LLM drives both intent classification AND response generation | ✅ |
| LLM name extraction | `extract_name_with_llm()` called inside `qualify_lead` only (no duplicate calls) | ✅ |
| Consistent state management | All nodes return partial dicts; no mutation + full-state return conflicts | ✅ |
| Test with assertions | `test_conversation.py` runs 5 turns and asserts all lead fields + `lead_captured` | ✅ |
| Graceful fallback (LLM disabled/unavailable) | Regex intent + static pricing snapshot when `ENABLE_LLM=false` or `GOOGLE_API_KEY` is absent | ✅ |
| Pinned dependencies | All versions locked in `requirements.txt` | ✅ |
| WhatsApp deployment plan | Documented above with working code sketch | ✅ |

---

<div align="center">
<br/>

Built by **Anish** for the **ServiceHive ML Internship Assignment**

*Infix · AutoStream · LangGraph*

</div>