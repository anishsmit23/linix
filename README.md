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
[![LangChain](https://img.shields.io/badge/LangChain-0.2.16-2E8B57?style=flat-square)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/GPT--4o--mini-Powered-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)
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
You:    hello
Infix:  Hello! I am Infix, and I can help you pick the right AutoStream plan
        and get started quickly. Are you creating for YouTube, Instagram, TikTok, or Twitch?

You:    can you explain me your plans
Infix:  Basic Plan:      $29/month  — 10 videos/month, 720p
        Pro Plan:        $79/month  — Unlimited videos, 4K, AI captions, Priority rendering, 24/7 chat support
        Enterprise Plan: Custom     — Unlimited, 4K+, Dedicated success manager, SLA-backed support
        Would you prefer Basic (budget-friendly) or Pro (unlimited + 4K + AI captions)?

You:    why is everything very expensive
Infix:  I understand pricing matters. Basic starts at $29/month for new creators,
        while Pro at $79/month is built for teams that need unlimited videos,
        4K exports, and faster turnaround. Tell me your monthly volume and
        platform and I can suggest the best-value plan for you.

You:    i want to subscribe pro version
Infix:  Great choice with the Pro plan. To get you started, what's your name?

You:    Anish
Infix:  Thanks Anish! What's your email address?

You:    anish@gmail.com
Infix:  Perfect! Which platform do you create content for? (YouTube, Instagram, TikTok, or Twitch)

You:    youtube
        >> Lead captured successfully: Anish, anish@gmail.com, YouTube

Infix:  Perfect! I've captured your details for the Pro plan.
        Our team will reach out to anish@gmail.com shortly to get you started with AutoStream.
```

---

## ✨ Features

| # | Capability | How it works |
|---|---|---|
| 🧠 | **Intent Classification** | Rule-based NLP classifies every turn as `greeting`, `product_or_pricing`, or `high_intent_lead` before routing |
| 📚 | **RAG Knowledge Retrieval** | Pricing, features, and policies are read from `data/knowledge_base.json`; an LLM synthesises natural answers for open queries |
| 🤖 | **LLM-Augmented Responses** | `gpt-4o-mini` generates fluid, context-aware replies for uncategorised questions using the knowledge base as its only source |
| 🪤 | **Gated Lead Qualification** | Name → Email → Platform collected one at a time; the capture tool never fires until all three are confirmed |
| 🔧 | **Tool Execution** | `mock_lead_capture(name, email, platform)` is invoked only at the correct graph node, never prematurely |
| 🔁 | **Full Conversation Memory** | `AgentState` carries all message history and lead fields across every turn with no external store required |
| 💬 | **Objection Handling** | Pricing pushback and doubt signals are matched against a configurable `objection_library.json` |
| 🧩 | **Graceful LLM Fallback** | If no API key is set, the agent runs entirely on rule-based logic — no crashes, no degraded UX |

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
         User Input ──► │ classify_intent │
                        └────────┬────────┘
                                 │
              ┌──────────────────┼───────────────────┐
              │                  │                   │
              ▼                  ▼                   ▼
    ┌──────────────────┐ ┌───────────────────┐ ┌──────────────┐
    │ respond_greeting │ │ retrieve_knowledge│ │ qualify_lead │
    └────────┬─────────┘ └─────────┬─────────┘ └──────┬───────┘
             │                     │                   │
             │              (LLM or rule-based)   ┌────┴──────────────┐
             │                     │              │                   │
             ▼                     ▼              ▼                   ▼
            END                   END       execute_tool             END
                                                  │           (still collecting)
                                                  ▼
                                                 END
```

---

### Hybrid Intent & Response Strategy

```
User message
     │
     ▼
classify_intent  ──── regex + keyword rules ────► routing signal
     │
     ▼
retrieve_knowledge
     │
     ├── known pattern? (objection / compare / refund / support)
     │         └──► rule-based templated response  (fast, predictable)
     │
     └── open / uncategorised query?
               └──► LLM call (gpt-4o-mini)
                         ├── system: "answer using only knowledge context"
                         ├── user:   message + JSON knowledge dump
                         └──► natural language answer + CTA
```

The LLM is only invoked where rule-based logic would produce a generic or unhelpful answer. Structured flows (greeting, qualification, tool execution) remain fully deterministic.

---

### State Management

Every turn passes through one shared `AgentState` dict:

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
├── test_conversation.py           # Scripted end-to-end conversation harness
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
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini        # or gpt-4o, gpt-3.5-turbo
```

> **No API key?** Leave `OPENAI_API_KEY` blank. The agent will run fully on rule-based logic — all pricing, objection handling, and lead qualification still work. Only open-ended queries fall back to a static pricing snapshot instead of an LLM-generated answer.

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
| Intent — greeting | `classify_intent` regex: `hi\|hello\|hey` | ✅ |
| Intent — product / pricing | Keywords: `price`, `plan`, `refund`, `feature` | ✅ |
| Intent — high-intent lead | Buy signals: `subscribe`, `buy`, `sign up`, `interested` | ✅ |
| RAG from local knowledge base | JSON lookup + LLM synthesis in `retrieve_knowledge` | ✅ |
| Lead gated before tool call | `qualify_lead` collects all 3 fields before routing to `execute_tool` | ✅ |
| `mock_lead_capture()` tool | Fires inside `execute_tool` node only | ✅ |
| State retained across 5–6 turns | `AgentState` TypedDict persists across all graph node invocations | ✅ |
| Objection handling | Keyword-matched from `objection_library.json` | ✅ |
| LLM integration | `gpt-4o-mini` via `langchain-openai` in `retrieve_knowledge` | ✅ |
| LLM name extraction | `extract_name_with_llm()` called inside `qualify_lead` | ✅ |
| Graceful fallback (no API key) | Rule-based path used when `OPENAI_API_KEY` is absent | ✅ |
| Pinned dependencies | All versions locked in `requirements.txt` | ✅ |
| WhatsApp deployment plan | Documented above with working code sketch | ✅ |

---

<div align="center">
<br/>

Built by **Anish** for the **ServiceHive ML Internship Assignment**

*Infix · AutoStream · LangGraph*

</div>