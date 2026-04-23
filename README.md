# Infix — Social-to-Lead Agentic Workflow

<div align="center">
██╗███╗   ██╗███████╗██╗██╗  ██╗
██║████╗  ██║██╔════╝██║╚██╗██╔╝
██║██╔██╗ ██║█████╗  ██║ ╚███╔╝ 
██║██║╚██╗██║██╔══╝  ██║ ██╔██╗ 
██║██║ ╚████║██║     ██║██╔╝ ██╗
╚═╝╚═╝  ╚═══╝╚═╝     ╚═╝╚═╝  ╚═╝

> An AI-powered conversational sales agent that turns social media conversations into qualified business leads — built for **ServiceHive** as part of the **Inflx** platform.
<br/>
---
</div>

---
## 📌 What is Infix?

**Infix** is a LangGraph-powered agentic AI built for **AutoStream**, a SaaS product offering automated video editing tools for content creators. It's not a simple chatbot — it's a stateful, multi-node reasoning agent that understands where a user is in the sales funnel and responds accordingly.

The agent can:

- 🧠 **Classify user intent** into greeting / product inquiry / high-intent lead using an LLM
- 📚 **Answer product & pricing questions** using a RAG pipeline over a local JSON knowledge base
- 🛡️ **Handle objections** empathetically using a structured objection library
- 🎯 **Detect high-intent signals** and collect lead details one field at a time (name → email → platform)
- ⚙️ **Trigger `mock_lead_capture()`** only once all three fields are confirmed — never prematurely
- 🔁 **Retry across Gemini model fallbacks** automatically on quota/timeout errors

---

## 🗂️ Project Structure

```
linix/
├── src/
│   ├── agent.py                    # Full LangGraph agent — all nodes, graph wiring, CLI
│   ├── test_conversation.py        # End-to-end scripted 5-turn test with assertions
│   └── validate_data.py            # Validates knowledge base
├── data/
│   ├── knowledge_base.json         # Product info, pricing (Basic/Pro/Enterprise), policies
│   ├── objection_library.json      # Objection keywords + reframe responses
│   └── conversation_examples.json  # Few-shot examples 
├── .env.example                    # Environment variable template
├── install.sh                      # One-command dependency installer
├── requirements.txt                # Pinned Python dependencies
└── README.md
```

---

## ⚡ Setup — Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/anishsmit23/linix.git
cd linix
```

### 2. Create & Activate a Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or use the provided installer script:

```bash
bash install.sh
```

**Pinned dependencies:**

```
langchain==1.2.15
langchain-google-genai==4.2.2
langgraph==1.1.9
python-dotenv==1.2.2
typing-extensions==4.15.0
```

### 4. Configure Environment Variables

Copy `.env.example` and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here
ENABLE_LLM=true

# Optional — defaults shown
LLM_MODEL=gemini-flash-latest
LLM_FALLBACK_MODELS=gemini-flash-latest,gemini-2.0-flash,gemini-2.5-flash
LLM_TIMEOUT_SECONDS=25
LLM_MAX_RETRIES=2
LLM_ATTEMPTS_PER_MODEL=2
LLM_RETRY_BACKOFF_SECONDS=0.75
```

> **Note:** Both `ENABLE_LLM=true` and `GOOGLE_API_KEY` are mandatory. The agent exits with a clear error on startup if either is missing.

### 5. Run the Agent

```bash
cd src
python agent.py
```

```
Infix Agent (type 'quit' to exit)

You:
```

Type `quit` or `exit` to end the session.

### 6. Run the End-to-End Test

```bash
cd src
python test_conversation.py
```

This runs a scripted 5-turn conversation and asserts that name, email, and platform are all correctly captured and `mock_lead_capture()` is invoked exactly once.

---

## 💬 Example Conversation

```
You: Hi, tell me about your pricing
Agent: Hello! I am Infix, and I can help you pick the right AutoStream plan...
       Basic is $29/month for 10 videos at 720p. Pro is $79/month with
       unlimited 4K videos and AI captions. Which sounds like the best fit?

You: This sounds expensive
Agent: I understand pricing is an important consideration. Basic starts at
       $29/month — think of it as getting back hours of editing time every
       week. Would you like to try the Basic plan today?

You: Okay, I want to start with Pro for my YouTube channel
Agent: Great choice with the Pro plan. To get you started, what's your name?

You: John Doe
Agent: Thanks John Doe! What's your email address?

You: john@example.com
Lead captured successfully: John Doe, john@example.com, YouTube
Agent: Perfect! I've captured your details for the Pro plan. Our team will
       reach out to john@example.com shortly to get you started with AutoStream.
```

---

## 🧱 Architecture

### Agent Graph

```
                     ┌────────────────────┐
   User Input ──────►│   classify_intent  │
                     └──────────┬─────────┘
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                    │
           ▼                    ▼                    ▼
  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────┐
  │respond_greeting │  │retrieveknowledge│  │   qualify_lead    │
  │                 │  │   (RAG + LLM)   │  │ name→email→platf. │
  └────────┬────────┘  └────────┬────────┘  └────────┬──────────┘
           │                    │                    │
           │                    │        all fields  │  field
           │                    │        collected   │  missing
           ▼                    ▼             ▼      ▼
          END                  END    ┌───────────────────┐
                                      │   execute_tool    │
                                      │mock_lead_capture()│
                                      └────────┬──────────└
                                               │
                                              END
```

### Nodes

| Node | Responsibility |
|---|---|
| `classify_intent` | LLM classifies each message as `greeting`, `product_or_pricing`, or `high_intent_lead`. Skips reclassification if `qualification_in_progress` is already `True`. |
| `respond_greeting` | Returns a contextual greeting. Personalises the response if platform or name is already known from the message. |
| `retrieve_knowledge` | LLM answers pricing / policy / objection questions with the full `knowledge_base.json` + `objection_library.json` injected as RAG context. Increments `objection_count` on price pushback. |
| `qualify_lead` | Collects name → email → platform one field at a time using regex, loose heuristic, and LLM extraction as a final fallback. Routes to `execute_tool` only when all three fields are non-null. |
| `execute_tool` | Calls `mock_lead_capture(name, email, platform)`, sets `lead_captured = True`, and guards against double-execution. |

---

## 🔍 Why LangGraph? How State Is Managed

### The Problem with Simple Chains

A standard `LLMChain` or `ConversationChain` treats every turn identically — it can't conditionally route to different logic, gate a tool call behind collected fields, or maintain a typed data structure across turns. For a sales agent that must move through distinct stages (discovery → consideration → qualification → closing), this breaks down fast.

### Why LangGraph Solves This

LangGraph models the agent as a **directed state graph** where each node is a pure function that receives the full state and returns a partial update. This gives the agent:

**Conditional routing** — `classify_intent` sets a `next_step` string, and `add_conditional_edges` routes to the correct node each turn without any if-else logic scattered across the codebase.

**Typed, persistent state** — the entire conversation is held in a single `AgentState` TypedDict:

```python
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # full history, append-only
    current_intent: Optional[str]
    sales_stage: Optional[str]               # discovery → consideration → closing → won
    qualification_in_progress: bool
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_plan: Optional[str]
    objection_count: int
    lead_captured: bool
    next_step: str
```

This single object is passed into every node and merged after each turn. `operator.add` on `messages` ensures history is appended, never overwritten. No external memory store, Redis, or manual buffer management is needed — LangGraph handles the merge natively.

**Mid-flow persistence** — when `qualification_in_progress = True`, `classify_intent` routes directly to `qualify_lead` without calling the LLM again. This prevents the model from misclassifying "John Doe" as a greeting mid-collection.

**Tool gating** — `mock_lead_capture()` lives inside `execute_tool`, which is only reachable via an edge condition inside `qualify_lead` once all three of `lead_name`, `lead_email`, and `lead_platform` are confirmed non-null. It cannot fire prematurely regardless of what the user says.

**Resilience** — `_invoke_llm_safely()` automatically retries across the configured Gemini fallback model list (`gemini-flash-latest → gemini-2.0-flash → gemini-2.5-flash`) with exponential backoff, so transient quota or timeout errors don't crash the agent.

---

## 📱 WhatsApp Deployment via Webhooks

### Architecture

```
WhatsApp User
      │
      ▼
WhatsApp Business API (Meta Cloud API)
      │  POST /webhook  (every incoming message)
      ▼
FastAPI Webhook Server
  • Verifies Meta token
  • Parses message + extracts sender ID
  • Loads per-user AgentState from Redis
      │
      ▼
Infix LangGraph Agent  ──  graph.invoke(state)
      │
      ▼
Redis Session Store
  Key:   session:{whatsapp_id}
  Value: AgentState JSON  (TTL: 1 hour)
      │
      ▼
WhatsApp Send API  →  reply delivered to user
```

### Step 1 — Create a Meta WhatsApp Business App

1. Go to [developers.facebook.com](https://developers.facebook.com) and create an app
2. Add the **WhatsApp** product
3. Note your `WHATSAPP_TOKEN` and `PHONE_NUMBER_ID`

### Step 2 — Webhook Server (FastAPI)

```python
from fastapi import FastAPI, Request
import httpx, json, os, redis as redis_lib
from agent import build_graph, new_state
from langchain_core.messages import HumanMessage

app = FastAPI()
graph = build_graph()
r = redis_lib.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()
    entry = data["entry"][0]["changes"][0]["value"]
    msg = entry["messages"][0]
    user_id = msg["from"]
    user_text = msg["text"]["body"]

    # Load existing session or start fresh
    raw = r.get(f"session:{user_id}")
    state = json.loads(raw) if raw else new_state()

    state["messages"].append(HumanMessage(content=user_text))
    state["next_step"] = ""

    result = graph.invoke(state)

    # Persist updated state with 1-hour TTL
    r.setex(f"session:{user_id}", 3600, json.dumps(result, default=str))

    reply = result["messages"][-1].content
    await send_whatsapp_message(user_id, reply)
    return {"status": "ok"}

@app.get("/webhook")
async def verify(request: Request):
    params = dict(request.query_params)
    if params.get("hub.verify_token") == os.getenv("WHATSAPP_VERIFY_TOKEN"):
        return int(params["hub.challenge"])
    return {"error": "Forbidden"}, 403
```

### Step 3 — Send Replies

```python
async def send_whatsapp_message(to: str, text: str):
    url = f"https://graph.facebook.com/v18.0/{os.getenv('PHONE_NUMBER_ID')}/messages"
    headers = {"Authorization": f"Bearer {os.getenv('WHATSAPP_TOKEN')}"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload, headers=headers)
```

### Step 4 — Test Locally with ngrok

```bash
uvicorn webhook:app --reload --port 8000

# In a separate terminal
ngrok http 8000
# Paste the HTTPS URL into the Meta Developer Dashboard as your webhook URL
```

### Step 5 — Production Deployment

Deploy to **Railway**, **Render**, or **AWS Lambda** and set these environment variables:

```
GOOGLE_API_KEY
ENABLE_LLM=true
WHATSAPP_TOKEN
PHONE_NUMBER_ID
WHATSAPP_VERIFY_TOKEN
REDIS_URL
```

### Key Considerations

| Concern | Solution |
|---|---|
| One session per user | Redis keyed by `whatsapp_id` |
| Session expiry | Redis TTL of 1 hour (configurable) |
| Opt-out / STOP | Detect "STOP" keyword → `r.delete(f"session:{user_id}")` |
| Media messages | Check `msg["type"]` — handle `audio` / `image` separately from `text` |
| Rate limits | Meta allows ~80 msg/sec; FastAPI async handles this without batching |
| Concurrent users | Each user has an independent Redis key — no cross-contamination |

---

## ✅ Evaluation Checklist

| Criterion | Implementation |
|---|---|
| Intent detection | LLM-only 3-class classifier (`greeting` / `product_or_pricing` / `high_intent_lead`) with strict JSON output |
| RAG knowledge retrieval | Full `knowledge_base.json` + `objection_library.json` injected as context into every `retrieve_knowledge` call |
| State management | LangGraph `StateGraph` with typed `AgentState` — persists natively across all turns without external storage |
| Tool calling logic | `mock_lead_capture()` gated by edge condition in `qualify_lead` — fires only when name + email + platform are all confirmed |
| Code clarity | Each concern isolated in its own node; retry/fallback logic encapsulated in `_invoke_llm_safely()` |
| Deployability | FastAPI + Redis session store maps directly onto the WhatsApp webhook architecture |

---

## 👤 Author

**Anish** — Machine Learning Intern Assignment  
Built for **ServiceHive × Inflx**  
GitHub: [anishsmit23/linix](https://github.com/anishsmit23/linix)
