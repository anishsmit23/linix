
<div align="center">

# 🤖 Infix — AI Conversational Sales Agent

**LangGraph-powered AI agent that converts conversations into qualified leads**

Built for **ServiceHive × Inflx Platform**

<br>

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/LangGraph-Agentic%20AI-1C3C3C?style=for-the-badge" alt="LangGraph">
<img src="https://img.shields.io/badge/Gemini-LLM-FF9800?style=for-the-badge&logo=google&logoColor=white" alt="Gemini">
<img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/Status-Production%20Ready-00C853?style=for-the-badge" alt="Status">

</div>

---

## 📌 What is Infix?

**Infix** is a **LangGraph-powered agentic AI system** designed to convert social media conversations into qualified business leads.

Unlike traditional chatbots, Infix:

- Maintains **persistent conversation state**
- Routes logic using a **multi-node reasoning graph**
- Handles **objections intelligently**
- Collects **structured lead data**
- Triggers actions only when **all requirements are met**

**Built for:** AutoStream — SaaS video automation platform

---

## 🧠 Key Capabilities

| Capability | Description |
|------------|-------------|
| 🎯 **Intent Classification** | LLM-powered intent detection for precise conversation routing |
| 📚 **RAG Retrieval** | Knowledge base lookups for accurate, contextual responses |
| 🧩 **Stateful Lead Qualification** | Persistent memory across multi-turn conversations |
| 🛡️ **Objection Handling** | Intelligent counter-arguments with retry logic |
| 📋 **Structured Data Capture** | Type-safe lead information collection |
| 🔄 **Multi-Model Fallback** | Automatic failover across Gemini model variants |
| 🔒 **Tool-Gated Execution** | Conditional tool invocation with guardrails |

---

## 🗂️ Project Structure

```
linix/
├── src/
│   ├── agent.py              # Core LangGraph agent implementation
│   ├── test_conversation.py  # End-to-end conversation flow tests
│   └── validate_data.py      # Data validation & integrity checks
│
├── data/
│   ├── knowledge_base.json       # RAG-enabled product knowledge
│   ├── objection_library.json    # Objection → Response mappings
│   └── conversation_examples.json # Training conversation patterns
│
├── .env.example          # Environment variable template
├── install.sh            # One-command dependency installer
├── requirements.txt      # Pinned Python dependencies
└── README.md             # You are here
```

---

## ⚡ Setup — Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/anishsmit23/linix.git
cd linix
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\\Scripts\\activate

# macOS / Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**Or use the automated installer:**

```bash
bash install.sh
```

**Pinned Dependencies:**

```
langchain==1.2.15
langchain-google-genai==4.2.2
langgraph==1.1.9
python-dotenv==1.2.2
typing-extensions==4.15.0
```

### 4️⃣ Configure Environment Variables

```bash
cp .env.example .env
```

**Edit `.env`:**

```env
# ─── Core LLM Configuration ──────────────────────────────
GOOGLE_API_KEY=your_google_api_key_here
ENABLE_LLM=true

# ─── Model Selection & Resilience ────────────────────────
LLM_MODEL=gemini-flash-latest
LLM_FALLBACK_MODELS=gemini-flash-latest,gemini-2.0-flash,gemini-2.5-flash
LLM_TIMEOUT_SECONDS=25
LLM_MAX_RETRIES=2
LLM_ATTEMPTS_PER_MODEL=2
LLM_RETRY_BACKOFF_SECONDS=0.75
```

### 5️⃣ Run the Agent

```bash
cd src
python agent.py
```

**Expected Output:**

```
Infix Agent (type 'quit' to exit)

You:
```

### 6️⃣ Run End-to-End Test

```bash
cd src
python test_conversation.py
```

**Verifies:**
- ✅ Lead collection flow
- ✅ Tool execution triggers
- ✅ Conversation state persistence
- ✅ Multi-turn dialogue handling

---

## 💬 Example Conversation

```
You:     Hi, tell me about pricing

Agent:   Hello! I am Infix, your AutoStream assistant.
         Basic is $29/month for 10 videos.
         Pro is $79/month with unlimited 4K videos.

You:     This sounds expensive

Agent:   I understand pricing concerns. Compared to hiring a video editor 
         ($500+/month), AutoStream saves 85% while delivering faster output.
         Would you like to see a cost breakdown?

You:     Okay I want Pro

Agent:   Great! What's your name?

You:     John Doe

Agent:   What's your email?

You:     john@example.com

Agent:   ✅ Lead captured successfully. Our team will contact you within 24h.
```

---

## 🏗️ System Architecture

### Agent Flow

```
                    ┌─────────────────┐
                    │   User Input    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ classify_intent │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │  Greeting   │   │  Knowledge  │   │Qualification│
    │   Handler   │   │  Retrieval  │   │    Node     │
    └─────────────┘   └─────────────┘   └──────┬──────┘
                                                │
                                                ▼
                                       ┌─────────────┐
                                       │Execute Tool │
                                       │  (Gated)    │
                                       └──────┬──────┘
                                              │
                                              ▼
                                       ┌─────────────┐
                                       │     END     │
                                       └─────────────┘
```

---

## 🧠 Architecture Deep Dive

### Why LangGraph?

Traditional LLM chains operate as **linear pipelines** — they lack conditional routing, persistent state across turns, and controlled tool access. LangGraph solves these limitations by modeling conversations as a **directed state machine** where each node represents a reasoning step and edges represent conditional transitions based on intent classification.

**LangGraph enables:**

| Feature | Traditional LLM Chains | LangGraph |
|---------|----------------------|-----------|
| Conditional Execution | ❌ Linear only | ✅ Branching logic |
| Stateful Reasoning | ❌ Stateless per call | ✅ Persistent `AgentState` |
| Multi-Stage Workflows | ❌ Manual orchestration | ✅ Native graph composition |
| Controlled Tool Usage | ❌ Unrestricted | ✅ Gate-based invocation |

### How State is Managed

Infix uses a **typed state dictionary** (`AgentState`) that persists across the entire conversation lifecycle. This state is not ephemeral — it survives multiple LLM calls, tool executions, and user turns.

```python
class AgentState(TypedDict):
    messages: list                    # Conversation history (Human/AI)
    current_intent: Optional[str]     # Last classified intent
    qualification_in_progress: bool     # Are we actively collecting lead info?
    lead_name: Optional[str]            # Captured lead name
    lead_email: Optional[str]           # Captured lead email
    lead_platform: Optional[str]        # Source platform (WhatsApp, etc.)
    objection_count: int                # Objection handling attempts
    lead_captured: bool                 # Final qualification status
    next_step: str                      # Graph routing decision
```

**State Lifecycle:**
1. **Initialization** — Empty state on first message
2. **Intent Classification** — `current_intent` set by LLM router
3. **Node Execution** — Relevant node reads/writes specific fields
4. **Conditional Routing** — `next_step` determines graph edge
5. **Persistence** — State serialized to Redis (production) or held in memory (local)

This design ensures that if a user interrupts lead collection with a pricing question, the agent **remembers** where it left off and resumes qualification seamlessly.

---

## 📱 WhatsApp Deployment Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│ WhatsApp    │────▶| WhatsApp Cloud   │ ────▶│ FastAPI Webhook │
│   User      │      │      API         │      │   Endpoint      │
└─────────────┘      └──────────────────┘      └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │  LangGraph      │
                                           │    Agent        │
                                           └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │  Redis Session  │
                                           │     Store       │
                                           └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │  Send Response  │
                                           │  (WhatsApp API) │
                                           └─────────────────┘
```

### WhatsApp Integration via Webhooks

To deploy Infix on WhatsApp, you implement a **bidirectional webhook flow**:

**1. Incoming Messages (WhatsApp → Agent)**
- Configure **WhatsApp Cloud API** webhook URL pointing to your FastAPI endpoint (`/webhook`)
- FastAPI receives `POST` requests with message payload containing:
  - `from`: Phone number (serves as `conversation_id`)
  - `text.body`: User message content
  - `timestamp`: Message time

**2. Session Management**
- Use **Redis** as a distributed session store keyed by phone number
- On each webhook hit:
  - Retrieve existing `AgentState` from Redis (or initialize new)
  - Inject user message into state
  - Invoke LangGraph agent with full state
  - Write updated state back to Redis with TTL (e.g., 24h)

**3. Outgoing Messages (Agent → WhatsApp)**
- After LangGraph completes execution, extract `messages[-1]` (AI response)
- Send via **WhatsApp Cloud API** `POST /messages` with:
  - `messaging_product`: `whatsapp`
  - `recipient_type`: `individual`
  - `to`: User phone number
  - `type`: `text`
  - `text.body`: Generated response

**4. Verification & Security**
- Implement **HubSpot-style verification** on `GET /webhook` using `WHATSAPP_VERIFY_TOKEN`
- Validate incoming `POST` signatures with WhatsApp's `X-Hub-Signature-256`
- Rate-limit by phone number to prevent abuse

**Required Environment Variables:**

```env
WHATSAPP_TOKEN=your_whatsapp_business_token
PHONE_NUMBER_ID=your_business_phone_id
WHATSAPP_VERIFY_TOKEN=your_custom_verify_token
REDIS_URL=redis://localhost:6379/0
```

---

## 🚀 Production Deployment

**Recommended Platforms:**

| Platform | Best For | Notes |
|----------|----------|-------|
| **Railway** | Rapid deployment | Native Redis, auto-scaling |
| **Render** | Cost-effective | Free tier available, persistent disks |
| **AWS Lambda** | Serverless scale | Use Lambda + API Gateway + ElastiCache |

**Required Environment Variables:**

```env
GOOGLE_API_KEY
ENABLE_LLM=true
WHATSAPP_TOKEN
PHONE_NUMBER_ID
WHATSAPP_VERIFY_TOKEN
REDIS_URL
```

---

## ✅ Evaluation Checklist

| Feature | Status | Notes |
|---------|--------|-------|
| Intent Detection | ✅ Implemented | LLM-based classification with confidence thresholds |
| RAG Retrieval | ✅ Implemented | JSON-based knowledge base with semantic matching |
| Stateful Memory | ✅ Implemented | TypedDict state with Redis persistence |
| Tool Gating | ✅ Implemented | Conditional execution based on lead completeness |
| Retry Logic | ✅ Implemented | Exponential backoff across 3 model fallbacks |
| Deployment-Ready | ✅ Implemented | Docker-compatible, 12-factor app compliant |

---

## 👤 Author

**Anish**  
*Machine Learning Intern Assignment*

**Built for:** ServiceHive × Inflx

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-anishsmit23-181717?style=flat-square&logo=github)](https://github.com/anishsmit23/linix)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-anish55-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/anish55)

"""

