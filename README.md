# 🤖 Infix — AI Conversational Sales Agent

<p align="center">

**LangGraph-powered AI agent that converts conversations into qualified leads**

Built for **ServiceHive × Inflx Platform**

</p>

---

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20AI-black)
![Gemini](https://img.shields.io/badge/Gemini-LLM-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

</p>

---

# 📌 What is Infix?

**Infix** is a **LangGraph-powered agentic AI system** designed to convert social media conversations into qualified business leads.

Unlike traditional chatbots, Infix:

* Maintains **persistent conversation state**
* Routes logic using a **multi-node reasoning graph**
* Handles **objections intelligently**
* Collects **structured lead data**
* Triggers actions only when **all requirements are met**

Built for:

**AutoStream — SaaS video automation platform**

---

# 🧠 Key Capabilities

✔ Intent classification using LLM
✔ RAG-powered knowledge retrieval
✔ Stateful lead qualification
✔ Objection handling logic
✔ Structured data capture
✔ Multi-model fallback resilience
✔ Tool-gated execution logic

---

# 🗂️ Project Structure

```bash
linix/
├── src/
│   ├── agent.py
│   ├── test_conversation.py
│   └── validate_data.py
│
├── data/
│   ├── knowledge_base.json
│   ├── objection_library.json
│   └── conversation_examples.json
│
├── .env.example
├── install.sh
├── requirements.txt
└── README.md
```

---

# ⚡ Setup — Run Locally

## 1️⃣ Clone Repository

```bash
git clone https://github.com/anishsmit23/linix.git
cd linix
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

OR

```bash
bash install.sh
```

Pinned Dependencies:

```
langchain==1.2.15
langchain-google-genai==4.2.2
langgraph==1.1.9
python-dotenv==1.2.2
typing-extensions==4.15.0
```

---

## 4️⃣ Configure Environment Variables

```bash
cp .env.example .env
```

Edit:

```env
GOOGLE_API_KEY=your_google_api_key_here
ENABLE_LLM=true

LLM_MODEL=gemini-flash-latest
LLM_FALLBACK_MODELS=gemini-flash-latest,gemini-2.0-flash,gemini-2.5-flash
LLM_TIMEOUT_SECONDS=25
LLM_MAX_RETRIES=2
LLM_ATTEMPTS_PER_MODEL=2
LLM_RETRY_BACKOFF_SECONDS=0.75
```

---

## 5️⃣ Run the Agent

```bash
cd src
python agent.py
```

Output:

```
Infix Agent (type 'quit' to exit)

You:
```

---

## 6️⃣ Run End-to-End Test

```bash
cd src
python test_conversation.py
```

This verifies:

✔ Lead collection
✔ Tool execution
✔ Conversation flow

---

# 💬 Example Conversation

```
You: Hi, tell me about pricing

Agent: Hello! I am Infix...
Basic is $29/month for 10 videos.
Pro is $79/month with unlimited 4K videos.

You: This sounds expensive

Agent: I understand pricing concerns...

You: Okay I want Pro

Agent: Great! What's your name?

You: John Doe

Agent: What's your email?

You: john@example.com

Lead captured successfully.
```

---

# 🧱 System Architecture

## Agent Flow

```
User Input
      │
      ▼
classify_intent
      │
 ┌────────────┼──────────────┐
 ▼            ▼              ▼
Greeting   Knowledge   Qualification
                             │
                             ▼
                        Execute Tool
                             │
                             ▼
                            END
```

---

# 🔍 Why LangGraph?

Traditional LLM chains:

❌ No routing
❌ No persistence
❌ No tool gating

LangGraph enables:

✔ Conditional execution
✔ Stateful reasoning
✔ Multi-stage workflows
✔ Controlled tool usage

---

# 🧠 Agent State Model

```python
class AgentState(TypedDict):

    messages: list
    current_intent: Optional[str]

    qualification_in_progress: bool

    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]

    objection_count: int
    lead_captured: bool

    next_step: str
```

---

# 📱 WhatsApp Deployment Architecture

```
WhatsApp User
        │
        ▼
WhatsApp Cloud API
        │
        ▼
FastAPI Webhook
        │
        ▼
LangGraph Agent
        │
        ▼
Redis Session Store
        │
        ▼
Send Response
```

---

# 🚀 Production Deployment

Recommended platforms:

* Railway
* Render
* AWS Lambda

Required Environment Variables:

```
GOOGLE_API_KEY
ENABLE_LLM=true
WHATSAPP_TOKEN
PHONE_NUMBER_ID
WHATSAPP_VERIFY_TOKEN
REDIS_URL
```

---

# ✅ Evaluation Checklist

| Feature          | Implemented |
| ---------------- | ----------- |
| Intent detection | ✅           |
| RAG retrieval    | ✅           |
| Stateful memory  | ✅           |
| Tool gating      | ✅           |
| Retry logic      | ✅           |
| Deployment-ready | ✅           |

---

# 👤 Author

**Anish**
Machine Learning Intern Assignment

Built for:

**ServiceHive × Inflx**

GitHub:

https://github.com/anishsmit23/linix


LinkedIn:

https://www.linkedin.com/in/anish55