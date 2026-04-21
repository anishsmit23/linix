# Infix Conversational Agent

Infix is a lightweight conversational sales agent that sells AutoStream and can:

- Classifies user intent (greeting, product/pricing, high-intent lead)
- Handles objections and gives value-based recommendations
- Answers product questions from local JSON data (RAG-style lookup)
- Captures lead details (name, email, platform, plan) before tool execution
- Uses a mock lead-capture tool once qualification is complete

## Project Structure

- `agent.py`: Main LangGraph agent flow and CLI loop
- `data/knowledge_base.json`: Product, pricing, and policy data
- `data/objection_library.json`: Objection-handling templates and keywords
- `data/conversation_examples.json`: Conversation and intent examples
- `validate_data.py`: JSON structure and quality validator
- `test_conversation.py`: End-to-end conversation test harness

## Setup

```bash
pip install -r requirements.txt
```

## Run the Agent

```bash
python agent.py
```

## Validate Data Files

```bash
python validate_data.py
```

## Run Conversation Test

```bash
python test_conversation.py
```