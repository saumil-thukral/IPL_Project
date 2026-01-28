🏏 IPL AI Analyst – RAG-based Cricket Intelligence System

An end-to-end Retrieval-Augmented Generation (RAG) system built on TinyLlama + LoRA fine-tuning, designed to answer analytical questions about IPL 2022 using structured cricket data.

This project demonstrates:

Practical LLM fine-tuning (LoRA, 4-bit quantization)

Tool-augmented reasoning (calculator + retrieval)

Real-world ML system design

API + UI + Kubernetes deployment readiness

🚀 Project Overview

IPL AI Analyst answers questions like:

“How many runs did Virat Kohli score in IPL 2022?”

“What was Bravo’s bowling performance?”

“Show commentary for MI vs CSK match”

Instead of hallucinating, the model:

Retrieves facts from IPL JSON datasets

Injects them into the LLM prompt

Generates grounded, cricket-aware answers

This is a tool-assisted LLM, not a plain chatbot.

🧠 Architecture (High Level)
User Query
   ↓
RetrieverAgent
   ↓
StatsTool (Structured IPL Data)
   ↓
Context (Facts)
   ↓
AnalystAgent (TinyLlama + LoRA)
   ↓
Final Answer

🧩 Core Components
1️⃣ StatsTool (Calculator / Data Engine)

File: agents/multi_agent.py

Responsibilities:

Loads IPL JSON datasets:

Batting stats

Bowling stats

Career stats

Matches

Teams

Normalizes schemas

Extracts numeric facts (runs, wickets, economy, matches)

Fetches match commentary snippets

This replaces unreliable LLM math with deterministic computation.

2️⃣ RetrieverAgent (Context Builder)

Purpose:
Converts a natural-language query into relevant cricket facts

Key features:

Partial player name matching (e.g. “Bravo” → Dwayne Bravo)

Avoids false positives (e.g. Roy vs Royal)

Retrieves:

2022 season stats

Career stats

Match commentary (if query contains vs or match)

Output:

Structured, factual context for the LLM

3️⃣ AnalystAgent (LLM Brain)

Model:
TinyLlama/TinyLlama-1.1B-Chat-v1.0

Enhancements:

4-bit quantization (BitsAndBytes)

LoRA adapter fine-tuned on IPL Q&A

GPU-aware loading (CUDA / CPU fallback)

Why TinyLlama?

Lightweight

Fast inference

Ideal for constrained environments

Perfect for tool-augmented reasoning

🏋️ Model Fine-Tuning (LoRA)

File: train_sft.py

Training strategy:

Supervised Fine-Tuning (SFT)

JSON dataset with formatted cricket Q&A

LoRA adapters only (base model frozen)

4-bit training for low VRAM usage

Training Flow

Load base TinyLlama (4-bit)

Prepare for k-bit training

Apply LoRA on attention layers

Train for 1 epoch

Save adapter → ipl_analyst_adapter/

Result:
A cricket-aware reasoning layer without full model retraining.

🌐 API Service (FastAPI)

File: app.py

Endpoints
Health Check
GET /


Response:

{
  "status": "active",
  "message": "IPL AI Analyst is running."
}

Ask a Question
POST /ask


Request:

{
  "query": "How many runs did Kohli score?"
}


Response:

{
  "query": "...",
  "answer": "...",
  "context_used": "..."
}

Design Notes

Models load once at startup (lifespan manager)

Prevents cold-start latency per request

Clean separation: retrieve → generate

🖥️ Frontend (Streamlit)

File: frontend.py

Features:

Chat interface

Real-time inference

Retrieval debugger panel

Cached model loading

This UI is for demo & debugging, not production.

🐳 Docker Support

File: Dockerfile

Python 3.10 slim

Installs dependencies

Runs FastAPI via Uvicorn

Exposes port 8000

Build:

docker build -t ipl-analyst .


Run:

docker run -p 8000:8000 ipl-analyst

☸️ Kubernetes Deployment (Helm)

Files:

Chart.yaml

values.yaml

deployment.yaml

Features:

GPU request (nvidia.com/gpu: 1)

Memory limits

Health probes

Scalable replicas

This shows production awareness, even if not deployed.

📁 Project Structure
.
├── agents/
│   └── multi_agent.py
├── ipl_analyst_adapter/
│   └── adapter_model.safetensors
├── train_sft.py
├── app.py
├── frontend.py
├── Dockerfile
├── Chart.yaml
├── deployment.yaml
├── values.yaml
└── requirements.txt

🧪 Example Questions

How many runs did Rohit Sharma score in 2022?

What was Bumrah’s economy rate?

How many wickets did Bravo take?

Show commentary for CSK vs MI match

What are Kohli’s career stats?

🎯 Why This Design?

LLM ≠ calculator → StatsTool handles numbers

RAG reduces hallucinations

LoRA saves compute

No heavy frameworks (LangChain avoided)

Clear separation of concerns

This mirrors how real production AI systems are built.

✅ What This Project Demonstrates

✔ LLM fine-tuning
✔ RAG architecture
✔ Tool-augmented reasoning
✔ API deployment
✔ GPU-aware inference
✔ Clean system design

👤 Author

BCA Final Year Student
AI/ML Internship Technical Assessment Project
