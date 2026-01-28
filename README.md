# 🏏 IPL AI Analyst – RAG-based Cricket Intelligence System

An end-to-end **Retrieval-Augmented Generation (RAG)** system built on **TinyLlama + LoRA fine-tuning**, designed to answer analytical questions about **IPL 2022** using structured cricket data.

This project demonstrates:

- Practical LLM fine-tuning (LoRA, 4-bit quantization)
- Tool-augmented reasoning (calculator + retrieval)
- Real-world ML system design
- API, UI, and Kubernetes deployment readiness

---

## 🚀 Project Overview

**IPL AI Analyst** answers questions such as:

- *How many runs did Virat Kohli score in IPL 2022?*
- *What was Dwayne Bravo’s bowling performance?*
- *Show commentary for MI vs CSK match*

Instead of hallucinating, the system:

1. Retrieves facts from IPL JSON datasets  
2. Injects them into the LLM prompt  
3. Generates grounded, cricket-aware answers  

> This is a **tool-assisted LLM**, not a plain chatbot.

---

## 🧠 High-Level Architecture

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

---

## 🧩 Core Components

### 1️⃣ StatsTool (Calculator / Data Engine)

**File:** `agents/multi_agent.py`

**Responsibilities:**

- Loads IPL JSON datasets:
  - Batting stats
  - Bowling stats
  - Career stats
  - Matches
  - Teams
- Normalizes schemas
- Extracts numeric facts (runs, wickets, economy, matches)
- Fetches match commentary snippets

> This replaces unreliable LLM math with **deterministic computation**.

---

### 2️⃣ RetrieverAgent (Context Builder)

**Purpose:**  
Converts a natural-language query into relevant cricket facts.

**Key features:**

- Partial player name matching  
  *(e.g. “Bravo” → Dwayne Bravo)*
- Avoids false positives  
  *(e.g. Roy vs Royal)*
- Retrieves:
  - 2022 season stats
  - Career stats
  - Match commentary (if query contains `vs` or `match`)

**Output:**

---

### 3️⃣ AnalystAgent (LLM Brain)

**Base Model:**  
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`

**Enhancements:**

- 4-bit quantization (BitsAndBytes)
- LoRA adapter fine-tuned on IPL Q&A
- GPU-aware loading (CUDA / CPU fallback)

**Why TinyLlama?**

- Lightweight
- Fast inference
- Ideal for constrained environments
- Well-suited for tool-augmented reasoning

---

## 🏋️ Model Fine-Tuning (LoRA)

**File:** `train_sft.py`

**Training Strategy:**

- Supervised Fine-Tuning (SFT)
- JSON dataset with formatted cricket Q&A
- LoRA adapters only (base model frozen)
- 4-bit training for low VRAM usage

### Training Flow

1. Load base TinyLlama (4-bit)
2. Prepare for k-bit training
3. Apply LoRA on attention layers
4. Train for 1 epoch
5. Save adapter → `ipl_analyst_adapter/`

**Result:**  
A cricket-aware reasoning layer without full model retraining.

---

## 🌐 API Service (FastAPI)

**File:** `app.py`

### Endpoints

#### Health Check


Response:
```json
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

🧠 Design Notes
Models load once at application startup using FastAPI lifespan manager

Prevents cold-start latency on each request

Clear pipeline separation:

Retrieve (RAG)

Analyze

Generate

🖥️ Frontend (Streamlit)
File: frontend.py

Features
Chat-based interface

Real-time inference

Retrieval debugger panel

Cached model loading

⚠️ This UI is intended for demo and debugging, not production use.

🐳 Docker Support
File: Dockerfile

Details
Base image: Python 3.10 slim

Installs all dependencies

Runs FastAPI via Uvicorn

Exposes port 8000

Build
docker build -t ipl-analyst .
Run
docker run -p 8000:8000 ipl-analyst
☸️ Kubernetes Deployment (Helm)
Files
Chart.yaml

values.yaml

deployment.yaml

Features
GPU request (nvidia.com/gpu: 1)

Memory limits

Liveness & readiness probes

Scalable replicas

Demonstrates production awareness, even if not deployed.

