from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from contextlib import asynccontextmanager

# Import your custom agents
from agents.multi_agent import StatsTool, RetrieverAgent, AnalystAgent

# --- Global Variables to hold loaded models ---
tool = None
retriever = None
analyst = None

# --- Lifespan Manager (Loads models on startup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tool, retriever, analyst
    
    # 1. Define Data Path (Update this if your path changes!)
    DATA_PATH = '/content/drive/MyDrive/AI_ML_Engineer_Tech_Test_Package/AI_ML_Engineer_Tech_Test_Package 2/data/Indian_Premier_League_2022-03-26/Indian_Premier_League_2022-03-26'
    
    print("🚀 Server Starting: Initializing AI Agents...")
    
    # 2. Load Tools
    try:
        tool = StatsTool(DATA_PATH)
        retriever = RetrieverAgent(tool)
        analyst = AnalystAgent() # Loads the LLM + Adapter
        print("✅ System Ready!")
    except Exception as e:
        print(f"❌ Critical Error Loading System: {e}")
    
    yield
    
    # Clean up (optional)
    print("🛑 Server Shutting Down")

# --- API Definition ---
app = FastAPI(title="IPL AI Analyst", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    context_used: str

@app.get("/")
def health_check():
    return {"status": "active", "message": "IPL AI Analyst is running."}

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not analyst:
        raise HTTPException(status_code=503, detail="System is still loading...")
    
    print(f"📩 Received: {request.query}")
    
    # 1. Retrieve
    context = retriever.retrieve(request.query)
    
    # 2. Generate
    answer = analyst.generate_answer(request.query, context)
    
    return {
        "query": request.query,
        "answer": answer,
        "context_used": context
    }
