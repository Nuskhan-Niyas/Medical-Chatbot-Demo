# main.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import os
import datetime

# ============================
# COMPONENT IMPORTS
# ============================
from app.components.llm import get_gemma_llm
from app.components.retriever import create_qa_chain
from app.components.vector_store import load_vector_store
from app.components.embeddings import get_embedding_model

# ============================
# FASTAPI APP SETUP
# ============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# STARTUP — LOAD ALL MODELS
# ============================
print("🔄 Initializing backend (LLM + Embeddings + Vectorstore + QA Chain)...")

# 1️⃣ Load Gemma LLM
llm = get_gemma_llm()

# 2️⃣ Load Embedding Model
embedding_model = get_embedding_model()

# 3️⃣ Load Vectorstore (no arguments!)
vectorstore = load_vector_store()

# 4️⃣ Initialize RAG QA Chain
qa_chain = create_qa_chain()

print("✅ Backend initialized successfully.")

# ============================
# STREAMING RESPONSE
# ============================
async def stream_answer_from_qa(user_query: str):
    """Stream answer from the QA chain in chunks."""
    result = qa_chain.invoke({"query": user_query})
    answer = result.get("result", "")

    chunk_size = 50
    for i in range(0, len(answer), chunk_size):
        yield f"data: {answer[i:i+chunk_size]}\n\n"
        await asyncio.sleep(0.05)

    yield "data: [DONE]\n\n"

@app.get("/chat/stream")
async def stream_chat(query: str):
    return StreamingResponse(
        stream_answer_from_qa(query),
        media_type="text/event-stream"
    )

# ============================
# NORMAL (NON-STREAMING) CHAT
# ============================
@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        query = body.get("query")

        if not query:
            return JSONResponse({"error": "No query provided"}, status_code=400)

        result = qa_chain.invoke({"query": query})
        return {"response": result.get("result", "")}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ============================
# CONVERSATION SAVE / LOAD
# ============================
CONV_DIR = "conversations"
os.makedirs(CONV_DIR, exist_ok=True)

@app.post("/conversation/save")
async def save_conversation(request: Request):
    data = await request.json()
    conversation = data.get("conversation")

    if not conversation:
        return {"success": False, "message": "No data provided"}

    filename = f"conversation_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    filepath = os.path.join(CONV_DIR, filename)

    with open(filepath, "w") as f:
        json.dump(conversation, f, indent=2)

    return {"success": True, "filename": filename}

@app.get("/conversation/list")
async def list_conversations():
    files = sorted(os.listdir(CONV_DIR))
    return {"files": files}

@app.get("/conversation/load/{filename}")
async def load_conversation(filename: str):
    filepath = os.path.join(CONV_DIR, filename)

    if not os.path.exists(filepath):
        return {"success": False, "message": "File not found"}

    with open(filepath, "r") as f:
        conversation = json.load(f)

    return {"success": True, "conversation": conversation}
