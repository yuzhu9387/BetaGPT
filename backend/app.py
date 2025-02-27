from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from airtable_client import AirtableClient
from llm_service import LLMService
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()
app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 初始化服务
airtable_client = AirtableClient()
documents = airtable_client.fetch_all_records()
llm_service = LLMService(documents)

class QuestionRequest(BaseModel):
    question: str
    chat_history: list = []

@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    try:
        answer = llm_service.get_answer(request.question, request.chat_history)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )