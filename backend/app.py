from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from airtable_client import AirtableClient
from llm_service import LLMService
from models import QuestionRequest
from dotenv import load_dotenv
from datetime import datetime
from fastapi.responses import FileResponse, Response
import os


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

# initialize services
airtable_client = AirtableClient()
llm_service = LLMService()

@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    try:
        # 添加请求调试日志
        print("Received question:", request.question)
        print("Received chat history:", request.chat_history)
        print("Full request object:", request.dict())
        
        answer = llm_service.get_answer(request.question, request.chat_history)
        # 添加 AI 回答的调试日志
        print("AI answer:", answer)
        
        return {"answer": answer}
    except Exception as e:
        print("Error occurred:", str(e))  # 添加错误日志
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reload_airtable")
async def reload_airtable():
    try:
        # airtable_client.delete_records_file()
        # airtable_client.fetch_all_records()
        llm_service.check_and_clean_vectorstore()
        success = llm_service.process_documents_to_vectorstore()
        if success:
            return {
                "status": "success",
                "message": "Successfully updated all documents to BetaGPT RAG",
                "updated_at": datetime.now().isoformat(),
            }
        else:
            raise HTTPException(status_code=500,
                                detail="Failed to update documents to BetaGPT RAG, error is: " + str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update documents, error is: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Server is running"}

@app.get("/favicon.ico")
async def favicon():
    return Response(
        content=b'',  # 返回空内容
        media_type="image/x-icon"
    )
