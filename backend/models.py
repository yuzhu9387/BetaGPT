from pydantic import BaseModel
from typing import List, Tuple, Optional

class ChatMessage(BaseModel):
    user_message: str
    ai_response: str

class QuestionRequest(BaseModel):
    question: str
    chat_history: List[ChatMessage] = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "什么是BetaGPT?",
                "chat_history": [
                    {
                        "user_message": "你好",
                        "ai_response": "你好！有什么我可以帮你的吗？"
                    }
                ]
            }
        } 