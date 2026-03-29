import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
from src.rag_chain import ask_query


app = FastAPI(
    title="Medical RAG API",
    description="Reterival Augemented Generation over MedQuad Medical dataset",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str

class SourceChunk(BaseModel):
    text: str
    metadata: dict
    distance: float

class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceChunk]



@app.get("/health")
def health_check():
    return {"status" : "ok" , "service" : "Medical RAG API running...."}


@app.post("/ask",response_model=AnswerResponse)
def ask_questions(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400,detail="Question Cannot be empty")
    try:
        result  = ask_query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    


@app.get("/")
def root():
    return {
        "message" : "Medical RAG API is RUnning",
        "docs" : "/docs",
        "health" : "/health",
        "ask" : "POST /ask"
    }