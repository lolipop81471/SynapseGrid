from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Question(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Hello, this is my backend!"}

@app.post("/ask")
def ask_question(question: Question):
    return {"you_asked": question.question, "answer": "ยังไม่มีคำตอบสำหรับคำถามนี้ในขณะนี้"}