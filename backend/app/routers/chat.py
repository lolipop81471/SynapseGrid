"""
เส้นทาง (route) สำหรับการแชท
=============================
API ตัวแรกของเรา! เมื่อ frontend ส่งข้อความมา:
    POST /api/chat  {"message": "สวัสดี"}

backend จะ:
    1. รับข้อความ (ตรวจสอบชนิดข้อมูลด้วย pydantic)
    2. ส่งต่อให้โมเดล AI ในเครื่อง (Ollama)
    3. คืนคำตอบกลับไปเป็น JSON

"router" = กลุ่มของเส้นทางที่เกี่ยวข้องกัน เรารวมมันเข้า app หลักใน main.py
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.llm.client import generate_chat, is_ollama_ready

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    """
    ตัวกำหนด "รูปร่าง" ของข้อมูลที่รับเข้ามา (schema)
    ถ้า client ส่งมาไม่ตรงนี้ pydantic จะตอบ error 422 ให้อัตโนมัติ
    """
    message: str = Field(..., min_length=1, description="ข้อความที่ส่งให้ AI")
    history: list[dict] = Field(
        default=[],
        description="ประวัติการสนทนาก่อนหน้า (optional)",
    )


class ChatResponse(BaseModel):
    """รูปร่างของคำตอบที่ส่งกลับไป"""
    reply: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """รับข้อความจากผู้ใช้ แล้วคืนคำตอบจากโมเดลในเครื่อง"""

    # 1. เช็คก่อนว่าเครื่องยนต์ AI รันอยู่ไหม — ถ้าไม่ จะได้ error ที่เข้าใจง่าย
    if not await is_ollama_ready():
        raise HTTPException(
            status_code=503,
            detail="Ollama ยังไม่รัน — เปิด Ollama ก่อน แล้วลองใหม่ (ดู README)",
        )

    # 2. ต่อประวัติเก่ากับข้อความใหม่เข้าเป็นรายการเดียวกัน
    #    (โมเดลจะได้รู้บริบทว่าเราคุยอะไรกันมาก่อน)
    messages = request.history + [{"role": "user", "content": request.message}]

    # 3. ส่งให้โมเดลคิด แล้วรับคำตอบกลับมา
    reply = await generate_chat(messages)

    return ChatResponse(reply=reply)
