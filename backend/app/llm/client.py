"""
ตัวติดต่อกับ Ollama (เครื่องยนต์ AI ในเครื่อง)
============================================
Ollama รันเป็นโปรแกรมแยกต่างหาก (พอร์ต 11434) และเปิด API ให้เราเรียก
งานของเราในไฟล์นี้คือ "ส่งข้อความไป และรับคำตอบกลับมา"

ทำไมต้องมีชั้นนี้?
- เวลาอยากเปลี่ยนเครื่องยนต์ (เช่น ไปใช้ OpenAI หรือ vLLM)
  แก้ที่ไฟล์นี้ที่เดียว ระบบส่วนอื่นไม่ต้องแตะ
- เรียก "LLM client" — เป็น pattern มาตรฐานในงาน AI เลย
"""

import httpx

from app.config import OLLAMA_URL, LLM_MODEL

# หมดเวลา 120 วิ เพราะโมเดลบน CPU คิดช้าได้ (โดยเฉพาะรอบแรกที่โหลด)
_TIMEOUT = httpx.Timeout(120.0)


async def generate_chat(messages: list[dict], model: str = LLM_MODEL) -> str:
    """
    ส่งประวัติการสนทนาให้โมเดล แล้วคืนคำตอบเป็นข้อความ

    messages: รายการข้อความ เช่น
        [{"role": "user", "content": "สวัสดี"},
         {"role": "assistant", "content": "สวัสดีครับ"}]
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,  # รอคำตอบเต็มก้อนก่อน (Phase 4 จะทำ streaming)
    }

    # httpx.AsyncClient = ลูกค้าที่ใช้เรียก HTTP แบบ async
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        response.raise_for_status()  # ถ้า error จะเด้ง exception ให้เห็น
        return response.json()["message"]["content"]


async def is_ollama_ready() -> bool:
    """เช็คว่า Ollama รันอยู่หรือยัง (ใช้ในหน้าเช็คสุขภาพ API)"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.get(f"{OLLAMA_URL}/api/tags")
            return True
    except httpx.HTTPError:
        return False
