"""
My Own AI — Backend API
=======================
ไฟล์นี้คือ "ประตูหน้า" ของ backend ทั้งหมด
เวลาคุณรัน server ขึ้นมา ไฟล์นี้จะถูกโหลดก่อนเป็นไฟล์แรก

แนวคิดที่ควรรู้:
- FastAPI = library ที่ช่วยสร้าง API (ตัวกลางให้เว็บ/แอปอื่นเรียกใช้)
- "route" = เส้นทาง URL เช่น /api/chat ที่ frontend เรียกหา
- "async def" = ฟังก์ชันแบบ asynchronous — รอคำตอบจากภายนอก
  (เช่น รอโมเดล AI) โดยไม่บล็อกงานอื่น เปรียบเหมือนสั่งข้าวที่ร้านแล้ว
  ไปนั่งโต๊ะ ไม่ต้องยืนจ้องหน้าคนขาย
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat

# สร้าง instance ของ FastAPI — นี่คือตัว app ของเรา
app = FastAPI(
    title="My Own AI",
    description="AI ที่รันในเครื่องของเราเอง — สร้างจากมือเราล้วน ๆ",
    version="0.1.0",
)

# CORS = อนุญาตให้เว็บ (ที่อยู่คนละ origin) เรียก API เราได้
# ตอนพัฒนา frontend กับ backend รันคนละพอร์ต จำเป็นต้องเปิดอันนี้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # เปิดทั้งหมด (สำหรับตอนพัฒนา)
    allow_methods=["*"],
    allow_headers=["*"],
)

# รวม router (กลุ่มของเส้นทาง) เข้ากับ app
app.include_router(chat.router)


@app.get("/")
async def root():
    """ทดสอบว่า server รันอยู่ — เปิด http://localhost:8000 ดูได้"""
    return {"message": "My Own AI API ทำงานแล้ว!", "docs": "/docs"}


@app.get("/api/health")
async def health():
    """เช็คว่า backend พร้อมใช้งานหรือยัง"""
    return {"status": "ok"}
