# SynapseGrid 🧠 — AI Infrastructure สร้างจากศูนย์ 100%

AI ที่สร้างเองทุกอย่างบนโน้ตบุ๊ค — **ไม่พึ่ง Ollama, ไม่พึ่งโมเดลสำเร็จรูป, ไม่พึ่ง cloud**
ทุกชั้นเขียนด้วยมือ ตั้งแต่ตัวแปลงข้อความเป็นตัวเลข จนถึงตัวโมเดลเอง

## ทำไมถึงทำโปรเจกต์นี้

เพื่อเข้าใจระบบภายในของ AI แบบรากฐานจริง ๆ และมี AI เป็นของตัวเอง
ทุก component สร้างเอง เลือกปรับ performance ได้เองทั้งหมด

## สถาปัตยกรรม (ทุกชั้นเขียนเอง)

```
┌─────────────────────────────────────────┐
│  Frontend — Web UI (HTML/CSS/JS)        │  ← Phase 7
├─────────────────────────────────────────┤
│  Backend — FastAPI (Python)             │  ← Phase 6
├─────────────────────────────────────────┤
│  Inference engine — generation + sample │  ← Phase 5
├─────────────────────────────────────────┤
│  Model — เทรนเอง 100%                    │
│  • Transformer (numpy)                  │  ← Phase 3-4
│  • BPE Tokenizer ✅ ทำเสร็จแล้ว           │  ← Phase 2
└─────────────────────────────────────────┘
```

## ความคืบหน้า

| Phase | สิ่งที่ทำ | สถานะ |
|---|---|---|
| 1 | โครงโปรเจกต์ + เรียน backend (FastAPI) | ✅ |
| 2 | **BPE Tokenizer จากศูนย์** | ✅ |
| 3 | Transformer architecture | ⏳ ถัดไป |
| 4 | Training (forward + backprop + loop) | ⏳ |
| 5 | Inference engine + sampling | ⏳ |
| 6 | เชื่อมกับ backend API | ⏳ |
| 7 | Web UI + polish + docs สำหรับพอร์ต | ⏳ |

## วิธีรันสิ่งที่ทำเสร็จแล้ว

```bash
# เทรน tokenizer และดูผล (จากโฟลเดอร์ my-own-ai)
py model/tokenizer/demo.py

# รัน backend API (docs อัตโนมัติที่ /docs)
cd backend
../.venv/Scripts/python -m uvicorn app.main:app --reload
```

## โฟลเดอร์

```
model/tokenizer/   BPE Tokenizer — แปลงข้อความ <-> ตัวเลข (เขียนเอง)
backend/           FastAPI API server
frontend/          Web UI (เร็ว ๆ นี้)
data/              ข้อมูลสำหรับเทรน + tokenizer ที่เทรนเสร็จ
docs/              เอกสาร + architecture diagram (เร็ว ๆ นี้)
lab/               การทดลอง/บทเรียนย่อย
```
