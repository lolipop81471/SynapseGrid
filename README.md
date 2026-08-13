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
# 1. เทรน tokenizer และดูผล
py model/tokenizer/demo.py

# 2. ดาวน์โหลดข้อมูลเริ่มต้น (Gutenberg 3 เล่ม + วิกิไทย) — ดู docs/data.md
py data/download_starter_data.py

# 3. แปลงข้อมูลเป็นไฟล์เทรน (ทำความสะอาด + เทรน vocab ใหม่ + แบ่ง train/val)
.venv/Scripts/python model/train/prepare_data.py

# 4. รัน backend API (docs อัตโนมัติที่ /docs)
cd backend
../.venv/Scripts/python -m uvicorn app.main:app --reload
```

## โฟลเดอร์

```
model/tokenizer/   BPE Tokenizer — แปลงข้อความ <-> ตัวเลข (เขียนเอง)
model/transformer/ Transformer: embeddings, attention (เขียนเอง)
model/train/       Data pipeline: ข้อมูลดิบ -> train.bin/val.bin
backend/           FastAPI API server
frontend/          Web UI (เร็ว ๆ นี้)
data/              ข้อมูลเทรน + tokenizer + ไฟล์ที่สร้างตอนรัน (ไม่ commit)
docs/              เอกสาร: data.md, architecture (เร็ว ๆ นี้)
lab/               การทดลอง/บทเรียนย่อย (matmul, สนามทดลอง attention)
```

## สถิติข้อมูลเริ่มต้น (รันสคริปต์แล้วได้เท่านี้)

- แหล่งข้อมูล: Alice in Wonderland, Frankenstein, Pride and Prejudice (Gutenberg) + บทความวิกิไทย + ตัวอย่าง
- ขนาด: ~1.3 ล้าน tokens (train 90% / val 10%), vocab 1,456
- ดูสถิติล่าสุดได้ที่ `data/meta.json`
