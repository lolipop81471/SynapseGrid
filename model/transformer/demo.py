"""
สาธิต Embeddings — ใช้ tokenizer ของเราเอง + โมเดลจริงที่กำลังสร้าง
=================================================================
รัน:  py model/transformer/demo.py

จะเห็น:
1. ข้อความ -> tokens (BPE ที่เราเขียน) -> เวกเตอร์ (embedding)
2. positional encoding ทำให้แต่ละตำแหน่งมีลายเซ็นไม่ซ้ำกัน
3. คำใกล้กันในความหมาย เริ่มมีเวกเตอร์ใกล้กัน (หลังเทรนจะชัดขึ้น)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.tokenizer.bpe import ByteLevelBPE            # noqa: E402
from model.transformer.embeddings import (              # noqa: E402
    PositionalEncoding,
    TokenEmbedding,
)

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tokenizer.json")

D_MODEL = 64  # ขนาดเวกเตอร์ของโมเดลเรา (ยิ่งมากยิ่ง "สมองกว้าง" แต่ช้าลง)


def main() -> None:
    # 1. โหลด tokenizer ที่เทรนไว้ตอน Phase 2
    tokenizer = ByteLevelBPE.load(TOKENIZER_PATH)
    print(f"Tokenizer โหลดแล้ว — ขนาดคำศัพท์: {len(tokenizer)}")

    # 2. สร้างชั้น embedding
    embed = TokenEmbedding(vocab_size=len(tokenizer), d_model=D_MODEL)
    pos = PositionalEncoding(d_model=D_MODEL)

    # 3. แปลงข้อความจริง -> tokens -> เวกเตอร์
    text = "แมวกัดหมา แต่หมากัดแมว"
    token_ids = np.array(tokenizer.encode(text))
    print(f"\nข้อความ: {text!r}")
    print(f"tokens : {token_ids.tolist()}  (จำนวน {len(token_ids)} tokens)")

    x = embed.forward(token_ids)                       # (seq_len, d_model)
    p = pos.forward(seq_len=len(token_ids))            # (seq_len, d_model)
    h = x + p                                          # เวกเตอร์สุดท้าย = ความหมาย + ตำแหน่ง

    print(f"embedding shape: {x.shape}  ← แต่ละ token กลายเป็นเวกเตอร์ {D_MODEL} มิติ")

    # 4. ดูว่า "แมว" ที่ปรากฏ 2 จุด ต่างกันยังไง
    #    (คำหนึ่งคำอาจแยกเป็นหลาย token — หาตำแหน่งของลำดับ token ของคำนั้น)
    mew_tokens = tokenizer.encode("แมว")
    spans = [i for i in range(len(token_ids) - len(mew_tokens) + 1)
             if list(token_ids[i:i + len(mew_tokens)]) == mew_tokens]
    if len(spans) >= 2:
        i1, i2 = spans[0], spans[1]
        same = h[i1] == h[i2]
        print(f"\n'แมว' ตัวแรก (token ตำแหน่ง {i1}) และตัวที่สอง (token ตำแหน่ง {i2})")
        print(f"  เวกเตอร์หลัง + positional เหมือนกันเป๊ะหรือไม่? "
              f"{'ใช่ ❌ (แย่)' if same.all() else 'ไม่ ✅ (ถูกแล้ว — โมเดลรู้ว่าคนละตำแหน่ง)'}")
        print(f"  ระยะห่างระหว่างสองเวกเตอร์: {np.linalg.norm(h[i1] - h[i2]):.2f}")

    # 5. ดูว่า positional encoding แต่ละตำแหน่งไม่ซ้ำกัน
    p1, p2, p3 = pos.forward(3)
    print(f"\npositional encoding ตำแหน่ง 0,1,2 (แสดง 6 มิติแรก):")
    print(f"  ตำแหน่ง 0: {np.round(p1[:6], 2)}")
    print(f"  ตำแหน่ง 1: {np.round(p2[:6], 2)}")
    print(f"  ตำแหน่ง 2: {np.round(p3[:6], 2)}")
    print(f"  → แต่ละตำแหน่งมีลายเซ็นไม่ซ้ำกัน ทำให้โมเดลรู้ 'คำนี้อยู่ที่ไหน'")

    # 6. ทดสอบความใกล้ของเวกเตอร์ (ความหมาย) — ยังไม่เทรน ค่าจึงสุ่ม
    print("\n(โน้ต: ตอนนี้ embedding ยังสุ่มอยู่ — ความหมายจะค่อย ๆ ถูกเรียนรู้ตอนเทรน)")


if __name__ == "__main__":
    main()
