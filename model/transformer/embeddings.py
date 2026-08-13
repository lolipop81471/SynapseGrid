"""
Embeddings — ตอนที่ 1 ของ Transformer
=====================================
สองชั้นแรกที่ข้อความต้องผ่าน ก่อนเข้าสู่ "ความฉลาด" (attention):

1. TokenEmbedding — แปลง token id เป็นเวกเตอร์
   จาก lab ที่เรียนไป: token id อย่างเดียว (เช่น 508) ไม่มีความหมาย
   embedding table = ตารางค้นหา vocab x d_model ที่ให้เวกเตอร์
   ตอนเริ่มสุ่ม แต่ระหว่างเทรนจะถูกปรับให้คำใกล้ความหมายอยู่ใกล้กัน

2. PositionalEncoding — บอกโมเดลว่าคำนี้อยู่ตำแหน่งที่เท่าไหร่
   ทำไมต้องมี? เพราะ attention (ที่กำลังจะเขียน) ทำให้ทุกคำ "เห็น"
   ทุกคำพร้อมกัน → ลำดับของคำจึงหายไปจากสัญญาณ!
   เช่น "แมวกัดหมา" กับ "หมากัดแมว" ใช้คำเดียวกัน แต่ความหมายตรงข้าม
   ถ้าไม่มีตำแหน่ง โมเดลจะแยกสองประโยคนี้ไม่ออก

วิธีแก้แบบต้นฉบับ (จากบทความ "Attention Is All You Need"):
   ใช้คลื่น sin/cos ความถี่ต่างกัน — ตำแหน่ง 0, 1, 2, ... แต่ละตำแหน่ง
   ได้ลายเซ็น (pattern) ที่ไม่ซ้ำกัน และตำแหน่งใกล้กันได้ pattern คล้ายกัน
"""

import numpy as np


class TokenEmbedding:
    """แปลง token id -> เวกเตอร์ ผ่านตารางค้นหา"""

    def __init__(self, vocab_size: int, d_model: int, seed: int = 42):
        # d_model = ขนาดของเวกเตอร์ (ความกว้างของ "สมอง" โมเดล)
        # เช่น 64 = แต่ละคำแทนด้วยตัวเลข 64 ตัว
        rng = np.random.default_rng(seed)
        # ค่าสุ่มเริ่มต้นเล็ก ๆ (0.02) — มาตรฐานที่ใช้กันจริง
        self.W = rng.normal(0.0, 0.02, (vocab_size, d_model))

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        token_ids: (seq_len,) — เช่น [508, 285, 156, ...]
        คืนค่า:    (seq_len, d_model) — แต่ละ token กลายเป็นเวกเตอร์

        การ indexing แบบนี้ W[token_ids] คือ "ดูตารางแล้วหยิบแถวตาม id"
        """
        return self.W[token_ids]


class PositionalEncoding:
    """เพิ่มสัญญาณบอกตำแหน่ง ด้วยคลื่น sin/cos ความถี่ต่างกัน"""

    def __init__(self, d_model: int, max_len: int = 2048):
        # สร้างตารางตำแหน่งล่วงหน้า: (max_len, d_model)
        pe = np.zeros((max_len, d_model))

        position = np.arange(max_len)[:, None]       # (max_len, 1) ตำแหน่ง 0..max_len-1
        # ความถี่ของคลื่น: ช่องคู่ (0,2,4,...) ใช้ sin, ช่องคี่ใช้ cos
        # ช่องแรกความถี่สูง (เปลี่ยนเร็ว) ช่องหลังความถี่ต่ำ (เปลี่ยนช้า)
        # → ตำแหน่งใกล้กัน: ช่องแรกต่างกันมาก แต่ช่องหลังใกล้กัน
        #   ตำแหน่งไกลกัน: ทุกช่องต่างกันหมด
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)    # ช่องคู่
        pe[:, 1::2] = np.cos(position * div_term)    # ช่องคี่
        self.pe = pe

    def forward(self, seq_len: int) -> np.ndarray:
        """คืนตารางตำแหน่ง (seq_len, d_model) ให้เอาไปบวกกับ embedding"""
        return self.pe[:seq_len]
