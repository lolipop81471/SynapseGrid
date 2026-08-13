"""
Scaled Dot-Product Attention — หัวใจของ Transformer
=====================================================
สูตร:  Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V

- Q (Query) = ทุกคำ "ถาม": ฉันกำลังหาข้อมูลแบบไหน
- K (Key)   = ทุกคำ "ตอบ": ฉันคืออะไร มีข้อมูลอะไร
- V (Value) = ทุกคำ "ให้": ข้อมูลจริงที่ฉันมี

ทำไมต้องหาร √d?  ถ้า d ใหญ่ ค่าจาก Q·Kᵀ จะใหญ่ตาม -> softmax จะ
กลายเป็น 0 หรือ 1 มากเกินไป (เลือกเผด็จการ ไม่ยืดหยุ่น)
หาร √d แล้วค่ากลับมาอยู่ช่วงที่ softmax ทำงานได้ดี
"""

import os
import sys

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """
    แปลงตัวเลขเป็นความน่าจะเป็น (แถวรวมกัน = 1)
    ลบค่าสูงสุดก่อน (กันเลขล้นเกิน) — จำจาก lab ที่เขียนไว้
    axis=-1 = ทำตามแกนสุดท้าย (ในที่นี้คือแถว)
    """
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Q, K, V: (seq_len, d_model) — เวกเตอร์ของทุกคำในประโยค
    คืนค่า:  (seq_len, d_model) — "ความคิด" ใหม่ของทุกคำ หลังเห็นบริบท

    ทั้งหมดมี 4 ขั้น:
    """
    d_k = Q.shape[-1]                                  # ขนาดเวกเตอร์

    # ① ตารางความสัมพันธ์: ทุกคำถาม (Q) เปรียบกับทุกคำ (K)
    #    Q @ K.T -> (seq, d) @ (d, seq) = (seq, seq)
    scores = Q @ K.T                                   # (seq, seq)

    # ② หาร √d เพื่อกันเลขใหญ่เกิน
    scores = scores / np.sqrt(d_k)

    # ③ แปลงแต่ละแถวเป็น "ความสนใจ" (รวมกัน = 1 ต่อแถว)
    weights = softmax(scores)                          # (seq, seq)

    # ④ ผสมข้อมูล V ตามสัดส่วนความสนใจ
    #    แต่ละแถว = ผลรวมถ่วงน้ำหนักของ V ทุกแถว
    out = weights @ V                                  # (seq, d)

    return out


if __name__ == "__main__":
    # ให้ Python หา package "model" เจอ (รันจากโฟลเดอร์ไหนก็ได้)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from model.console import setup_utf8
    setup_utf8()

    # ── ทดสอบด้วยตัวอย่างเล็ก ๆ ที่คำนวณด้วยมือได้ ──
    print("ทดสอบ: 3 คำ (ผม, รัก, หมา) เวกเตอร์ 4 มิติ")

    X = np.array([
        [1, 0, 1, 0],   # ผม
        [0, 1, 0, 1],   # รัก
        [1, 1, 0, 0],   # หมา
    ])

    # self-attention: ให้ทุกคำถามตัวเอง (Q = K = V = X)
    out = scaled_dot_product_attention(X, X, X)

    print("\nตารางความสนใจ (ก่อน softmax, หาร √4 แล้ว):")
    scores = X @ X.T / np.sqrt(4)
    print(np.round(scores, 2))
    print("\nความสนใจหลัง softmax (แต่ละแถวรวม = 1):")
    print(np.round(softmax(scores), 3))
    print("\n'ความคิด' ใหม่ของแต่ละคำ (ผลรวมถ่วงน้ำหนักของ V):")
    print(np.round(out, 3))
