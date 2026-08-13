"""
Lab: การคูณเมทริกซ์ด้วยมือ — รากฐานคณิตศาสตร์ของ AI
=================================================
ทุกอย่างที่ AI ทำ (จำ, คิด, สร้าง) ล้วนเป็นการคูณเมทริกซ์จำนวนมหาศาล
ไฟล์นี้เขียน matmul ด้วย Python ล้วน ๆ เพื่อให้เห็นกลไกจริง
(การคูณเมทริกซ์ = เอาแถวของฝั่งซ้าย กับหลักของฝั่งขวา มาคูณแล้วรวม)

รัน:  py lab/matmul.py
"""

import random
import sys

# Windows console มักเป็น cp874/cp437 พิมพ์อีโมจิไม่ได้ — บังคับใช้ UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def matmul(A, B):
    """
    คูณเมทริกซ์ A (แถว x กลาง) กับ B (กลาง x หลัก)

    A = [[a11, a12],      B = [[b11, b12],
         [a21, a22]]           [b21, b22]]

    ผลลัพธ์[i][j] = ผลรวมของ A[i][k] * B[k][j] ทุก k
    """
    rows_a, cols_a = len(A), len(A[0])
    rows_b, cols_b = len(B), len(B[0])
    assert cols_a == rows_b, "จำนวนหลักของ A ต้องเท่ากับจำนวนแถวของ B"

    C = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):          # ไล่ทุกแถวของ A
        for j in range(cols_b):      # ไล่ทุกหลักของ B
            total = 0.0
            for k in range(cols_a):  # คูณ-รวมตามสูตร
                total += A[i][k] * B[k][j]
            C[i][j] = total
    return C


def matvec(A, v):
    """คูณเมทริกซ์กับเวกเตอร์ — กรณีพิเศษของ matmul ที่ใช้บ่อยมาก"""
    return [sum(row[k] * v[k] for k in range(len(v))) for row in A]


def softmax(logits):
    """
    แปลงตัวเลขดิบให้เป็น "ความน่าจะเป็น" (รวมกัน = 1)
    ใช้ตอนท้ายสุดของโมเดล เพื่อเลือกว่าคำไหนควรออกมา
    """
    # ลบด้วยค่าสูงสุดก่อน (กันเลขล้นเกิน — ตัวเลขใหญ่เกินจะ error)
    m = max(logits)
    exps = [pow(2.718281828, x - m) for x in logits]  # e^(x-m)
    total = sum(exps)
    return [e / total for e in exps]


# ───────────────────────── ทดสอบ ─────────────────────────

def test_matmul():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    C = matmul(A, B)
    assert C == [[19.0, 22.0], [43.0, 50.0]], f"matmul ผิด! ได้ {C}"
    print("✅ matmul ถูกต้อง: [[1,2],[3,4]] x [[5,6],[7,8]] =", C)


def test_softmax():
    p = softmax([2.0, 1.0, 0.1])
    assert abs(sum(p) - 1.0) < 1e-9, "ความน่าจะเป็นต้องรวมกันเป็น 1"
    assert p[0] > p[1] > p[2], "เลขมากต้องได้โอกาสมากกว่า"
    print("✅ softmax ถูกต้อง: [2.0, 1.0, 0.1] ->", [round(x, 4) for x in p])


def demo_timings():
    """วัดว่าการคูณเมทริกซ์ Python ล้วนช้าแค่ไหน (เพื่อเทียบกับ numpy)"""
    random.seed(42)
    n = 64
    A = [[random.random() for _ in range(n)] for _ in range(n)]
    B = [[random.random() for _ in range(n)] for _ in range(n)]

    import time
    start = time.perf_counter()
    C = matmul(A, B)
    elapsed = time.perf_counter() - start

    print(f"matmul {n}x{n} ด้วย Python ล้วน: {elapsed:.3f} วิ")
    print(f"→ โมเดล Transformer ต้องคูณเมทริกซ์แบบนี้หลายล้านครั้งต่อการเทรน")
    print(f"→ นี่คือเหตุผลว่าทำไมต้องมี numpy (เครื่องคิดเลข) ช่วย")


if __name__ == "__main__":
    test_matmul()
    test_softmax()
    demo_timings()
    print("\n🎓 สรุป: matmul + softmax คือสองชิ้นหลักของทุกโมเดล AI")
