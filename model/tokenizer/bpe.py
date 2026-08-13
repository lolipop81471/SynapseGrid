"""
Byte-Level BPE Tokenizer — เขียนจากศูนย์ 100%
============================================
นี่คือ "รากฐาน" ของ AI ทั้งหมด

AI อ่านตัวเลขได้อย่างเดียว อ่านตัวหนังสือไม่ได้
Tokenizer = ตัวกลางที่แปลง ข้อความ <-> ตัวเลข

วิธีทำงาน (Byte-Pair Encoding — วิธีเดียวกับที่ GPT ใช้จริง):
1. แปลงข้อความเป็น bytes (UTF-8) — แต่ละ byte แทนด้วยเลข 0-255
   ข้อดี: รองรับทุกภาษา (ไทย, อังกฤษ, อีโมจิ) โดยไม่ต้องรู้ภาษาล่วงหน้า
2. นับ "คู่ byte" ที่เจอบ่อยที่สุด แล้วรวมมันเป็น token ใหม่
   เช่น ถ้า "th" เจอบ่อย จะรวมเป็น token ตัวใหม่
3. ทำซ้ำไปเรื่อย ๆ จนได้จำนวน token ที่ต้องการ
   → ข้อความยาว ๆ กลายเป็น token น้อยลง สอน AI ได้เร็วขึ้น

สิ่งที่ได้เรียนรู้จากไฟล์นี้:
- เห็นว่า "คำศัพท์" ของ AI ไม่ใช่คำจริง ๆ แต่เป็น "ก้อน bytes ที่เจอบ่อย"
- เห็นว่าโมเดลแต่ละตัวมี "คำศัพท์" (vocab) ของตัวเองที่ต้องเทรนจากข้อมูล
"""

import json
from collections import Counter
from typing import Iterable


class ByteLevelBPE:
    """Tokenizer แบบ Byte-Level BPE — เทรนเองได้ ใช้เองได้"""

    def __init__(self):
        # token 0-255 = bytes ดิบ (256 ตัวแรก สำรองไว้แล้ว)
        self.vocab_size = 256

        # ตารางการรวม: {(token_a, token_b): token_ใหม่}
        # เช่น {(101, 104): 256} หมายถึง "e"+"h" รวมกันเป็น token #256
        self.merges: dict[tuple[int, int], int] = {}

    # ─────────────────────────────────────────────────────────
    # 1. การเทรน (เรียนรู้การแบ่งคำจากข้อมูล)
    # ─────────────────────────────────────────────────────────

    def learn(self, texts: Iterable[str], num_merges: int) -> "ByteLevelBPE":
        """
        เรียนรู้อะไรควรรวมกัน จากตัวอย่างข้อความ

        texts:       รายการข้อความ (ยิ่งหลาย/หลาก ยิ่งดี)
        num_merges:  จำนวนรอบที่รวม (ยิ่งมาก vocab ยิ่งใหญ่)
        """
        # แปลงทุกข้อความเป็นรายการ token id (เริ่มจาก byte ดิบ 0-255)
        sequences = [list(t.encode("utf-8")) for t in texts]

        for step in range(num_merges):
            # นับว่าคู่ไหนเจอบ่อยที่สุดในข้อมูลทั้งหมด
            pair_counts = self._count_pairs(sequences)
            if not pair_counts:
                break

            # หาคู่ที่เจอบ่อยที่สุด
            best_pair = max(pair_counts, key=pair_counts.get)

            # สร้าง token ใหม่ (id ถัดไป) ให้คู่นี้
            new_id = self.vocab_size
            self.vocab_size += 1
            self.merges[best_pair] = new_id

            # แทนที่คู่นี้ทุกตำแหน่งด้วย token ใหม่
            sequences = [self._merge_sequence(seq, best_pair, new_id)
                         for seq in sequences]

        return self

    @staticmethod
    def _count_pairs(sequences: list[list[int]]) -> Counter:
        """นับจำนวนคู่ token ที่อยู่ติดกันทุกคู่ในทุก sequence"""
        counts: Counter = Counter()
        for seq in sequences:
            for pair in zip(seq, seq[1:]):
                counts[pair] += 1
        return counts

    @staticmethod
    def _merge_sequence(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """เดินทีละตัว ถ้าเจอคู่ที่ตรง ให้แทนที่ด้วย token ใหม่"""
        out: list[int] = []
        i = 0
        while i < len(seq):
            if (i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]):
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    # ─────────────────────────────────────────────────────────
    # 2. การใช้งาน (แปลงข้อความ <-> ตัวเลข)
    # ─────────────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """
        แปลงข้อความ -> รายการ token id
        ใช้กฎการรวมแบบเดียวกับตอนเทรน: คู่ที่ถูกรวม "ก่อน" มีสิทธิ์ก่อน
        """
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            # หาคู่ที่ "ปรากฏอยู่" และ "เคยถูกรวม" ทั้งหมด
            present = set(zip(ids, ids[1:]))
            mergeable = [p for p in present if p in self.merges]
            if not mergeable:
                break
            # เลือกคู่ที่ถูกรวมเร็วที่สุด (id น้อยสุด = รวมก่อน = สำคัญกว่า)
            best = min(mergeable, key=lambda p: self.merges[p])
            ids = self._merge_sequence(ids, best, self.merges[best])
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        แปลงรายการ token id -> ข้อความ
        ต้อง "แกะ" token ที่รวมกันกลับเป็น bytes ก่อน แล้วค่อยถอด UTF-8
        """
        reverse = {new_id: pair for pair, new_id in self.merges.items()}

        def expand(token: int) -> list[int]:
            """แกะ token ใหญ่กลับเป็น bytes ดั้งเดิม"""
            if token < 256:
                return [token]
            a, b = reverse[token]
            return expand(a) + expand(b)

        raw = b"".join(bytes(expand(t)) for t in ids)
        return raw.decode("utf-8", errors="replace")

    # ─────────────────────────────────────────────────────────
    # 3. บันทึก/โหลด (ฝึกเทรนครั้งเดียว ใช้ได้ตลอด)
    # ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                # tuple ต้องแปลงเป็น list เพื่อให้ JSON เก็บได้
                "merges": [[list(pair), new_id] for pair, new_id in self.merges.items()],
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ByteLevelBPE":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab_size = data["vocab_size"]
        tok.merges = {tuple(pair): new_id for pair, new_id in data["merges"]}
        return tok

    def __len__(self) -> int:
        """ขนาดคำศัพท์ — มีกี่ token ให้ AI ใช้ได้"""
        return self.vocab_size
