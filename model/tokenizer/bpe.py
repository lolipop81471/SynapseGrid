"""
Byte-Level BPE Tokenizer — เขียนจากศูนย์ 100% (เวอร์ชันปรับความเร็วแล้ว ⚡)
=====================================================================
AI อ่านตัวเลขได้อย่างเดียว อ่านตัวหนังสือไม่ได้
Tokenizer = ตัวกลางที่แปลง ข้อความ <-> ตัวเลข

วิธีทำงาน (Byte-Pair Encoding — วิธีเดียวกับที่ GPT ใช้จริง):
1. แปลงข้อความเป็น bytes (UTF-8) — แต่ละ byte แทนด้วยเลข 0-255
2. นับ "คู่ byte" ที่เจอบ่อยที่สุด แล้วรวมมันเป็น token ใหม่
3. ทำซ้ำจนได้จำนวน token ที่ต้องการ

🎯 ปรับ performance (การเพิ่มครั้งนี้):
- เวอร์ชันแรก: ทุก ๆ 1 merge สแกนข้อมูลทั้งชุดใหม่ → ช้า (O(merge × ข้อมูล))
- เวอร์ชันนี้: ติดตามว่าคู่ไหนอยู่ที่ไหน (active set) + อัปเดตเฉพาะจุด
  ที่เปลี่ยนรอบ ๆ การ merge → เร็วหลายสิบเท่า (หลักการเดียวกับที่
  GPT-2 tokenizer ใช้จริง) — เป็นตัวอย่าง "ปรับ performance" ในพอร์ตได้เลย
"""

import heapq
import json
from collections import Counter
from typing import Iterable


class ByteLevelBPE:
    """Tokenizer แบบ Byte-Level BPE — เทรนเองได้ ใช้เองได้"""

    def __init__(self):
        # token 0-255 = bytes ดิบ (256 ตัวแรก สำรองไว้แล้ว)
        self.vocab_size = 256

        # ตารางการรวม: {(token_a, token_b): token_ใหม่}
        self.merges: dict[tuple[int, int], int] = {}

    # ─────────────────────────────────────────────────────────
    # 1. การเทรน (เรียนรู้การแบ่งคำจากข้อมูล)
    # ─────────────────────────────────────────────────────────

    def learn(self, texts: Iterable[str], num_merges: int) -> "ByteLevelBPE":
        """
        เรียนรู้อะไรควรรวมกัน จากตัวอย่างข้อความ

        texts:       รายการข้อความ
        num_merges:  จำนวนรอบที่รวม (ยิ่งมาก vocab ยิ่งใหญ่)

        อัลกอริทึม (แบบมืออาชีพ):
        - เก็บจำนวนคู่ (pair count) ของแต่ละข้อความแยกกัน (local)
        - เก็บคู่ -> ชุดของข้อความที่มีคู่นั้น (active set)
        - เวลา merge: แก้เฉพาะคู่ที่ "ติดกับ" จุดที่ถูกเปลี่ยน
          ไม่ต้องสแกนข้อมูลทั้งหมดซ้ำ
        """
        # แปลงทุกข้อความเป็นรายการ token id (เริ่มจาก byte ดิบ)
        seqs = [list(t.encode("utf-8")) for t in texts]

        # จำนวนคู่ต่อข้อความ (local) + รวมทั้งชุด (stats) + active set
        local = [Counter(zip(seq, seq[1:])) for seq in seqs]
        stats: Counter = Counter()
        active: dict[tuple[int, int], set[int]] = {}
        for idx, counts in enumerate(local):
            for pair, cnt in counts.items():
                stats[pair] += cnt
                active.setdefault(pair, set()).add(idx)

        def decrement(idx: int, pair: tuple[int, int]) -> None:
            """ลดจำนวนคู่ในข้อความ idx — ถ้าถึง 0 ให้ออกจาก active set"""
            if local[idx][pair] > 0:
                local[idx][pair] -= 1
                stats[pair] -= 1
                if local[idx][pair] == 0:
                    active[pair].discard(idx)

        def increment(idx: int, pair: tuple[int, int]) -> None:
            """เพิ่มจำนวนคู่ในข้อความ idx — ถ้าจาก 0 เป็น >0 เข้า active set"""
            was_zero = local[idx][pair] == 0
            local[idx][pair] += 1
            stats[pair] += 1
            if was_zero:
                active.setdefault(pair, set()).add(idx)

        for _ in range(num_merges):
            # หาคู่ที่มีจำนวนมากที่สุด (ข้ามคู่ที่กลายเป็น 0 แล้ว)
            best, best_count = None, 0
            for pair, cnt in stats.items():
                if cnt > best_count:
                    best, best_count = pair, cnt
            if best is None or best_count == 0:
                break

            a, b = best
            new_id = self.vocab_size
            self.vocab_size += 1
            self.merges[best] = new_id

            # รวมคู่นี้เฉพาะในข้อความที่มีมันอยู่ (active) — เร็วมาก
            for idx in list(active.get(best, ())):
                seq = seqs[idx]
                i = 0
                while i < len(seq) - 1:
                    if seq[i] == a and seq[i + 1] == b:
                        # ลบจำนวนคู่ที่ตำแหน่ง i-1, i, i+1 (ก่อน merge)
                        for j in (i - 1, i, i + 1):
                            if 0 <= j < len(seq) - 1:
                                decrement(idx, (seq[j], seq[j + 1]))
                        # รวม a,b เป็น token ใหม่
                        seq[i:i + 2] = [new_id]
                        # เพิ่มจำนวนคู่ใหม่ที่ตำแหน่ง i-1, i (หลัง merge)
                        for j in (i - 1, i):
                            if 0 <= j < len(seq) - 1:
                                increment(idx, (seq[j], seq[j + 1]))
                        # หลัง merge แล้วคู่ (a,b) ใหม่จะเกิดไม่ได้ (token ใหม่ไม่ซ้ำ)
                        # เลยเดินหน้าต่อได้เลย
                    i += 1

        return self

    # ─────────────────────────────────────────────────────────
    # 2. การใช้งาน (แปลงข้อความ <-> ตัวเลข)
    # ─────────────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """
        แปลงข้อความ -> รายการ token id

        เวอร์ชันเร็ว: ใช้ min-heap (คิวลำดับความสำคัญ)
        - คู่ที่ถูกรวม "ก่อน" (merge id น้อย) มีสิทธิ์ก่อน
        - หลัง merge แต่ละครั้ง มีแค่ 2 จุดที่เปลี่ยน (i-1 และ i)
          → ใส่คู่ใหม่ลง heap เท่านั้น ไม่ต้องสแกนทั้งประโยคซ้ำ
        """
        ids = list(text.encode("utf-8"))
        if len(ids) < 2:
            return ids

        heap: list[tuple[int, int]] = []
        for i in range(len(ids) - 1):
            rank = self.merges.get((ids[i], ids[i + 1]))
            if rank is not None:
                heapq.heappush(heap, (rank, i))  # rank น้อย = รวมก่อน = สำคัญกว่า

        while heap:
            rank, i = heapq.heappop(heap)
            if i >= len(ids) - 1:
                continue
            # เช็คว่ารายการนี้ยังใช้ได้ไหม (ตำแหน่งอาจขยับไปแล้วจากการ merge ก่อนหน้า)
            if self.merges.get((ids[i], ids[i + 1])) != rank:
                continue  # ล้าสมัย — ข้าม

            new_id = rank  # rank = id ของ token ที่เกิดจากการรวมคู่นี้
            ids[i:i + 2] = [new_id]
            # มีแค่คู่ที่ตำแหน่ง i-1 และ i เท่านั้นที่เปลี่ยนไป
            for j in (i - 1, i):
                if 0 <= j < len(ids) - 1:
                    rank2 = self.merges.get((ids[j], ids[j + 1]))
                    if rank2 is not None:
                        heapq.heappush(heap, (rank2, j))
        return ids

    def decode(self, ids: list[int]) -> str:
        """แปลงรายการ token id -> ข้อความ (แกะ token กลับเป็น bytes แล้วถอด UTF-8)"""
        reverse = {new_id: pair for pair, new_id in self.merges.items()}

        def expand(token: int) -> list[int]:
            if token < 256:
                return [token]
            a, b = reverse[token]
            return expand(a) + expand(b)

        raw = b"".join(bytes(expand(t)) for t in ids)
        return raw.decode("utf-8", errors="replace")

    # ─────────────────────────────────────────────────────────
    # 3. บันทึก/โหลด
    # ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
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
        return self.vocab_size
