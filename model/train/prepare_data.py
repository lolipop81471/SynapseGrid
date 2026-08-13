"""
แปลงข้อมูลดิบ -> ไฟล์เทรน 🤖
============================
ขั้นตอน:
  1. อ่านไฟล์ .txt ทั้งหมดจาก data/raw/ (+ ตัวอย่าง)
  2. ทำความสะอาดข้อความ (ตัดหัว-ท้ายหนังสือ Gutenberg, จัดช่องว่าง)
  3. เทรน BPE tokenizer ใหม่กับข้อมูลจริง (คำศัพท์จะตรงกับข้อมูล)
  4. แปลงข้อความทั้งหมดเป็นลำดับ token
  5. แบ่งเป็น train (90%) / val (10%) และบันทึก

ผลลัพธ์:
  data/train.bin      — ข้อมูลสำหรับเทรน (numpy uint16)
  data/val.bin        — ข้อมูลสำหรับตรวจสอบ (numpy uint16)
  data/tokenizer.json — คำศัพท์ที่เทรนจากข้อมูลจริง
  data/meta.json      — สถิติทั้งหมด

รัน:  .venv/Scripts/python model/train/prepare_data.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from model.console import setup_utf8
from model.tokenizer.bpe import ByteLevelBPE

setup_utf8()

# ── ค่าตั้ง ──
RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
SAMPLE_CORPUS = Path(__file__).parent.parent.parent / "data" / "sample_corpus.txt"
OUT_DIR = Path(__file__).parent.parent.parent / "data"

NUM_MERGES = 1200        # จำนวนรอบที่ tokenizer รวมคำ (ยิ่งมาก vocab ยิ่งใหญ่)
TOKENIZER_TRAIN_CHARS = 400_000   # ใช้ข้อความกี่ตัวอักษรในการเทรน vocab (เร็วพอสมควร)
MAX_TOTAL_CHARS = 3_000_000       # จำกัดขนาดข้อมูลรวม (ปรับเพิ่มได้ถ้าต้องการ)
TRAIN_RATIO = 0.9        # สัดส่วน train : val


def clean_text(text: str) -> str:
    """
    ทำความสะอาดข้อความดิบ:
    1. ตัดส่วนหัว/ท้ายของหนังสือ Gutenberg (คำเตือนลิขสิทธิ์ ฯลฯ)
    2. บีบช่องว่างหลายบรรทัดให้เหลือ 2 บรรทัด (ย่อหน้า)
    """
    # หนังสือ Gutenberg มีกรอบกำกับไว้ — เอาข้อความในกรอบเท่านั้น
    for marker in ("*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"):
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    # ทิ้งเศษที่เหลือจากบรรทัด START (เช่น " EBOOK ALICE'S ... ***")
    if text.startswith(" "):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    for marker in ("*** END OF THE PROJECT GUTENBERG", "*** END OF THIS PROJECT GUTENBERG"):
        if marker in text:
            text = text.split(marker, 1)[0]
            break

    # จัดช่องว่าง: 3+ บรรทัดติดกัน -> 2 (ย่อหน้า), ตัดบรรทัดว่างหัว-ท้าย
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned = []
    blank = 0
    for ln in lines:
        if not ln.strip():
            blank += 1
            if blank <= 2:
                cleaned.append("")
        else:
            blank = 0
            cleaned.append(ln.strip())
    return "\n".join(cleaned).strip() + "\n"


def collect_texts() -> list[str]:
    """รวมไฟล์ข้อมูลทั้งหมด -> รายการข้อความที่ทำความสะอาดแล้ว"""
    paths = sorted(RAW_DIR.glob("*.txt")) if RAW_DIR.exists() else []
    if SAMPLE_CORPUS.exists():
        paths.append(SAMPLE_CORPUS)

    if not paths:
        print("⚠️  ไม่พบไฟล์ข้อมูลใน data/raw/ — รัน data/download_starter_data.py ก่อน")
        sys.exit(1)

    texts, total = [], 0
    for p in paths:
        try:
            raw = p.read_text(encoding="utf-8")
        except Exception as e:
            print(f"   ⚠️ อ่าน {p.name} ไม่ได้: {e}")
            continue
        cleaned = clean_text(raw)
        if len(cleaned) < 500:
            continue
        texts.append(cleaned)
        total += len(cleaned)
        print(f"   📄 {p.name}: {len(cleaned):,} ตัวอักษร")

    if total > MAX_TOTAL_CHARS:
        print(f"   ✂️  ตัดให้เหลือ {MAX_TOTAL_CHARS:,} ตัวอักษร (จำกัดเพื่อความเร็ว)")
    return texts


def train_tokenizer(texts: list[str]) -> ByteLevelBPE:
    """เทรน tokenizer กับตัวอย่างข้อมูล (ไม่ต้องใช้ทั้งชุด — vocab เรียนรู้จากตัวอย่างพอ)"""
    print(f"\n🧠 เทรน BPE tokenizer ใหม่ ({NUM_MERGES} merges) — รอสักครู่...")
    sample = " ".join(texts)[:TOKENIZER_TRAIN_CHARS]

    tok = ByteLevelBPE()
    t0 = time.perf_counter()
    for start in range(0, NUM_MERGES, 100):
        tok.learn([sample], num_merges=min(100, NUM_MERGES - start))
        elapsed = time.perf_counter() - t0
        print(f"   ⏳ merge {min(start + 100, NUM_MERGES)}/{NUM_MERGES} "
              f"(vocab {len(tok)}) — {elapsed:.0f} วิ")
    return tok


def main() -> None:
    print("🤖 เตรียมข้อมูลสำหรับเทรน\n")
    print("ขั้นตอนที่ 1/4: อ่าน + ทำความสะอาดข้อมูล")
    texts = collect_texts()

    tokenizer = train_tokenizer(texts)

    print(f"\nขั้นตอนที่ 2/4: แปลงข้อความเป็น token (ใช้ tokenizer ใหม่ {len(tokenizer)} คำศัพท์)")
    all_ids: list[int] = []
    for i, t in enumerate(texts):
        all_ids.extend(tokenizer.encode(t))
        if i % 5 == 0:
            print(f"   ⏳ ผ่านไฟล์ {i + 1}/{len(texts)}...")
    total_tokens = len(all_ids)
    print(f"   ✅ รวมทั้งหมด {total_tokens:,} tokens")

    print("\nขั้นตอนที่ 3/4: แบ่ง train/val และบันทึก")
    split = int(total_tokens * TRAIN_RATIO)
    train = np.array(all_ids[:split], dtype=np.uint16)
    val = np.array(all_ids[split:], dtype=np.uint16)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # np.save เติม .npy ต่อท้ายเอง — เลยต้องบันทึกแล้วเปลี่ยนชื่อ
    np.save(OUT_DIR / "train.bin.npy", train)
    os.replace(OUT_DIR / "train.bin.npy", OUT_DIR / "train.bin")
    np.save(OUT_DIR / "val.bin.npy", val)
    os.replace(OUT_DIR / "val.bin.npy", OUT_DIR / "val.bin")
    tokenizer.save(OUT_DIR / "tokenizer.json")

    meta = {
        "vocab_size": len(tokenizer),
        "num_merges": NUM_MERGES,
        "total_tokens": total_tokens,
        "train_tokens": int(train.size),
        "val_tokens": int(val.size),
        "train_ratio": TRAIN_RATIO,
        "num_files": len(texts),
    }
    (OUT_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n✅ เสร็จ! สรุปสถิติ:")
    for k, v in meta.items():
        print(f"   {k}: {v:,}" if isinstance(v, int) else f"   {k}: {v}")

    print("\nขั้นตอนที่ 4/4: ตัวอย่าง token ที่เรียนรู้ใหม่:")
    sample_pairs = [(p, n) for p, n in list(tokenizer.merges.items())[-5:]]
    reverse = {n: p for p, n in tokenizer.merges.items()}

    def preview(token: int) -> str:
        if token < 256:
            return chr(token)
        a, b = reverse[token]
        return preview(a) + preview(b)

    for pair, new_id in sample_pairs:
        print(f"   token #{new_id} = {preview(new_id)!r}")


if __name__ == "__main__":
    main()
