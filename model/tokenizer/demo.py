"""
สาธิตการใช้งาน BPE Tokenizer
============================
รันด้วย:  py model/tokenizer/demo.py   (จากโฟลเดอร์ my-own-ai)
"""

import os
import sys

# ให้ Python หา package "model" เจอ ไม่ว่าเรารันจากโฟลเดอร์ไหน
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Windows console มักเป็น cp874/cp437 ซึ่งพิมพ์อีโมจิไม่ได้ — บังคับใช้ UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.tokenizer.bpe import ByteLevelBPE  # noqa: E402

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_corpus.txt")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tokenizer.json")


def main() -> None:
    # 1. อ่านข้อมูลตัวอย่าง (แต่ละบรรทัด = 1 ข้อความ)
    with open(CORPUS_PATH, encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    # 2. เทรน tokenizer — เรียนรู้ว่า "คู่อะไรควรรวมกัน"
    tokenizer = ByteLevelBPE()
    tokenizer.learn(texts, num_merges=300)

    print(f"✅ เทรนเสร็จ — ขนาดคำศัพท์: {len(tokenizer)} tokens")
    print(f"   (256 bytes ดิบ + {len(tokenizer) - 256} คู่ที่เรียนรู้การรวม)")
    print()

    # 3. ทดสอบ encode -> decode ว่ากลับมาครบเหมือนเดิมไหม
    test_sentences = [
        "สวัสดีครับ ผมชื่ออะไรก็ได้",
        "We are building AI from scratch on a laptop.",
        "ภาษาไทยผสม English ในประโยคเดียวก็ได้",
    ]
    for sent in test_sentences:
        ids = tokenizer.encode(sent)
        back = tokenizer.decode(ids)
        ok = "✅" if back == sent else "❌"
        print(f"{ok} [{len(ids)} tokens] {sent[:40]}")
        print(f"   tokens: {ids[:20]}{'...' if len(ids) > 20 else ''}")

    # 4. ดูตัวอย่างคู่ที่ถูกรวม — เราเห็น "รากฐานคำศัพท์" ของโมเดลเราเอง
    print()
    print("ตัวอย่างคู่ที่ถูกรวม (10 อันแรก):")
    for pair, new_id in list(tokenizer.merges.items())[:10]:
        print(f"   token #{new_id:>3} = {_preview_pair(tokenizer, pair)!r}")

    # 5. บันทึกไว้ใช้ต่อ (โมเดลใน Phase 3 จะใช้ tokenizer ตัวนี้)
    tokenizer.save(SAVE_PATH)
    print(f"\n💾 บันทึก tokenizer ไว้ที่: {SAVE_PATH}")


def _preview_pair(tokenizer: ByteLevelBPE, pair: tuple[int, int]) -> str:
    """ลองถอด token id กลับเป็นข้อความจริงเพื่อให้เห็นว่าคู่นั้นคืออะไร"""
    reverse = {new_id: p for p, new_id in tokenizer.merges.items()}

    def expand(token: int) -> list[int]:
        if token < 256:
            return [token]
        a, b = reverse[token]
        return expand(a) + expand(b)

    raw = bytes(expand(pair[0]) + expand(pair[1]))
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"bytes{raw!r}"


if __name__ == "__main__":
    main()
