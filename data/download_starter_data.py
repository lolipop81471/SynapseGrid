"""
ดาวน์โหลดข้อมูลเริ่มต้นสำหรับเทรน 📥
====================================
สิ่งที่ดาวน์โหลด:
- หนังสือสาธารณสมบัติ 3 เล่มจาก Project Gutenberg (อังกฤษ)
- บทความสุ่มจาก Wikipedia ไทย (ข้อความล้วน)

รัน:  py data/download_starter_data.py
ข้อมูลจะถูกเก็บใน:  data/raw/  (รันซ้ำได้ — ข้ามไฟล์ที่มีอยู่แล้ว)
"""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# หนังสือ Gutenberg: {id: ชื่อไฟล์} — เป็นสาธารณสมบัติ (public domain)
GUTENBERG_BOOKS = {
    11: "alice_in_wonderland.txt",    # Alice's Adventures in Wonderland
    84: "frankenstein.txt",           # Frankenstein
    1342: "pride_and_prejudice.txt",  # Pride and Prejudice
}

THAI_WIKI_COUNT = 12    # จำนวนบทความวิกิไทยที่จะลองสุ่ม (ได้จริงน้อยกว่านี้เล็กน้อย)
THAI_WIKI_CHARS = 2500  # ตัวอักษรสูงสุดต่อบทความ

HEADERS = {"User-Agent": "SynapseGrid/0.1 (learning project; local training)"}


def fetch(url: str) -> bytes:
    """โหลดข้อมูลจาก URL — มี timeout + กัน error กลางทาง"""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def download_gutenberg() -> None:
    """โหลดหนังสือจาก Project Gutenberg (ข้อความล้วน รูปแบบ pg{id}.txt)"""
    print(f"📖 ดาวน์โหลดหนังสือจาก Project Gutenberg ({len(GUTENBERG_BOOKS)} เล่ม)...")
    for book_id, filename in GUTENBERG_BOOKS.items():
        dest = RAW_DIR / filename
        if dest.exists():
            print(f"   ⏭️  {filename} มีอยู่แล้ว — ข้าม")
            continue
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        try:
            text = fetch(url).decode("utf-8", errors="replace")
            dest.write_text(text, encoding="utf-8")
            size_kb = len(text) // 1024
            print(f"   ✅ {filename} ({size_kb} KB)")
        except Exception as e:
            print(f"   ❌ {filename} ล้มเหลว: {e}")


def _api(action_params: str) -> dict:
    url = "https://th.wikipedia.org/w/api.php?format=json&" + action_params
    return json.loads(fetch(url).decode("utf-8"))


def download_thai_wiki() -> None:
    """
    ดึงบทความจาก Wikipedia ไทย แบบ 2 ขั้นตอน (เชื่อถือได้กว่า generator=random):
      1. สุ่มชื่อบทความ (list=random)
      2. ดึงเนื้อหาของชื่อเหล่านั้น (titles=...) — ข้ามหน้าที่ไม่มีเนื้อหา
    """
    print(f"\n🌏 ดาวน์โหลดบทความสุ่มจาก Wikipedia ไทย (พยายาม {THAI_WIKI_COUNT} หน้า)...")

    # ขั้น 1: สุ่มชื่อ
    try:
        data = _api(f"action=query&list=random&rnnamespace=0&rnlimit={THAI_WIKI_COUNT}")
        titles = [r["title"] for r in data["query"]["random"]]
    except Exception as e:
        print(f"   ❌ เรียก API วิกิไทยล้มเหลว (ขั้นสุ่มชื่อ): {e}")
        return

    # ขั้น 2: ดึงเนื้อหาทีละชุด (MediaWiki รับ titles ได้หลายชื่อคั่นด้วย |)
    saved = 0
    for i in range(0, len(titles), 10):
        batch = titles[i:i + 10]
        joined = urllib.parse.quote("|".join(batch), safe="|")
        try:
            data = _api(
                f"action=query&prop=extracts&explaintext=1&exchars={THAI_WIKI_CHARS}"
                f"&exlimit=max&titles={joined}"
            )
        except Exception as e:
            print(f"   ❌ เรียก API ล้มเหลว (ขั้นเนื้อหา): {e}")
            continue

        for page in (data.get("query") or {}).get("pages", {}).values():
            title = page.get("title", "untitled")
            extract = page.get("extract", "").strip()
            if not extract:
                continue  # หน้า stub/เปล่า — ข้าม
            safe = "".join(c for c in title if c.isalnum() or c in " _-")[:60].strip()
            dest = RAW_DIR / f"wiki_th_{safe}.txt"
            if dest.exists():
                continue
            dest.write_text(f"{title}\n\n{extract}\n", encoding="utf-8")
            saved += 1
            print(f"   ✅ {title[:45]} ({len(extract)} ตัวอักษร)")
    print(f"   บันทึกใหม่ {saved} หน้า")


def main() -> None:
    print(f"ข้อมูลจะถูกเก็บใน: {RAW_DIR}\n")
    download_gutenberg()
    download_thai_wiki()
    files = list(RAW_DIR.glob("*.txt"))
    print(f"\n🎉 เสร็จ — มีไฟล์ข้อมูลทั้งหมด {len(files)} ไฟล์ใน data/raw/")
    print("ขั้นต่อไป: รัน .venv/Scripts/python model/train/prepare_data.py เพื่อแปลงเป็นไฟล์เทรน")


if __name__ == "__main__":
    main()
