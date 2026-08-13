"""
ตัวช่วยกลาง: บังคับให้ Windows console พิมพ์ภาษาไทย + อีโมจิ + สัญลักษณ์คณิตศาสตร์ได้
Windows ใช้ encoding cp874/cp437 ซึ่งพิมพ์บางตัวอักษรไม่ได้ — บังคับเป็น UTF-8
สคริปต์ทุกตัวในโปรเจกต์เรียกใช้ฟังก์ชันนี้ตอนเริ่ม
"""

import sys


def setup_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
