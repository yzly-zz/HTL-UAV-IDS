"""
工具脚本：检测 data/raw/UAV-NIDD/ 下所有 CSV 文件的实际字符编码。
用途：当 preprocessor.py 的多编码自动探测失败时，手动运行此脚本获取确切编码。

使用方法：
    cd ProjBridge-V2
    python scripts/detect_encoding.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import chardet
except ImportError:
    print("❌ 缺少 chardet，请执行: pip install chardet")
    sys.exit(1)

CSV_DIR = os.path.join("data", "raw", "UAV-NIDD")

if not os.path.isdir(CSV_DIR):
    print(f"❌ 目录不存在: {CSV_DIR}")
    print("   请在项目根目录下运行此脚本。")
    sys.exit(1)

csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith(".csv")]

if not csv_files:
    print(f"⚠️ 在 {CSV_DIR} 中未找到任何 .csv 文件。")
    sys.exit(0)

print(f"正在扫描目录: {CSV_DIR}\n")
for fname in sorted(csv_files):
    fpath = os.path.join(CSV_DIR, fname)
    with open(fpath, "rb") as f:
        raw = f.read(50000)
    result = chardet.detect(raw)
    encoding = result["encoding"] or "unknown"
    confidence = result["confidence"] or 0.0
    status = "✅" if confidence > 0.8 else ("⚠️" if confidence > 0.5 else "❌")
    print(f"  {status} {fname}")
    print(f"      编码: {encoding}  置信度: {confidence:.1%}")
    print()