"""
工具脚本：查看所有数据集的列名、维度与标签分布。
用途：确认 preprocessor.py 中的 label_col 参数应填什么值。

使用方法：
    cd ProjBridge-V2
    python scripts/inspect_data.py
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

LABEL_CANDIDATES = [
    "Label", "label", "attack", "Attack",
    "class", "Class", "type", "Type", "target", "Target", "is_attack"
]

CSV_ENCODINGS = ["utf-8", "gbk", "latin-1", "ISO-8859-1", "cp1252"]


def inspect_csv_files(csv_dir: str):
    if not os.path.isdir(csv_dir):
        print(f"❌ 目录不存在: {csv_dir}")
        return

    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith(".csv")])
    if not csv_files:
        print(f"⚠️ {csv_dir} 下无 CSV 文件")
        return

    for fname in csv_files:
        fpath = os.path.join(csv_dir, fname)
        for enc in CSV_ENCODINGS:
            try:
                df = pd.read_csv(fpath, encoding=enc, nrows=5)
                print(f"\n{'=' * 70}")
                print(f"📄 文件: {fname}")
                print(f"   编码: {enc}")
                print(f"   规模(预览): {df.shape[0]} 行 × {df.shape[1]} 列")
                print(f"   全部列名 ({df.shape[1]} 列):")
                for i, col in enumerate(df.columns):
                    dtype_hint = str(df[col].dtype)
                    sample = str(df[col].iloc[0])[:50] if len(df) > 0 else "N/A"
                    print(f"     [{i:03d}] {col:<35s}  dtype={dtype_hint:<10s}  sample={sample}")

                found_labels = [c for c in LABEL_CANDIDATES if c in df.columns]
                if found_labels:
                    for lbl in found_labels:
                        unique_vals = df[lbl].dropna().unique()
                        print(f"   🎯 标签列 '{lbl}': 去重值 = {unique_vals.tolist()}")
                else:
                    print(f"   ⚠️ 未匹配到任何已知标签列名，请手动确认。")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                print(f"   ❌ 读取异常 ({enc}): {e}")
        else:
            print(f"\n❌ 无法解码: {fname}")


def inspect_parquet_file(parquet_path: str):
    if not os.path.exists(parquet_path):
        print(f"\n❌ 文件不存在: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    print(f"\n{'=' * 70}")
    print(f"📦 文件: {os.path.basename(parquet_path)}")
    print(f"   总规模: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    print(f"   全部列名 ({df.shape[1]} 列):")
    for i, col in enumerate(df.columns):
        dtype_hint = str(df[col].dtype)
        sample = str(df[col].iloc[0])[:50] if len(df) > 0 else "N/A"
        print(f"     [{i:03d}] {col:<35s}  dtype={dtype_hint:<10s}  sample={sample}")

    found_labels = [c for c in LABEL_CANDIDATES if c in df.columns]
    if found_labels:
        for lbl in found_labels:
            unique_vals = df[lbl].dropna().unique()
            print(f"   🎯 标签列 '{lbl}': 去重值数量 = {len(unique_vals)}, "
                  f"示例 = {unique_vals[:10].tolist()}")
    else:
        print(f"   ⚠️ 未匹配到任何已知标签列名，请手动确认。")


if __name__ == "__main__":
    print("=" * 70)
    print("  ProjBridge-V2  数据集探查工具")
    print("=" * 70)

    print("\n▶ 阶段 1/2：检查 UAV-NIDD 目标域数据 (CSV)")
    inspect_csv_files(os.path.join("data", "raw", "UAV-NIDD"))

    print("\n▶ 阶段 2/2：检查 ToN-IoT 源域数据 (Parquet)")
    inspect_parquet_file(os.path.join("data", "raw", "CIC-ToN-IoT-V2.parquet"))

    print("\n" + "=" * 70)
    print("  探查完成。请将以上输出完整复制给代码评审人。")
    print("=" * 70)