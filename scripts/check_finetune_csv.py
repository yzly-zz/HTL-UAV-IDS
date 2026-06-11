"""诊断 uav_finetune_data.csv 文件状态"""
import os

path = "data/raw/UAV-NIDD/uav_finetune_data.csv"

if not os.path.exists(path):
    print(f"❌ 文件不存在: {path}")
    exit(1)

size = os.path.getsize(path)
print(f"文件路径: {path}")
print(f"文件大小: {size} 字节 ({size/1024:.1f} KB)")

if size == 0:
    print("\n⚠️ 文件大小为 0 字节——这是一个空文件！")
    print("   建议：检查文件是否在复制/解压过程中损坏，")
    print("   或用文本编辑器（Notepad++）直接打开查看。")
    exit(0)

print(f"\n--- 前 800 字节原始内容 (repr) ---")
with open(path, "rb") as f:
    raw = f.read(800)
print(repr(raw))

print(f"\n--- 前 800 字节尝试以文本显示 ---")
print(raw[:800])

print(f"\n--- 总行数统计 ---")
with open(path, "rb") as f:
    line_count = sum(1 for _ in f)
print(f"总行数: {line_count}")