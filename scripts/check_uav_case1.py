"""诊断 UAV-Case1-Label.csv 的编码与标签分布"""
import pandas as pd

path = "data/raw/UAV-NIDD/UAV-Case1-Label.csv"

for enc in ["utf-8", "gbk", "latin-1", "ISO-8859-1", "cp1252"]:
    try:
        df = pd.read_csv(path, encoding=enc, low_memory=False)
        print(f"✅ {enc}: {df.shape[0]} 行 × {df.shape[1]} 列")
        label_col = "Label" if "Label" in df.columns else df.columns[-1]
        print(f"   标签列 '{label_col}' 分布:")
        print(df[label_col].value_counts().to_string())
        break
    except Exception as e:
        print(f"❌ {enc}: {type(e).__name__}: {e}")