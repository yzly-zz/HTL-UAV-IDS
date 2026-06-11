"""
异构数据预处理引擎（增强版）。
- 自动探测 CSV 编码（utf-8 → gbk → latin-1 → cp1252）
- 智能标签列定位（支持 Label / label / Normal 等 12 种命名）
- 字符串标签自动二值化：Normal → 0，其余 → 1
- 仅保留数值列，隔离 IP/MAC 等文本列
- Inf / NaN → 零值插补
- Z-score 标准化
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

CSV_ENCODINGS = ["utf-8", "gbk", "latin-1", "ISO-8859-1", "cp1252"]

LABEL_CANDIDATES = [
    "Label", "label", "LABEL",
    "attack", "Attack", "ATTACK",
    "class", "Class", "CLASS",
    "type", "Type", "TYPE",
    "target", "Target",
    "Normal", "normal",
    "is_attack", "Is_Attack",
]


class HeterogeneousDataPreprocessor:
    def __init__(self, scaler_save_dir="weights/"):
        self.scaler = StandardScaler()
        self.scaler_save_dir = scaler_save_dir
        os.makedirs(self.scaler_save_dir, exist_ok=True)

    def _find_label_column(self, df):
        for candidate in LABEL_CANDIDATES:
            if candidate in df.columns:
                return candidate
        return None

    def _binarize_labels(self, y_series):
        if y_series.dtype in [np.int64, np.int32, np.float64, np.float32, np.int8]:
            return (y_series.values != 0).astype(int)
        y_str = y_series.astype(str).str.strip().str.lower()
        return (y_str != "normal").astype(int).values  # ← 加 .values，返回 numpy array

    def _clean_dataframe(self, df, label_col=None):
        # —— 回退：若指定列名不存在，切换自动探测 ——
        if label_col is not None and label_col not in df.columns:
            print(f"   ⚠️ 指定标签列 '{label_col}' 不存在，切换自动探测...")
            label_col = None

        # —— 自动探测 ——
        if label_col is None:
            label_col = self._find_label_column(df)
            if label_col is None:
                raise ValueError(
                    f"未找到标签列。当前列名: {df.columns.tolist()}\n"
                    f"候选列表: {LABEL_CANDIDATES}"
                )

        print(f"   🎯 标签列: '{label_col}'")

        # —— 分离 ——
        y_raw = df[label_col]
        X_df = df.drop(columns=[label_col], errors="ignore")

        # —— 二值化 ——
        y = self._binarize_labels(y_raw)
        unique, counts = np.unique(y, return_counts=True)
        print(f"   📊 标签分布: {dict(zip(unique.astype(int), counts))}  (0=正常, 1=攻击)")

        # —— 仅保留数值列 ——
        X_numeric = X_df.select_dtypes(include=[np.number]).copy()
        dropped_cols = set(X_df.columns) - set(X_numeric.columns)
        if dropped_cols:
            print(f"   ⚠️ 已隔离 {len(dropped_cols)} 个非数值列: {list(dropped_cols)[:6]}...")

        # —— Inf → NaN → 0 ——
        X_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_numeric.fillna(0, inplace=True)

        feature_names = X_numeric.columns.tolist()
        X_raw = X_numeric.values.astype(np.float64)

        return X_raw, y, feature_names

    def process_domain_data(
        self, file_path, domain_name, is_training=True, label_col=None, sample_frac=0.1
    ):
        print(f"\n[{domain_name.upper()}] 加载: {file_path}")

        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
            if sample_frac is not None:
                df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
                print(f"   ⚠️ 已采样 {sample_frac:.0%}，当前 {len(df):,} 行")
        elif file_path.endswith(".csv"):
            df = None
            for enc in CSV_ENCODINGS:
                try:
                    df = pd.read_csv(file_path, low_memory=False, encoding=enc)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            if df is None:
                raise ValueError(
                    f"无法解码 CSV: {file_path}\n"
                    f"尝试的编码: {CSV_ENCODINGS}\n"
                    f"请运行 python scripts/detect_encoding.py 获取实际编码。"
                )
        else:
            raise ValueError("仅支持 .parquet 或 .csv")

        X_raw, y, feature_names = self._clean_dataframe(df, label_col=label_col)
        input_dim = X_raw.shape[1]
        print(f"   有效数值特征维度: {input_dim}")

        scaler_path = os.path.join(self.scaler_save_dir, f"{domain_name}_scaler.pkl")

        if is_training:
            X_scaled = self.scaler.fit_transform(X_raw)
            joblib.dump(self.scaler, scaler_path)
            print(f"   ✅ Scaler → {scaler_path}")
        else:
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                X_scaled = self.scaler.transform(X_raw)
            else:
                raise FileNotFoundError(f"Scaler 不存在: {scaler_path}")

        return X_scaled, y, input_dim