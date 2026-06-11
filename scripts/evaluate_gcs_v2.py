"""评估优化后的 GCS 模型"""
import torch, numpy as np, pandas as pd, joblib, json, os, sys
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_engine.preprocessor import HeterogeneousDataPreprocessor
from src.data_engine.dataset import IDSStreamDataset
from src.models.hda_1dcnn import ProjBridge

device = "cuda"
suffix = "gcs"
print("加载 GCS 优化模型...")
model = ProjBridge(input_dim=44, shared_dim=64, num_classes=2).to(device)
model.load_state_dict(torch.load(f"weights/target_uav_model_{suffix}.pth"))
model.eval()

df = pd.read_csv("data/raw/UAV-NIDD/GSC Case3 Label.csv")
prep = HeterogeneousDataPreprocessor()
X_raw, y, _ = prep._clean_dataframe(df, label_col=None)
_, X_test, _, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)
scaler = joblib.load(f"weights/target_scaler_{suffix}.pkl")
X_test_scaled = scaler.transform(X_test)

loader = DataLoader(IDSStreamDataset(X_test_scaled, y_test), batch_size=512)
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in loader:
        inputs = inputs.to(device, dtype=torch.float32)
        all_preds.extend(torch.argmax(model(inputs), 1).cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
print("\n" + "="*50)
print("  GCS 优化模型评估（92.54% 版本）")
print("="*50)
print(classification_report(all_labels, all_preds, target_names=["Normal", "Attack"], digits=4))
cm = confusion_matrix(all_labels, all_preds)
print(f"混淆矩阵:")
print(f"  实际Normal→预测Normal: {cm[0,0]:>6d}  |  实际Normal→预测Attack: {cm[0,1]:>6d}")
print(f"  实际Attack→预测Normal: {cm[1,0]:>6d}  |  实际Attack→预测Attack: {cm[1,1]:>6d}")
print(f"\nAccuracy: {(all_preds==all_labels).mean():.4f}")
print(f"AUC-ROC:  {roc_auc_score(all_labels, all_preds):.4f}")