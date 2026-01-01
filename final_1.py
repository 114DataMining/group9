import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------------------------
# 1. 資料載入與準備
# ----------------------------------------------------
file_path = "pokemon_newtype.csv"

# 自動嘗試常見編碼（避免 UnicodeDecodeError）
encodings = ["utf-8", "utf-8-sig", "cp950", "big5", "cp1252"]
df = None
used_encoding = None

for enc in encodings:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        used_encoding = enc
        break
    except UnicodeDecodeError:
        continue

if df is None:
    raise RuntimeError("讀取 CSV 失敗：請確認檔案編碼或檔案是否損壞。")

print(f"✅ CSV 讀取成功：{file_path} (encoding={used_encoding})")

# 欄位名稱清理（避免欄名有前後空白導致 KeyError）
df.columns = df.columns.str.strip()

# 指定的特徵與標籤
feature_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"] #自變數(Y)
target_col = "Type_group" #應變數(X)

# 檢查必要欄位是否存在
missing = [c for c in feature_cols + [target_col] if c not in df.columns]
if missing:
    print("❌ 你的 CSV 欄位如下：")
    print(df.columns.tolist())
    raise KeyError(f"缺少必要欄位：{missing}\n請確認 CSV 欄位名稱是否完全一致（包含大小寫、空白、符號）。")

# 取出 X / y
X = df[feature_cols].copy()
y = df[target_col].copy()

# 若有缺值，先刪掉（你也可以改成填補）
before = len(df)
data = pd.concat([X, y], axis=1).dropna()
after = len(data)
if after != before:
    print(f"⚠️ 偵測到缺值，已移除 {before - after} 筆資料（剩 {after} 筆）")

X = data[feature_cols]
y = data[target_col]

# ----------------------------------------------------
# 2. 切分訓練集與測試集
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------
# 3. 訓練邏輯斯回歸多元分類模型
#    加入 StandardScaler：讓 6 個能力值尺度一致，效果通常更穩
# ----------------------------------------------------
model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=42
    ))
])


model.fit(X_train, y_train)

# ----------------------------------------------------
# 4. 模型評估
# ----------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("=" * 40)
print("模型訓練結果 (使用 6 個能力值特徵)")
print("=" * 40)
print(f"Accuracy (準確度): {accuracy:.4f}")
print("\nClassification Report (分類報告):\n", report)

# ----------------------------------------------------
# 5. 輸出模型係數
#    注意：因為有 StandardScaler，係數是「標準化後特徵」的係數
# ----------------------------------------------------
logreg = model.named_steps["logreg"]

coefficients_df = pd.DataFrame(
    logreg.coef_,
    columns=feature_cols,
    index=[f"Class {c}" for c in logreg.classes_]
)

print("\nModel Coefficients :\n", coefficients_df)

coefficients_df.to_csv("final_coefficients.csv", encoding="utf-8-sig")
print("\n✅ 係數已儲存至 final_coefficients.csv (utf-8-sig)")

# ----------------------------------------------------
# 6. 混淆矩陣 (Confusion Matrix)
# ----------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

# 繪製混淆矩陣
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=logreg.classes_,
            yticklabels=logreg.classes_,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=20, fontweight='bold')
plt.ylabel('True Label', fontsize=20)
plt.xlabel('Predicted Label', fontsize=20)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✅ 混淆矩陣已儲存至 confusion_matrix.png")
