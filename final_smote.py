import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------------------------
# 1. 資料載入與準備
# ----------------------------------------------------
file_path = "pokemon_newtype.csv"

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

df.columns = df.columns.str.strip()

# 指定特徵與標籤
feature_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
target_col = "Type_group"

missing = [c for c in feature_cols + [target_col] if c not in df.columns]
if missing:
    print("❌ 你的 CSV 欄位如下：")
    print(df.columns.tolist())
    raise KeyError(f"缺少必要欄位：{missing}\n請確認 CSV 欄位名稱是否完全一致。")

X = df[feature_cols].copy()
y = df[target_col].copy()

# 去掉缺值
before = len(df)
data = pd.concat([X, y], axis=1).dropna()
after = len(data)
if after != before:
    print(f"⚠️ 偵測到缺值，已移除 {before - after} 筆資料（剩 {after} 筆）")

X = data[feature_cols]
y = data[target_col]

print("\n📌 類別分佈（全部資料）")
print(y.value_counts().sort_index())

# ----------------------------------------------------
# 2. 切分訓練集與測試集（Test 永遠不參與調參）
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n📌 類別分佈（Train）")
print(pd.Series(y_train).value_counts().sort_index())
print("\n📌 類別分佈（Test）")
print(pd.Series(y_test).value_counts().sort_index())

# ----------------------------------------------------
# 3. 建立 Pipeline + GridSearchCV 調超參數
# ----------------------------------------------------
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        random_state=42
    ))
])

# 你真正能調、也最有效的 LR 超參數主要是 C 與 class_weight
param_grid = {
    "logreg__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20],
    "logreg__class_weight": [None, "balanced"]
}

# 交叉驗證（分層）避免某折某類太少
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 想衝「準確率」就用 accuracy；如果老師在意小類別，用 f1_macro 更合理
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="accuracy",   # 你要更公平可改成 "f1_macro"
    cv=cv,
    n_jobs=-1,
    verbose=0
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("\n" + "="*60)
print("✅ GridSearch 完成")
print("Best Params:", grid.best_params_)
print(f"Best CV Score ({grid.scoring}): {grid.best_score_:.4f}")
print("="*60)

# ----------------------------------------------------
# 4. 用最佳模型評估 Train / Test（明確分開）
# ----------------------------------------------------
def evaluate(split_name, X_split, y_split, model):
    y_pred = model.predict(X_split)
    acc = accuracy_score(y_split, y_pred)
    print("\n" + "="*60)
    print(f"Classification Report ({split_name})")
    print("="*60)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_split, y_pred, digits=4))
    return y_pred

y_pred_train = evaluate("Train Set", X_train, y_train, best_model)
y_pred_test  = evaluate("Test Set",  X_test,  y_test,  best_model)

# ----------------------------------------------------
# 5. 係數輸出（Logistic Regression 才有）
# ----------------------------------------------------
logreg = best_model.named_steps["logreg"]
coef_df = pd.DataFrame(
    logreg.coef_,
    columns=feature_cols,
    index=[f"Class {c}" for c in logreg.classes_]
)
print("\nModel Coefficients:\n", coef_df)

coef_df.to_csv("final_coefficients.csv", encoding="utf-8-sig")
print("\n✅ 係數已儲存至 final_coefficients.csv (utf-8-sig)")

# ----------------------------------------------------
# 6. 混淆矩陣（Test）：Count + Normalized by True Label
# ----------------------------------------------------
classes = np.sort(y.unique())
cm = confusion_matrix(y_test, y_pred_test, labels=classes)

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix (Test Set) - Count", fontsize=16, fontweight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_count.png", dpi=300)
print("\n✅ 已輸出：confusion_matrix_count.png")

# 每列正規化（每一列加總=1）
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

plt.figure(figsize=(9, 7))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=classes, yticklabels=classes, vmin=0, vmax=1)
plt.title("Confusion Matrix (Test Set) - Normalized by True Label", fontsize=16, fontweight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_normalized.png", dpi=300)
print("✅ 已輸出：confusion_matrix_normalized.png")
