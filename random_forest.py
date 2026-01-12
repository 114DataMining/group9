# 導入函式庫
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from collections import defaultdict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# 讀取資料
data = pd.read_csv("pokemon_newtype_zscore.csv")

#定義變數
x = data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
y = data['Type_group']

#切分訓練集與測試集
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#參數設定
rf_model = RandomForestClassifier(
    n_estimators=401,
    max_depth=6,
    bootstrap=True,
    max_samples=0.8,
    min_samples_leaf=1,
    random_state=42,
    max_features=None,
    class_weight='balanced'
)

#訓練模型
rf_model.fit(x_train, y_train)

#模型預測
y_train_pred = rf_model.predict(x_train)
y_test_pred = rf_model.predict(x_test)

#模型評估
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("訓練集準確率 (Train Accuracy):", train_accuracy)
print("測試集準確率 (Test Accuracy):", test_accuracy)

train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

print("訓練集混淆矩陣 (Train Confusion Matrix):\n", train_cm)
print("測試集混淆矩陣 (Train Confusion Matrix):\n", test_cm)

print("訓練集 分類報告 (Classification Report):\n")
print(classification_report(y_train, y_train_pred))
print("測試集 分類報告 (Classification Report):\n")
print(classification_report(y_test, y_test_pred))

# 特徵重要度分析
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=x.columns
).sort_values(ascending=False)

print("\n特徵重要度：")
print(feature_importance)

# 五折交叉驗證
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
reports = []
conf_matrices = []
fold = 1

for train_idx, test_idx in skf.split(x, y):
    print(f"\n========== 第 {fold} 折 分類報告 ==========")

    X_train, X_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 顯示訓練集類別比例
    train_counts = y_train.value_counts().sort_index()
    train_ratios = y_train.value_counts(normalize=True).sort_index()

    print(f"訓練集樣本總數: {len(y_train)}")
    print("訓練集各類別佔比：")
    for cls, count in train_counts.items():
        ratio = train_ratios[cls] * 100
        print(f"  類別 {cls}: {count} 筆 ({ratio:.2f}%)")

    # 訓練模型
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    # 模型預測
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)

    # 顯示訓練集與測試集的分類報告
    print("\n[訓練集] 分類報告：")
    print(classification_report(y_train, y_train_pred))

    print("[測試集] 分類報告：")
    test_report = classification_report(y_test, y_test_pred)
    print(test_report)

    # 存成 dict，後面算平均用
    reports.append(
        classification_report(y_test, y_pred, output_dict=True)
    )

    fold += 1


#五折平均分類報告
avg_report = defaultdict(dict)

for label in reports[0].keys():
    if label == "accuracy":
        mean = np.mean([r[label] for r in reports])
        std = np.std([r[label] for r in reports])

        # sklearn 的 accuracy 是顯示在 precision 欄
        avg_report["accuracy"]["precision"] = f"{mean:.3f} ± {std:.3f}"
        avg_report["accuracy"]["recall"] = ""
        avg_report["accuracy"]["f1-score"] = ""
        avg_report["accuracy"]["support"] = ""
    else:
        for metric in ["precision", "recall", "f1-score"]:
            values = [r[label][metric] for r in reports]
            mean = np.mean(values)
            std = np.std(values)
            avg_report[label][metric] = f"{mean:.3f} ± {std:.3f}"

        # support：取平均（不算 std）
        supports = [r[label]["support"] for r in reports]
        avg_report[label]["support"] = f"{np.mean(supports):.1f}"

# 轉成 DataFrame
avg_report_df = pd.DataFrame(avg_report).T

# 欄位順序與 sklearn classification_report 一致
avg_report_df = avg_report_df[
    ["precision", "recall", "f1-score", "support"]
]

print("\n========== 五折平均 ± 標準差 分類報告 ==========")
print(avg_report_df)

# 繪製混淆矩陣圖
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 輸入你的測試集數據
data = np.array([[17, 12, 4, 0],
                 [4, 51, 6, 0],
                 [5, 19, 14, 1],
                 [7, 16, 1, 3]])

plt.figure(figsize=(8, 6))
sns.heatmap(data, annot=True, fmt='d', cmap='Blues',
            xticklabels=['1', '2', '3', '4'],
            yticklabels=['1', '2', '3', '4'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Test Set Confusion Matrix')
plt.show()



# 繪製PR圖
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy import interpolate

# 1. 準備繪圖與資料暫存
n_classes = 4
classes = [1, 2, 3, 4]
# 用於儲存每一折的 Precision 資料以便後續計算平均
mean_recall = np.linspace(0, 1, 100)
# 結構：precisions_per_class[class_idx] = [fold1_prec, fold2_prec, ...]
precisions_per_class = {i: [] for i in range(n_classes)}
aps_per_class = {i: [] for i in range(n_classes)}
mAPs = []

# 設定畫布：每一折一張圖 (5折) + 最後一張平均圖 = 6張
fig_folds, axes_folds = plt.subplots(2, 3, figsize=(18, 10))
axes_folds = axes_folds.flatten()

# --- 修改你的五折交叉驗證迴圈 ---
fold = 1
for train_idx, test_idx in skf.split(x, y):
    X_train_fold, X_test_fold = x.iloc[train_idx], x.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    # 訓練模型
    rf_model.fit(X_train_fold, y_train_fold)
    
    # 取得預測機率 (重要：PR曲線需要機率)
    y_score = rf_model.predict_proba(X_test_fold)
    
    # 將標籤二值化 (OvR)
    y_test_bin = label_binarize(y_test_fold, classes=classes)
    
    current_ax = axes_folds[fold-1]
    fold_aps = []

    for i in range(n_classes):
        # 計算該折、該類別的 PR
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
        fold_aps.append(ap)
        aps_per_class[i].append(ap)
        
        # 繪製該折的 PR 曲線
        current_ax.plot(recall, precision, label=f'Class {classes[i]} (AP={ap:.2f})')
        
        # 插值以便後續計算平均值
        f = interpolate.interp1d(recall[::-1], precision[::-1], bounds_error=False, fill_value=(1, 0))
        precisions_per_class[i].append(f(mean_recall))

    # 計算該折的 mAP
    current_mAP = np.mean(fold_aps)
    mAPs.append(current_mAP)
    
    current_ax.set_title(f'Fold {fold} PR Curve (mAP={current_mAP:.2f})')
    current_ax.set_xlabel('Recall')
    current_ax.set_ylabel('Precision')
    current_ax.legend(loc='lower left', fontsize='small')
    current_ax.grid(True, alpha=0.3)
    
    fold += 1

# 移除第六個多餘的空白子圖
fig_folds.delaxes(axes_folds[5])
plt.tight_layout()
plt.show()

# --- 繪製最後的平均 PR 曲線 (Mean ± Std) ---
plt.figure(figsize=(10, 8))

for i in range(n_classes):
    # 計算平均與標準差
    mean_prec = np.mean(precisions_per_class[i], axis=0)
    std_prec = np.std(precisions_per_class[i], axis=0)
    
    mean_ap = np.mean(aps_per_class[i])
    std_ap = np.std(aps_per_class[i])
    
    # 確保曲線從 Recall=0, Precision=1 開始 (視覺美化)
    mean_prec = np.clip(mean_prec, 0, 1)
    
    # 繪製平均曲線
    plt.plot(mean_recall, mean_prec, 
             label=f'Class {classes[i]} (Mean AP = {mean_ap:.2f} ± {std_ap:.2f})',
             lw=2)
    
    # 繪製標準差區域 (fill_between)
    prec_upper = np.minimum(mean_prec + std_prec, 1)
    prec_lower = np.maximum(mean_prec - std_prec, 0)
    plt.fill_between(mean_recall, prec_lower, prec_upper, alpha=0.1)

overall_mAP = np.mean(mAPs)
overall_std_mAP = np.std(mAPs)

plt.title(f'Overall Mean PR Curve across 5 Folds\nTotal mAP = {overall_mAP:.3f} ± {overall_std_mAP:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n總體平均 mAP: {overall_mAP:.4f} ± {overall_std_mAP:.4f}")







