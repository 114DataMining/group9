import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE

data = pd.read_csv("pokemon_newtype_zscore.csv")

X = data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
y = data['Type_group']
classes = np.unique(y)

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("訓練集 SMOTE 後類別分布:")
print(pd.Series(y_train_res).value_counts())

def custom_random_forest(
    X_train, y_train, X_test, y_test,
    m_trees=401, n_samples=0.8, i_features=6,
    max_depth=6, odd_best_k=31
):
    forest = []
    feature_importances = []

    for _ in range(m_trees):
        idx = np.random.choice(len(X_train), int(n_samples*len(X_train)), replace=True)
        feats = np.random.choice(X_train.columns, i_features, replace=False)
        X_s, y_s = X_train.iloc[idx][feats], y_train.iloc[idx]

        tree = RandomForestClassifier(n_estimators=1, max_depth=max_depth, bootstrap=False)
        tree.fit(X_s, y_s)
        forest.append((tree, feats))
        feature_importances.append(pd.Series(tree.feature_importances_, index=feats))

    # 選擇最佳樹
    scores = [(tree.score(X_test[feats], y_test), tree, feats) for tree, feats in forest]
    scores.sort(reverse=True, key=lambda x: x[0])
    best_trees = scores[:odd_best_k]

    # 多數決預測
    def vote_predict(X):
        preds, probas = [], []
        for _, t, f in best_trees:
            preds.append(t.predict(X[f]))
            probas.append(t.predict_proba(X[f]))
        final_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, preds)
        return final_pred, np.mean(probas, axis=0)
    pred_train, _ = vote_predict(X_train)
    pred_test, proba_test = vote_predict(X_test)
    feat_imp = pd.concat(feature_importances, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
    return pred_train, pred_test, proba_test, feat_imp

pred_train_all, pred_test_all, proba_test_all, feat_imp_all = custom_random_forest(
    X_train_res, y_train_res, X_test, y_test
)

print("===== 總體模型評估 =====")
print("訓練集準確率:", accuracy_score(y_train_res, pred_train_all))
print("測試集準確率:", accuracy_score(y_test, pred_test_all))

print("\n訓練集混淆矩陣")
print(confusion_matrix(y_train_res, pred_train_all))

print("\n測試集混淆矩陣")
print(confusion_matrix(y_test, pred_test_all))

print("\n訓練集分類報告")
print(classification_report(y_train_res, pred_train_all))

print("\n測試集分類報告")
print(classification_report(y_test, pred_test_all))

print("\n特徵重要度")
print(feat_imp_all)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

mean_recall = np.linspace(0, 1, 100)
interp_precisions = []
fold_maps = []
reports = []

for fold, (tr_idx, te_idx) in enumerate(skf.split(X_train_res, y_train_res), 1):
    print(f"\n===== Fold {fold} =====")
    X_tr, X_te = X_train_res.iloc[tr_idx], X_train_res.iloc[te_idx]
    y_tr, y_te = y_train_res.iloc[tr_idx], y_train_res.iloc[te_idx]

    print("訓練樣本數:", len(y_tr))
    print("類別比例:\n", y_tr.value_counts(normalize=True))

    pred_tr, pred_te, proba_te, _ = custom_random_forest(X_tr, y_tr, X_te, y_te)

    print("\nTrain Report")
    print(classification_report(y_tr, pred_tr))

    print("\nTest Report")
    report = classification_report(y_te, pred_te, output_dict=True)
    print(classification_report(y_te, pred_te))
    reports.append(report)

    # PR Curve
    y_bin = label_binarize(y_te, classes=classes)
    plt.figure(figsize=(6,5))
    ap_scores = []
    macro_precisions = []

    for i, c in enumerate(classes):
        p, r, _ = precision_recall_curve(y_bin[:,i], proba_te[:,i])
        ap = average_precision_score(y_bin[:,i], proba_te[:,i])
        ap_scores.append(ap)
        plt.plot(r, p, lw=1, label=f"Class {c} (AP={ap:.3f})")
        interp_p = np.interp(mean_recall, r[::-1], p[::-1])
        macro_precisions.append(interp_p)

    macro_precision = np.mean(macro_precisions, axis=0)
    interp_precisions.append(macro_precision)
    fold_map = np.mean(ap_scores)
    fold_maps.append(fold_map)

    plt.plot(mean_recall, macro_precision, lw=3, linestyle="--", color="black", label=f"Macro PR (mAP={fold_map:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Fold {fold} PR Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def mean_std_classification_table_full(reports, classes):
    metrics = ['precision','recall','f1-score']
    index = list(classes) + ['accuracy','macro avg','weighted avg']
    table_data = {m: [] for m in metrics}

    for item in index:
        item_str = str(item)
        for m in metrics:
            vals = []
            for r in reports:
                if item_str == 'accuracy':
                    vals.append(r['accuracy'])
                else:
                    vals.append(r[item_str][m])
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            table_data[m].append(f"{mean_val:.3f} ± {std_val:.3f}")

    df = pd.DataFrame(table_data, index=index)
    return df

df_full_report = mean_std_classification_table_full(reports, classes)
print("\n===== 五折平均 ± 標準差完整分類報告 =====")
print(df_full_report)
