from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 继续使用之前预处理的数据 (X_train, X_test, y_train, y_test)
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE


pd.set_option('display.max_columns', None)
# 读取数据
file_path = r""
data = pd.read_excel(file_path)

# 提取 y（第三列）和 X（第四列及之后的所有列）
y = data.iloc[:, 2]
X = data.iloc[:, 3:]

X = X.fillna(X.median())
# 处理非数值列
def convert_to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if '亿' in str(value):
        return float(str(value).replace('亿', '')) * 1e8
    elif '万' in str(value):
        return float(str(value).replace('万', '')) * 1e4
    else:
        return float(value)


for col in X.columns:
    X[col] = X[col].apply(convert_to_float)

# 标准化和重采样
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_res, y_res = SMOTE().fit_resample(X_scaled, y)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=8965)

# 定义AdaBoost和随机森林模型
additional_models = {
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}


# 扩展的评估函数（与之前相同）
def extended_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    gmean = np.sqrt(sensitivity * specificity)
    type1_error = fp / (fp + tn)  # False Positive Rate
    type2_error = fn / (fn + tp)  # False Negative Rate

    return {
        'ACC': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_proba),
        'G-MEAN': gmean,
        'F1-SCORE': f1_score(y_true, y_pred),
        'TYPE1ERROR': type1_error,
        'TYPE2ERROR': type2_error
    }


# 统一评估函数（添加特征重要性输出）
def evaluate_ensemble_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = extended_metrics(y_test, y_pred, y_proba)
    print(f"\n=== {type(model).__name__} ===")
    print(classification_report(y_test, y_pred))
    print(pd.DataFrame([metrics]).T.rename(columns={0: 'Value'}))

    # 输出特征重要性
    if hasattr(model, 'feature_importances_'):
        print("\n特征重要性:")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for idx in indices:
            print(f"{data.columns[1:][idx]}: {importances[idx]:.4f}")

    return metrics


# 执行评估
ensemble_results = []
for name, model in additional_models.items():
    print(f"\n{'=' * 30}\nEvaluating {name}\n{'=' * 30}")
    metrics = evaluate_ensemble_model(model, X_train, y_train, X_test, y_test)
    metrics['Model'] = name
    ensemble_results.append(metrics)

# 合并所有结果（包含之前的基础模型）
all_results = ensemble_results
results_df = pd.DataFrame(all_results).set_index('Model')

# 按AUC排序输出
print("\n=== 所有模型性能对比 ===")
print(results_df[['ACC', 'AUC', 'G-MEAN', 'F1-SCORE', 'TYPE1ERROR', 'TYPE2ERROR']]
      .sort_values('AUC', ascending=False))

# 可视化特征重要性（以随机森林为例）
import matplotlib.pyplot as plt

rf = additional_models['Random Forest']
rf.fit(X_train, y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[-10:]  # 取最重要的10个特征

plt.figure(figsize=(10, 6))
plt.title('Random Forest - Top 10 Feature Importance')
plt.barh(range(10), importances[indices], align='center')
plt.yticks(range(10), [data.columns[1:][i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()