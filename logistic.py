import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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


# 计算G-mean和错误类型
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


# 模型列表
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear Discriminant": LinearDiscriminantAnalysis(),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Multilayer Perceptron": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=88)
}


# 统一评估函数
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test)

    metrics = extended_metrics(y_test, y_pred, y_proba)
    print(f"\n=== {type(model).__name__} ===")
    print(classification_report(y_test, y_pred))
    print(pd.DataFrame([metrics]).T.rename(columns={0: 'Value'}))

    # 输出特征重要性（如果存在）
    if hasattr(model, 'coef_'):
        print("\n特征重要性（系数绝对值）:")
        for feat, coef in zip(data.columns[3:], np.abs(model.coef_[0])):
            print(f"{feat}: {coef:.4f}")
    elif hasattr(model, 'feature_importances_'):
        print("\n特征重要性:")
        for feat, imp in zip(data.columns[3:], model.feature_importances_):
            print(f"{feat}: {imp:.4f}")

    return metrics


# 执行所有模型评估
results = []
for name, model in models.items():
    print(f"\n{'=' * 30}\nEvaluating {name}\n{'=' * 30}")
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    metrics['Model'] = name
    results.append(metrics)

# 汇总结果
results_df = pd.DataFrame(results).set_index('Model')
print("\n=== 模型性能对比 ===")
print(results_df[['ACC', 'AUC', 'G-MEAN', 'F1-SCORE', 'TYPE1ERROR', 'TYPE2ERROR']])