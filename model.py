import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from deap import base, creator, tools, algorithms
import random
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# 读取数据
file_path = r""
data = pd.read_excel(file_path)

# 提取 y（第三列）和 X（第四列及之后的所有列）
y = data.iloc[:, 2]
X = data.iloc[:, 3:]
 #处理缺失值
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 转换为 NumPy 数组
X = X.values
y = y.values

X_res, y_res = SMOTE().fit_resample(X_scaled, y)

# 划分数据：80% 训练集，20% 测试集
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=48)

# 定义基分类器
base_models = {
    'LGBM': LGBMClassifier(random_state=55, verbosity=-1),
    'RandomForest': RandomForestClassifier(random_state=65),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=74),
    'LDA': LinearDiscriminantAnalysis(),
    'XGBoost': XGBClassifier(random_state=24),
    'GBDT': GradientBoostingClassifier(random_state=44)
}

# 生成元特征
def generate_meta_features(X_train, X_test, y_train, base_models, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=4)
    X_meta_train = np.zeros((X_train.shape[0], len(base_models)), dtype=np.float32)
    X_meta_test = np.zeros((X_test.shape[0], len(base_models)), dtype=np.float32)

    for i, (name, model) in enumerate(base_models.items()):
        print(f"Generating meta features for {name}...")
        test_preds = np.zeros((X_test.shape[0], n_folds))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold = y_train[train_idx]

            model_clone = clone(model)
            model_clone.fit(X_train_fold, y_train_fold)
            X_meta_train[val_idx, i] = model_clone.predict_proba(X_val_fold)[:, 1]
            test_preds[:, fold_idx] = model_clone.predict_proba(X_test)[:, 1]

        X_meta_test[:, i] = test_preds.mean(axis=1)

    return X_meta_train, X_meta_test

# 计算G-mean
def calculate_g_mean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity)

# 定义适应度函数
def evaluate_individual(individual):
    n_estimators, learning_rate, max_depth = individual

    model = LGBMClassifier(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        random_state=15,
        verbosity=-1
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=4)
    scores = []
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        scores.append(accuracy_score(y_val_fold, y_pred))

    return np.mean(scores),

# 定义变异操作
def custom_mutate(individual, indpb):
    individual[0] = random.randint(100, 500)  # n_estimators
    individual[2] = random.randint(3, 10)     # max_depth
    individual[1] = random.uniform(0.01, 0.2)  # learning_rate
    return individual,

# 遗传算法优化
def optimize_with_ga():
    POPULATION_SIZE = 30
    P_CROSSOVER = 0.9
    P_MUTATION = 0.1
    MAX_GENERATIONS = 20
    RANDOM_SEED = 56

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_n_estimators", random.randint, 100, 500)
    toolbox.register("attr_learning_rate", random.uniform, 0.01, 0.2)
    toolbox.register("attr_max_depth", random.randint, 3, 10)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                   (toolbox.attr_n_estimators, toolbox.attr_learning_rate, toolbox.attr_max_depth), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutate, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=POPULATION_SIZE)

    algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                       ngen=MAX_GENERATIONS, verbose=True)

    return tools.selBest(population, k=1)[0]

# 第一轮：训练集上训练基分类器并生成元特征
print("=== 第一轮：训练集上生成元特征 ===")
X_meta_train, _ = generate_meta_features(X_train, X_train, y_train, base_models)

# 合并原始特征和元特征
X_train_combined = np.hstack([X_train, X_meta_train])

# 遗传算法优化元分类器
print("\n=== 遗传算法优化元分类器 ===")
best_params = optimize_with_ga()
print(f"Best parameters: n_estimators={int(best_params[0])}, learning_rate={best_params[1]:.4f}, max_depth={int(best_params[2])}")

# 使用最优参数训练元分类器
best_model = LGBMClassifier(
    n_estimators=int(best_params[0]),
    learning_rate=best_params[1],
    max_depth=int(best_params[2]),
    random_state=42,
    verbosity=-1
)

# 训练元分类器
best_model.fit(X_train_combined, y_train)

# 第二轮：在整个数据集上生成元特征
print("\n=== 第二轮：测试集上生成元特征 ===")
# 合并训练集和测试集用于基分类器训练
X_full = np.vstack([X_train, X_test])
y_full = np.concatenate([y_train, y_test])

# 生成测试集的元特征
_, X_meta_test = generate_meta_features(X_full, X_test, y_full, base_models)

# 合并测试集的原始特征和元特征
X_test_combined = np.hstack([X_test, X_meta_test])

# 预测
y_pred_prob = best_model.predict_proba(X_test_combined)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)

# 评估
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
g_mean = calculate_g_mean(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Final evaluation result ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"G-mean: {g_mean:.4f}")
print(f"F1-score: {f1:.4f}")

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix,
                            index=['Actual Negative', 'Actual Positive'],
                            columns=['Predicted Negative', 'Predicted Positive'])
print("\nConfusion Matrix:")
print(conf_matrix_df)

# 详细指标
tn, fp, fn, tp = conf_matrix.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)

print("\nDetailed Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"False Positive Rate: {fp / (fp + tn):.4f}")
print(f"False Negative Rate: {fn / (fn + tp):.4f}")