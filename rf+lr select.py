import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.preprocessing import MinMaxScaler
# 忽略警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="some_module")

# 读取 Excel 文件
file_path = r"D:\卷积网络\财务风险\重采样模型\新模型\随机抽取样本两年.xlsx"
data = pd.read_excel(file_path)

# 提取 y（第三列）
y = data.iloc[:, 2]

# 提取 X（第四列及之后的所有列）
X = data.iloc[:, 3:]

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

# 标准化 X 数据
# 处理缺失值
X = X.fillna(X.median())
#scaler = MinMaxScaler()
#X_scaled = scaler.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 使用随机森林筛选特征
rf = RandomForestClassifier(random_state=4)
rf.fit(X_scaled_df, y)

# 初始化变量来存储最佳准确率和所有达到最佳准确率的特征组合
best_accuracy = 0
best_feature_combinations = []

# 遍历不同的特征数量
for num_features in range(1, X_scaled_df.shape[1] + 1):
    # 使用 SelectFromModel 进行特征选择
    selector = SelectFromModel(rf, threshold="median", max_features=num_features, prefit=True)
    X_selected = selector.transform(X_scaled_df)

    # 获取选择的特征
    selected_features = X_scaled_df.columns[selector.get_support()]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=4)

    # 使用 Logistic 回归建模
    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(X_train, y_train)

    # 预测并评估模型
    y_pred = logistic.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 如果当前特征数量的准确率更高，则更新最佳准确率和特征组合
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_feature_combinations = [(num_features, selected_features, accuracy)]
    elif accuracy == best_accuracy:
        best_feature_combinations.append((num_features, selected_features, accuracy))

    # 打印当前特征数量和准确率
    print(f"特征数量: {num_features}, 准确率: {accuracy:.4f}")

# 输出最佳结果
print("\n最佳准确率:", best_accuracy)
print("达到最佳准确率的特征组合:")
for num_features, features, accuracy in best_feature_combinations:
    print(f"特征数量: {num_features}, 特征: {features.tolist()}, 准确率: {accuracy:.4f}")

# 保存所有达到最佳准确率的特征组合到 Excel 文件
for idx, (num_features, features, accuracy) in enumerate(best_feature_combinations):
    # 提取选择的特征数据
    selected_features_df = X_scaled_df[features]

    # 保存到 Excel 文件
    output_file_path = f"D:\卷积网络\财务风险\重采样模型\新模型\前推后两年筛选\森林诉讼1年前推后-{num_features}_{idx + 1}.xlsx"
    selected_features_df.to_excel(output_file_path, index=False)
    print(f"特征数量 {num_features} 的组合 {idx + 1} 已保存到: {output_file_path}")