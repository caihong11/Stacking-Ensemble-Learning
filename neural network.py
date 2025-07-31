import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, accuracy_score, f1_score
)
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 数据预处理部分
# ==============================
file_path = r""
data = pd.read_excel(file_path)

# 提取列名（在转换为数组前）
feature_columns = data.columns[3:]  # 保存特征列名供后续使用

# 确保 y 转换为 numpy 数组
y = data.iloc[:, 2].values  # 使用 .values 转换为 numpy array
X = data.iloc[:, 3:]  # 保持为DataFrame进行预处理
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


# 对每列应用转换
for col in X.columns:
    X[col] = X[col].apply(convert_to_float)

# 转换为NumPy数组
X = X.values

# 标准化和重采样
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_res, y_res = SMOTE().fit_resample(X_scaled, y)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=7
)

# 转换为PyTorch张量
# ==============================
X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # [N, 1, features]
X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
y_train_tensor = torch.FloatTensor(y_train.astype(np.float32))
y_test_tensor = torch.FloatTensor(y_test.astype(np.float32))


# CNN模型定义
# ==============================
class FinancialCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        # 计算全连接层输入维度
        self.fc_input_dim = self._get_fc_input_dim(input_dim)
        self.fc1 = nn.Linear(self.fc_input_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def _get_fc_input_dim(self, input_dim):
        # 模拟计算经过卷积和池化后的维度
        x = torch.randn(1, 1, input_dim)
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.pool2(self.bn2(self.conv2(x)))
        return x.view(1, -1).shape[1]

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


# 训练和评估
# ==============================
def calculate_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'ACC': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_proba),
        'G-MEAN': np.sqrt((tp / (tp + fn)) * (tn / (tn + fp))),
        'F1': f1_score(y_true, y_pred),
        'TypeI': fp / (fp + tn),
        'TypeII': fn / (fn + tp)
    }


def train_and_evaluate():
    # 初始化模型
    input_dim = X_train.shape[1]
    model = FinancialCNN(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 数据加载
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 训练循环
    model.train()
    for epoch in range(50):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # 评估
    model.eval()
    with torch.no_grad():
        y_proba = model(X_test_tensor).squeeze().numpy()
    y_pred = (y_proba >= 0.5).astype(int)

    # 输出结果
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    print("\n=== 评估结果 ===")
    print(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']))
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_and_evaluate()