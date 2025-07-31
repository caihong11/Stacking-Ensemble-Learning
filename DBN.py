import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             f1_score, confusion_matrix,
                             classification_report)

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

# DBN实现（包含RBM预训练和微调）
class DBN(nn.Module):
    def __init__(self, visible_dim, hidden_dims=[256, 128]):
        super().__init__()
        self.rbm_layers = nn.ModuleList()
        self.finetune_layers = nn.ModuleList()

        # 构建RBM层
        prev_dim = visible_dim
        for dim in hidden_dims:
            self.rbm_layers.append(RBM(prev_dim, dim))
            prev_dim = dim

        # 构建微调层
        prev_dim = visible_dim
        for dim in hidden_dims:
            self.finetune_layers.append(nn.Linear(prev_dim, dim))
            self.finetune_layers.append(nn.Sigmoid())
            prev_dim = dim
        self.finetune_layers.append(nn.Linear(prev_dim, 1))
        self.finetune_layers.append(nn.Sigmoid())

    def pretrain(self, X, epochs=10, batch_size=32):
        data = torch.FloatTensor(X)
        for i, rbm in enumerate(self.rbm_layers):
            print(f"Pretraining RBM layer {i + 1}...")
            rbm.train_rbm(data, epochs=epochs, batch_size=batch_size)
            with torch.no_grad():
                data = torch.sigmoid(F.linear(data, rbm.W, rbm.h_bias))

    def forward(self, x):
        for layer in self.finetune_layers:
            x = layer(x)
        return x.squeeze()


# RBM实现
class RBM(nn.Module):
    def __init__(self, visible_dim, hidden_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(hidden_dim, visible_dim) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.v_bias = nn.Parameter(torch.zeros(visible_dim))

    def sample_h(self, v):
        ph = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return ph, torch.bernoulli(ph)

    def sample_v(self, h):
        pv = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return pv, torch.bernoulli(pv)

    def train_rbm(self, data, epochs=10, batch_size=32, lr=0.01):
        optimizer = optim.SGD([self.W, self.h_bias, self.v_bias], lr=lr)
        for epoch in range(epochs):
            for batch in DataLoader(data, batch_size=batch_size, shuffle=True):
                v0 = batch
                # Contrastive Divergence (CD-1)
                ph0, h0 = self.sample_h(v0)
                v1, _ = self.sample_v(h0)
                ph1, _ = self.sample_h(v1)

                loss = torch.mean((v0 - v1) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


# 训练和评估函数
def train_dbn(X_train, y_train, X_test, y_test):
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.FloatTensor(y_train)
    y_test_tensor = torch.FloatTensor(y_test)

    # 初始化DBN
    dbn = DBN(visible_dim=X_train.shape[1], hidden_dims=[256, 128])

    # 预训练
    print("\n=== DBN Pretraining ===")
    dbn.pretrain(X_train, epochs=15, batch_size=32)

    # 微调
    print("\n=== DBN Fine-tuning ===")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(dbn.parameters(), lr=0.001)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=32, shuffle=True
    )

    for epoch in range(50):
        dbn.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = dbn(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    # 评估
    dbn.eval()
    with torch.no_grad():
        y_proba = dbn(X_test_tensor).numpy()
    y_pred = (y_proba >= 0.5).astype(int)

    return y_pred, y_proba


# 指标计算函数
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


# 主流程
if __name__ == "__main__":
    # 数据预处理（保持与之前相同）
    # ... [您的数据加载和预处理代码] ...

    # 训练和评估DBN
    y_pred, y_proba = train_dbn(X_train, y_train, X_test, y_test)

    # 输出结果
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    print("\n=== DBN评估结果 ===")
    print(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']))
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))