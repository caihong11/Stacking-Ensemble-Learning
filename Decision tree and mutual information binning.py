import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# 读取数据
file_path = r""
data = pd.read_excel(file_path, sheet_name=0)
y = data.iloc[:, 2]
X = data.iloc[:, 3:]


# 处理缺失值和非数值列
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
X = X.fillna(X.median())


# 1. 基于决策树的最优划分
def discretize_with_tree(X, y, max_bins=3):
    discrete_df = pd.DataFrame(index=X.index)
    split_points = {}

    for col in X.columns:
        tree = DecisionTreeClassifier(max_leaf_nodes=max_bins, min_samples_leaf=0.1)
        tree.fit(X[[col]], y)

        threshold = tree.tree_.threshold[tree.tree_.threshold != -2]
        threshold = np.unique(threshold)
        split_points[col] = threshold

        if len(threshold) == 0:
            discrete_df[col] = (X[col] > X[col].median()).astype(int)
        else:
            bins = [-np.inf] + sorted(threshold) + [np.inf]
            discrete_df[col] = pd.cut(X[col], bins=bins, labels=False)

    return discrete_df, split_points


# 2. 基于信息增益比的最优分箱
def discretize_with_mutual_info(X, y, n_bins_range=range(2, 6)):
    discrete_df = pd.DataFrame(index=X.index)
    best_bins_info = {}

    for col in X.columns:
        max_info = -1
        best_n = 2

        for n in n_bins_range:
            discretizer = KBinsDiscretizer(n_bins=n, encode='ordinal', strategy='quantile')
            discretized = discretizer.fit_transform(X[[col]])
            info = mutual_info_classif(discretized, y, random_state=42)[0]

            if info > max_info:
                max_info = info
                best_n = n

        best_bins_info[col] = {'best_n_bins': best_n, 'mutual_info': max_info}
        discretizer = KBinsDiscretizer(n_bins=best_n, encode='ordinal', strategy='quantile')
        discrete_df[col] = discretizer.fit_transform(X[[col]]).flatten()

    return discrete_df, best_bins_info


# 执行两种离散化方法
tree_discrete, tree_splits = discretize_with_tree(X, y)
mi_discrete, mi_info = discretize_with_mutual_info(X, y)


# 计算IV值评估离散化效果
# 替换原calculate_iv函数
def calculate_iv(discrete_features, target):
    iv_values = {}
    woe_info = {}

    for col in discrete_features.columns:
        temp_df = pd.DataFrame({
            'feature': discrete_features[col],
            'target': target
        })

        # 分组统计（添加平滑）
        grouped = temp_df.groupby('feature')['target'].agg(['count', 'sum'])
        grouped.columns = ['total', 'bad']
        grouped['good'] = grouped['total'] - grouped['bad']

        # 平滑处理（避免零除）
        grouped['bad'] += 0.5
        grouped['good'] += 0.5

        # 计算分布
        total_bad = grouped['bad'].sum()
        total_good = grouped['good'].sum()
        grouped['bad_dist'] = grouped['bad'] / total_bad
        grouped['good_dist'] = grouped['good'] / total_good

        # 计算WOE和IV
        grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
        grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']

        iv_values[col] = grouped['iv'].sum()
        woe_info[col] = grouped[['woe', 'iv']]

    return pd.Series(iv_values), woe_info


# 重新计算IV值
tree_iv, tree_woe = calculate_iv(tree_discrete, y)
mi_iv, mi_woe = calculate_iv(mi_discrete, y)



# 准备输出结果
output_path = r""

with pd.ExcelWriter(output_path) as writer:
    # 原始数据
    X.to_excel(writer, sheet_name='Original_Data')

    # 决策树离散化结果
    tree_discrete.to_excel(writer, sheet_name='Tree_Discretized')

    # 决策树划分点
    pd.DataFrame.from_dict(tree_splits, orient='index').to_excel(
        writer, sheet_name='Tree_Split_Points')

    # 互信息离散化结果
    mi_discrete.to_excel(writer, sheet_name='MI_Discretized')

    # 互信息分箱信息
    pd.DataFrame.from_dict(mi_info, orient='index').to_excel(
        writer, sheet_name='MI_Binning_Info')

    # IV值比较
    iv_comparison = pd.DataFrame({
        'Tree_Discretization_IV': tree_iv,
        'MI_Discretization_IV': mi_iv
    })
    iv_comparison.to_excel(writer, sheet_name='IV_Comparison')

    # 添加WOE信息
    for col in tree_woe.keys():
        tree_woe[col].to_excel(writer, sheet_name=f'Tree_WOE_{col[:20]}')
        mi_woe[col].to_excel(writer, sheet_name=f'MI_WOE_{col[:20]}')

print(f"离散化结果已保存到: {output_path}")


# 输出前十个重要特征
def print_top_features(iv_series, method_name):
    print(f"\n{method_name} - 前10个重要特征:")
    top_features = iv_series.sort_values(ascending=False).head(10)
    for i, (feature, iv) in enumerate(top_features.items(), 1):
        print(f"{i}. {feature}: {iv:.4f}")


print_top_features(tree_iv, "决策树离散化")
print_top_features(mi_iv, "互信息离散化")
