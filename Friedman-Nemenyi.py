import pandas as pd
import numpy as np
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare


def load_data(file_path):
    """读取Excel文件并整理为字典形式"""
    df = pd.read_excel(file_path, sheet_name=0)

    models = df.iloc[:, 0].unique()
    metrics = ['ACC', 'AUC', 'G-mean', 'F1-score', 'TypeⅠerror', 'TypeⅡerror','average']

    results = {metric: {model: [] for model in models} for metric in metrics}

    for _, row in df.iterrows():
        model = row.iloc[0]
        for i, metric in enumerate(metrics, start=1):
            val = row.iloc[i]
            # 确保数据为数值类型，否则转为NaN
            try:
                val = float(val) if not pd.isna(val) else np.nan
            except (ValueError, TypeError):
                val = np.nan
                print(f"警告：忽略非数值数据（模型={model}, {metric}={row.iloc[i]})")
            results[metric][model].append(val)

    return results


def friedman_nemenyi(data, metric_name, alpha=0.05):
    """执行Friedman检验和Nemenyi事后检验"""
    models = list(data.keys())

    # 检查数据完整性并填充NaN
    n_folds = max(len(data[model]) for model in models)
    data_matrix = []
    for model in models:
        model_data = data[model] + [np.nan] * (n_folds - len(data[model]))
        data_matrix.append(model_data)

    # 转换为NumPy数组并删除全为NaN的行
    data_matrix = np.array(data_matrix).T
    data_matrix = data_matrix[~np.isnan(data_matrix).all(axis=1)]

    if data_matrix.shape[0] < 2:
        raise ValueError(f"有效样本量不足（{data_matrix.shape[0]}折），至少需要2折")

    # Friedman检验（注意处理NaN）
    valid_cols = ~np.isnan(data_matrix).all(axis=0)
    if sum(valid_cols) < 2:
        raise ValueError("有效模型不足2个")

    friedman_stat, friedman_p = friedmanchisquare(
        *[data_matrix[:, i] for i in np.where(valid_cols)[0]]
    )

    print(f"\n=== {metric_name} Friedman检验 ===")
    print(f"有效样本量: {data_matrix.shape[0]}折")
    print(f"统计量: {friedman_stat:.3f}, p值: {friedman_p:.4f}")

    if friedman_p > alpha:
        print("未检测到显著差异")
        return None

    # Nemenyi检验（输入必须是排名矩阵）
    rank_matrix = data_matrix.argsort(axis=1).argsort(axis=1) + 1
    nemenyi = sp.posthoc_nemenyi_friedman(rank_matrix)
    nemenyi.columns = models
    nemenyi.index = models

    return nemenyi


if __name__ == "__main__":
    try:
        # 文件路径（使用原始字符串r""或双反斜杠）
        file_path = r""

        # 加载数据
        data = load_data(file_path)

        # 选择需要分析的指标
        metric_name = 'average'
        if metric_name not in data:
            raise ValueError(f"数据中未找到指标: {metric_name}")

        # 执行检验
        nemenyi = friedman_nemenyi(data[metric_name], metric_name)

        if nemenyi is not None:
            # 生成两两比较结果
            results = []
            for i in range(len(nemenyi)):
                for j in range(i + 1, len(nemenyi)):
                    results.append({
                        'Model1': nemenyi.index[i],
                        'Model2': nemenyi.columns[j],
                        'p-value': nemenyi.iloc[i, j],
                        'Significant': 'Yes' if nemenyi.iloc[i, j] < 0.05 else 'No'
                    })

            # 输出结果
            df_results = pd.DataFrame(results).sort_values('p-value')
            pd.set_option('display.max_rows', None)
            print("\nNemenyi检验两两比较结果：")
            print(df_results)

            # 保存结果
            output_path = f"{metric_name}_Nemenyi检验结果.xlsx"
            df_results.to_excel(output_path, index=False)
            print(f"\n结果已保存至: {output_path}")

    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        print("请检查：")
        print("1. 文件路径是否正确（建议使用原始字符串r'path'）")
        print("2. Excel数据是否包含非数值（如文本）")
        print("3. 每个模型的观测值数量是否一致")