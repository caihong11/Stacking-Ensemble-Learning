import re
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from openai import OpenAI
from PyPDF2 import PdfReader
from datetime import datetime
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 初始化DeepSeek客户端
client = OpenAI(
    base_url='model',
    api_key='{your API}'  # 替换为实际API密钥
)


class ResearchReportAnalyzer:
    def __init__(self):
        # 情感分析权重配置
        self.sentiment_weights = {
            'intensity': 0.4,  # 情感强度
            'certainty': 0.3,  # 确定性
            'risk_balance': 0.3  # 风险平衡
        }

    def extract_text_from_pdf(self, pdf_path):
        """使用PyPDF2提取PDF文本内容"""
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"PDF解析失败: {str(e)}")
            return None

    def analyze_sentiment(self, text):
        """调用大模型进行情感分析"""
        prompt = f"""：
        {{
            "intensity": 0-10分,    
            "certainty": 0-1,      
            "risk_balance": 0-5分, 
            "keywords": [           
                ["关键词1", 0.8],
                ["关键词2", -0.3]
            ]
        }}

        研报内容（节选）：
        {text[:2000]}..."""  # 限制输入长度

        try:
            response = client.chat.completions.create(
                model="model",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2  # 降低输出随机性
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return None

    def extract_financials(self, text):
        """正则提取财务数据"""
        results = {}
        for key, pattern in self.financial_patterns.items():
            match = re.search(pattern, text)
            results[key] = float(match.group(1)) if match else np.nan
        return results

    def detect_risks(self, text):
        """风险关键词扫描"""
        risk_counts = defaultdict(int)
        for risk_type, keywords in self.risk_keywords.items():
            for kw in keywords:
                risk_counts[risk_type] += len(re.findall(kw, text, re.IGNORECASE))
        return dict(risk_counts)

    def visualize_results(self, analysis_result):
        plt.figure(figsize=(12, 10))

        # 修复雷达图数据维度
        categories = ['情感强度', '确定性', '风险平衡']
        values = [
            analysis_result['sentiment']['intensity'],
            analysis_result['sentiment']['certainty'] * 10,
            analysis_result['sentiment']['risk_balance'] * 2
        ]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        # 闭合曲线（去掉values和angles的重复追加）
        values = np.concatenate((values, [values[0]]))  # 闭合曲线
        angles = np.concatenate((angles, [angles[0]]))  # 闭合角度

        plt.subplot(2, 2, 1, polar=True)
        plt.plot(angles, values, 'o-')
        plt.fill(angles, values, alpha=0.25)
        plt.thetagrids(np.degrees(angles[:-1]), categories)
        plt.title('情感分析雷达图', pad=20)

        # 财务数据柱状图
        plt.subplot(2, 2, 2)
        financials = analysis_result['financials']
        labels = ['营收(亿)', '净利润(亿)', '增长率(%)']
        values = [financials['revenue'], financials['net_profit'], financials['growth_rate']]
        plt.bar(labels, values, color=['#4CAF50', '#2196F3', '#FFC107'])
        for i, v in enumerate(values):
            plt.text(i, v, f"{v:.1f}", ha='center', va='bottom')
        plt.title('核心财务指标')

        # 风险词云
        plt.subplot(2, 2, 3)
        risks = analysis_result['risks']
        plt.barh(list(risks.keys()), list(risks.values()), color='#F44336')
        plt.title('风险关键词频率')

        # 综合评分
        plt.subplot(2, 2, 4)
        composite_score = (
                0.4 * analysis_result['sentiment']['intensity'] +
                0.3 * (analysis_result['financials']['growth_rate'] / 20) * 10 +
                0.3 * (5 - len(analysis_result['risks']))
        )
        plt.gca().axis('off')
        plt.text(0.5, 0.6, f"综合评分: {composite_score:.1f}/10",
                 ha='center', va='center', fontsize=20)
        plt.text(0.5, 0.4, "评级: " +
                 ("强烈推荐" if composite_score > 8 else "推荐" if composite_score > 6 else "中性"),
                 ha='center', va='center', fontsize=16)

        plt.tight_layout()
        plt.savefig('analysis_report.png')
        plt.close()

    def generate_report(self, pdf_path):
        """生成完整分析报告"""
        # 1. 提取文本
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None

        # 2. 执行分析
        analysis_result = {
            "sentiment": self.analyze_sentiment(text),
            "financials": self.extract_financials(text),
            "risks": self.detect_risks(text),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        }

        # 3. 生成可视化
        self.visualize_results(analysis_result)

        # 4. 保存JSON报告
        with open('analysis_result.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)

        return analysis_result


# 使用示例
if __name__ == "__main__":
    analyzer = ResearchReportAnalyzer()
    result = analyzer.generate_report(r"")

    if result:
        print("分析完成！结果已保存到:")
        print("- analysis_result.json")
        print("- analysis_report.png")
    else:
        print("分析失败，请检查PDF文件或API密钥")