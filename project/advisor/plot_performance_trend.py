import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置中文字体（如果系统支持）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def parse_metrics_files(results_dir):
    data = []
    # 匹配文件名: metrics_{model_type}_{timestamp}.json
    # timestamp 格式通常是 YYYYMMDD_HHMMSS
    pattern = re.compile(r"metrics_(.+)_(\d{8}_\d{6})\.json")

    if not os.path.exists(results_dir):
        print(f"目录不存在: {results_dir}")
        return pd.DataFrame()

    for fname in os.listdir(results_dir):
        match = pattern.match(fname)
        if match:
            model_type = match.group(1)
            timestamp_str = match.group(2)
            try:
                dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except ValueError:
                continue

            filepath = os.path.join(results_dir, fname)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    metrics = json.load(f)

                # 提取指标
                accuracy = metrics.get("accuracy", 0)
                macro_f1 = metrics.get("macro avg", {}).get("f1-score", 0)
                weighted_f1 = metrics.get("weighted avg", {}).get("f1-score", 0)

                data.append({
                    "timestamp": dt,
                    "model_type": model_type,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                    "filename": fname
                })
            except Exception as e:
                print(f"读取文件失败 {fname}: {e}")

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values("timestamp")
    return df

def plot_trends(df, output_path):
    if df.empty:
        print("没有数据可绘图")
        return

    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    # 重新设置字体，因为 sns.set_theme 可能会覆盖
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 获取唯一的模型类型
    model_types = df["model_type"].unique()

    # 创建图表：每个指标一个子图
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    metrics_to_plot = [
        ("accuracy", "Accuracy (准确率)"),
        ("macro_f1", "Macro F1-Score (宏平均F1)"),
        ("weighted_f1", "Weighted F1-Score (加权F1)")
    ]

    for i, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[i]
        sns.lineplot(data=df, x="timestamp", y=metric, hue="model_type", marker="o", ax=ax)
        ax.set_title(f"{title} 随时间变化趋势")
        ax.set_ylabel(metric)
        ax.legend(title="模型类型", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)

    plt.xlabel("时间")
    plt.xticks(rotation=45)
    plt.tight_layout()

    print(f"保存图表到: {output_path}")
    plt.savefig(output_path)
    plt.close()

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    results_dir = os.path.join(project_root, "results")

    print(f"正在扫描结果目录: {results_dir}")
    df = parse_metrics_files(results_dir)

    if df.empty:
        print("未找到任何 metrics_*.json 文件或解析失败。")
    else:
        print(f"找到 {len(df)} 条记录。")
        print(df[["timestamp", "model_type", "accuracy"]].head())

        output_file = os.path.join(results_dir, "performance_trend.png")
        plot_trends(df, output_file)
        print("完成。")

if __name__ == "__main__":
    main()

