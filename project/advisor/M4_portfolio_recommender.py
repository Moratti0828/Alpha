# 资产配置推荐：基于用户风险预测结果生成个性化资产配置方案
import os
import numpy as np
import pandas as pd


def mean_variance(mu, sigma, risk_aversion=2.0):
    """
    均值-方差优化算法：计算各资产的最优配置权重（最大化风险调整后收益）

    参数：
        mu: 各资产的期望收益率（1D数组）
        sigma: 资产收益率的协方差矩阵（2D数组，正定矩阵）
        risk_aversion: 风险厌恶系数（越大越保守，默认2.0）

    返回：
        weights: 各资产的配置权重（求和为1，非负，无卖空）
    """
    inv_sigma = np.linalg.inv(sigma)  # 协方差矩阵的逆矩阵
    # 计算初始权重（基于均值-方差理论：权重 ∝ 协方差逆矩阵 × 期望收益 / 风险厌恶系数）
    weights = (inv_sigma @ mu) / risk_aversion
    weights = weights.clip(0, None)  # 禁止卖空（权重≥0）
    # 归一化权重（确保总和为1）
    return weights / weights.sum() if weights.sum() != 0 else np.ones_like(weights) / len(weights)


def generate_portfolio(workdir="work", outdir="work"):
    """
    为每个用户生成个性化资产配置方案，基于M3的风险预测结果

    参数：
        workdir: 输入数据目录（含M2特征和M3预测结果，默认"work"）
        outdir: 配置结果的保存目录（默认"work"）
    """
    # 创建输出目录
    os.makedirs(outdir, exist_ok=True)

    # 1. 读取数据（M2的全量特征+M3的风险预测结果）
    all_features = pd.read_csv(os.path.join(workdir, "all_features.csv"))
    predictions = pd.read_csv(os.path.join(workdir, "predictions.csv"))

    # 2. 合并数据（仅保留有预测结果的用户）
    user_data = all_features.merge(predictions, on="user_id", how="inner")

    # 3. 资产参数设置（定义可配置的资产及收益特性）
    assets = ["股票", "债券", "基金", "REITs", "大宗商品", "现金"]  # 资产类型（中文）
    mu = np.array([0.08, 0.03, 0.06, 0.045, 0.05, 0.01])  # 各资产期望年收益率（股票8%，债券3%等）
    # 生成正定协方差矩阵（确保均值-方差优化可解）
    np.random.seed(42)
    base = np.random.randn(6, 6)  # 随机矩阵
    sigma = (base @ base.T) / 100  # 协方差矩阵（缩放至合理范围）

    # 4. 为每个用户生成配置方案
    portfolios = []
    for _, user in user_data.iterrows():
        # 风险厌恶系数：高风险概率（risk_prob）→ 低厌恶（更进取），范围2.0-4.0
        risk_aversion = 4.0 - 2.0 * user["risk_prob"]
        # 计算资产权重（调用均值-方差优化函数）
        weights = mean_variance(mu, sigma, risk_aversion)
        # 格式化为易读的配置字符串（如"股票: 30.0%, 债券: 20.0%"）
        portfolio_str = ", ".join([f"{asset}: {w:.1%}" for asset, w in zip(assets, weights)])
        # 存储用户ID、配置字符串及各资产权重
        portfolios.append([user["user_id"], portfolio_str] + weights.tolist())

    # 5. 保存配置结果（供M5生成投顾建议使用）
    portfolio_df = pd.DataFrame(
        portfolios,
        columns=["user_id", "portfolio"] + assets  # 列名：用户ID、配置字符串、各资产权重
    )
    portfolio_df.to_csv(os.path.join(outdir, "portfolio_recs.csv"), index=False)
    print(f"M4完成：资产配置保存至 {outdir}/portfolio_recs.csv")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    work_dir = os.path.join(project_root, "work")
    generate_portfolio(workdir=work_dir, outdir=work_dir)  # 执行资产配置生成
