# 特征工程与数据集拆分：从原始数据提取特征并拆分为训练/测试集
# 功能：将M1的原始数据转化为模型可输入的特征，并按比例拆分训练集和测试集
import os  # 用于目录创建、文件路径处理
import numpy as np  # 用于数值计算（填充缺失值等）
import pandas as pd  # 用于数据处理（合并、拆分）
from sklearn.model_selection import train_test_split  # 用于拆分训练集和测试集


def process_features(m1_dir="outputs", workdir="work", test_size=0.2, seed=42):
    """
    从M1生成的原始数据中提取特征，合并后拆分为训练集和测试集

    参数：
        m1_dir: M1模块输出的原始数据目录（默认"outputs"，指定原始数据来源）
        workdir: 特征文件的保存目录（默认"work"，集中存储特征数据）
        test_size: 测试集占比（默认0.2，常见拆分比例，保留80%数据用于训练）
        seed: 随机种子（默认42，保证拆分结果可复现，便于模型对比）
    """
    # 创建工作目录（若已存在则不报错，避免程序中断）
    os.makedirs(workdir, exist_ok=True)

    # 1. 读取M1生成的原始数据（用户画像、交易记录、对话数据）
    profiles = pd.read_csv(os.path.join(m1_dir, "profiles.csv"))  # 用户画像（基础特征）
    transactions = pd.read_csv(os.path.join(m1_dir, "transactions.csv"))  # 交易记录（行为特征）
    dialogs = pd.read_csv(os.path.join(m1_dir, "dialogs.csv"))  # 对话数据（交互特征）

    # 2. 生成交易特征（按用户聚合交易行为，提炼用户交易习惯）
    trans_feat = transactions.groupby("user_id", as_index=False).agg(
        action_mean=("action", "mean"),  # 用户平均交易行为（-1到1之间，越大越倾向买入）
        action_std=("action", "std"),  # 交易行为标准差（越大说明交易越不稳定）
        avg_mkt_ret=("mkt_ret", "mean")  # 用户交易时的平均市场收益率（反映交易时机偏好）
    ).fillna(0)  # 填充缺失值（若用户无交易记录，用0替代，避免后续模型报错）

    # 3. 提取对话特征（仅保留用户提问情绪，反映用户交互时的态度）
    dialog_feat = dialogs[["user_id", "q_sentiment"]].copy().fillna(0)  # 缺失情绪用0（中性）填充

    # 4. 合并所有特征（用户画像+交易特征+对话特征，形成完整特征集）
    # 第一步：合并用户画像与交易特征（按user_id关联，左连接保留所有用户）
    all_features = profiles.merge(trans_feat, on="user_id", how="left")
    # 第二步：合并对话特征（继续按user_id关联，左连接）
    all_features = all_features.merge(dialog_feat, on="user_id", how="left")
    # 最终兜底填充（确保无缺失值，避免模型训练时因NaN报错）
    all_features = all_features.fillna(0)

    # 5. 拆分训练/测试集（分层抽样，保证风险标签分布一致）
    # 分层抽样原因：风险标签可能不平衡（如稳健型用户占比高），避免测试集标签分布偏差
    train, test = train_test_split(
        all_features,
        test_size=test_size,  # 测试集占比20%
        random_state=seed,  # 固定随机种子，结果可复现
        stratify=all_features["risk_label"]  # 按风险标签分层，保持分布一致
    )

    # 6. 保存特征文件（供M3模型训练和M5分析使用）
    all_features.to_csv(os.path.join(workdir, "all_features.csv"), index=False)  # 全量特征
    train.to_csv(os.path.join(workdir, "train_features.csv"), index=False)  # 训练集特征
    test.to_csv(os.path.join(workdir, "test_features.csv"), index=False)  # 测试集特征

    # 验证关键特征列是否存在（确保后续模型可正常读取所需特征，提前发现错误）
    required_cols = ["action_mean", "q_sentiment"]  # M3模型依赖的核心特征
    missing = [col for col in required_cols if col not in all_features.columns]
    if missing:
        raise ValueError(f"M2错误：缺失关键列 {missing}")  # 抛错提示，避免下游模块失败
    print(f"M2完成：特征保存至 {workdir}，包含关键列 {required_cols}")  # 打印完成信息


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    # Default to 'data' if it exists, else 'outputs'
    data_dir = os.path.join(project_root, "data")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(project_root, "outputs")

    work_dir = os.path.join(project_root, "work")
    process_features(m1_dir=data_dir, workdir=work_dir)  # 当脚本被直接运行时，执行特征工程与拆分
