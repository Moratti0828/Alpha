# 可解释性分析与投顾建议：分析特征重要性并生成个性化投资建议
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")  # 忽略警告信息

# 解决中文显示问题
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

# 新增：导入投顾文本生成与目录工具
try:
    from run_experiment import ensure_advisor_dir, generate_text_advice
except Exception:
    ensure_advisor_dir = None
    generate_text_advice = None


def m5_analysis(indir="outputs", outdir="m5_results", seed=42, n_users=100):
    """
    分析风险标签预测的特征重要性，并为用户生成个性化投顾建议

    参数：
        indir: 输入数据目录（含M1用户画像和M4资产配置，默认"outputs"）
        outdir: 分析结果和建议的保存目录（默认"m5_results"）
        seed: 随机种子（默认42）
        n_users: 生成建议的用户数量（默认100）
    """
    # 创建输出目录
    os.makedirs(outdir, exist_ok=True)

    # 1. 加载数据（M1生成的用户画像 + M4的资产配置结果）
    profiles = pd.read_csv(os.path.join(indir, "profiles.csv"))  # 用户画像
    # 加载M4的资产配置结果（路径调整：M4默认保存在"work"目录）
    portfolio_recs = pd.read_csv(os.path.join(indir.replace("outputs", "work"), "portfolio_recs.csv"))
    # 合并用户画像和资产配置（按user_id关联）
    profiles = profiles.merge(portfolio_recs[["user_id", "portfolio"]], on="user_id", how="left")
    print(f"加载数据完成，共{len(profiles)}条用户记录，将为{min(n_users, len(profiles))}位用户生成建议")
    print("特征列表（含新增特征）：", profiles.columns.tolist())

    # 2. 特征工程（处理分类特征，用于模型训练和特征重要性分析）
    # 输入特征（包含学历、负债、子女数量等新增特征）
    features = [
        "age", "education", "income10k", "asset10k",
        "debt10k", "children", "exp_years"
    ]
    target = "risk_label"  # 目标变量：风险标签

    X = profiles[features]  # 特征数据
    y = profiles[target]  # 标签数据

    # 划分训练集和测试集（分层抽样，保持风险标签分布）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    # 预处理：对分类特征（学历）进行独热编码，数值特征直接保留
    categorical_features = ["education"]  # 分类特征：学历
    numerical_features = [f for f in features if f not in categorical_features]  # 数值特征

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),  # 数值特征：直接传递
            ("cat", OneHotEncoder(sparse_output=False), categorical_features)  # 分类特征：独热编码
        ])

    # 3. 训练模型（随机森林，用于特征重要性分析）
    model = Pipeline([
        ("preprocessor", preprocessor),  # 预处理步骤（编码+数值保留）
        ("classifier", RandomForestClassifier(
            n_estimators=100,  # 100棵树
            max_depth=8,  # 限制树深度，避免过拟合
            random_state=seed,
            n_jobs=-1  # 并行计算
        ))
    ])

    model.fit(X_train, y_train)  # 训练模型
    y_pred = model.predict(X_test)  # 测试集预测

    # 模型评估（输出分类报告：精确率、召回率、F1值）
    print("\n模型评估结果：")
    print(classification_report(y_test, y_pred))

    # 4. 特征重要性分析（识别影响风险标签的关键特征）
    # 获取独热编码器，提取编码后的分类特征名称
    ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))  # 如"education_本科"
    all_feature_names = numerical_features + cat_feature_names  # 所有特征名称（数值+编码后的分类）

    # 提取随机森林的特征重要性（每个特征对预测的贡献度）
    importances = model.named_steps["classifier"].feature_importances_
    importance_df = pd.DataFrame({
        "特征": all_feature_names,
        "重要性": importances
    }).sort_values(by="重要性", ascending=False)  # 按重要性降序排列

    # 保存特征重要性结果
    importance_df.to_csv(os.path.join(outdir, "feature_importance.csv"), index=False)
    print("\n特征重要性已保存，前5名特征：")
    print(importance_df.head())

    # 可视化特征重要性（条形图）
    plt.figure(figsize=(12, 8))
    sns.barplot(x="重要性", y="特征", data=importance_df)
    plt.title("特征重要性排序（含学历、负债、子女数量）")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "feature_importance.png"))
    plt.close()

    # 5. 生成个性化投顾建议（基于用户特征和资产配置）
    def generate_advice(user):
        """
        为单个用户生成个性化投顾建议（分模块：配置、风险、长期规划、行为）

        参数：
            user: 单条用户数据（含特征、风险标签、资产配置）

        返回：
            格式化的建议文本
        """
        # 提取用户关键特征
        uid = user["user_id"]
        age = user["age"]
        edu = user["education"]
        income = user["income10k"]  # 年收入（万元）
        asset = user["asset10k"]  # 总资产（万元）
        debt = user["debt10k"]  # 负债（万元）
        children = user["children"]
        exp = user["exp_years"]  # 投资经验（年）
        risk = user["risk_label"]  # 风险标签（0=稳健，1=进取）
        portfolio = user.get("portfolio", "未生成配置")  # 资产配置方案

        # 计算衍生指标（用于细化建议）
        debt_ratio = debt / asset if asset > 0 else 0  # 负债/资产比（衡量负债压力）
        # 估算储蓄率（收入扣除负债后的可储蓄比例，限制在10%-30%）
        savings_rate = min(0.3, max(0.1, (income * 12 - debt * 10) / (income * 12)))
        is_near_retirement = age >= 55  # 是否临近退休（55岁以上）
        is_new_investor = exp < 2  # 是否投资新手（经验<2年）

        # 建议框架（分模块组织）
        advice = [
            f"用户ID：{uid} | 风险等级：{'进取型' if risk == 1 else '稳健型'}",
            f"当前推荐配置：{portfolio}",
            "--- 一、基础配置建议 ---"
        ]

        # 1. 资产配置细化建议（基于风险等级、投资经验、资产规模）
        if risk == 1:  # 进取型用户
            if is_new_investor:
                advice.append("• 作为投资新手，建议以指数基金为主（如沪深300、中证500），降低个股风险")
            else:
                advice.append("• 可配置30%-40%主动管理型基金+20%-30%行业ETF（如科技、新能源）")
            # 高资产用户增加另类资产配置
            if asset > 500:
                advice.append("• 建议配置10%-15%另类资产（如REITs、黄金ETF）分散风险")
        else:  # 稳健型用户
            # 高负债用户优先配置低波动资产
            if debt_ratio > 0.3:
                advice.append(f"• 负债比例{debt_ratio:.0%}，建议优先配置50%以上低波动资产（如国债、货币基金）")
            else:
                advice.append("• 可配置40%-50%中短债基金+20%-30%股息率较高的蓝筹股")

        # 2. 风险管理建议（负债和应急资金）
        advice.append("\n--- 二、风险管理建议 ---")
        if debt_ratio > 0.5:
            advice.append(f"• ⚠️ 高负债预警：负债/资产比{debt_ratio:.0%}，建议每月偿还至少收入的20%用于降低负债")
        else:
            advice.append(f"• 负债健康：当前负债比{debt_ratio:.0%}，可维持现有还款计划")

        # 应急资金建议（基于负债和基础开支估算）
        emergency_need = 6 * (debt + 5)  # 6个月的负债+基础开支（5万/月）
        if asset * 0.1 < emergency_need:
            advice.append(f"• 建议储备{emergency_need:.0f}万元应急资金（当前缺口约{emergency_need - asset * 0.1:.0f}万）")
        else:
            advice.append("• 应急资金充足，可继续按计划投资")

        # 3. 长期规划建议（退休、子女教育等）
        advice.append("\n--- 三、长期规划建议 ---")
        if is_near_retirement:
            advice.append("• 临近退休：未来5年建议每年降低5%-10%权益资产比例，增加年金类产品配置")
        elif age < 35:
            advice.append("• 年轻用户：可采用定投策略（每月投入收入的15%-20%），长期复利效应更显著")

        # 子女教育/养老规划
        if children > 0:
            advice.append(f"• 子女规划：建议配置{children * 5}万元教育金（可选择529计划或教育金保险）")
        if age > 40 and "硕士及以上" not in edu:
            advice.append("• 可关注商业养老保险，补充基础养老金不足")

        # 4. 投资行为建议（基于经验和学历）
        advice.append("\n--- 四、投资行为建议 ---")
        if is_new_investor:
            advice.append("• 建议先通过模拟交易熟悉市场，避免频繁操作（月交易次数控制在5次以内）")
        else:
            advice.append("• 可设置止盈止损点（如盈利20%止盈，亏损10%止损），避免情绪化决策")

        # 结合学历的建议
        if edu in ["本科", "硕士及以上"]:
            advice.append("• 可利用专业知识深入研究1-2个行业，建立能力圈投资")
        else:
            advice.append("• 推荐关注理财科普内容（如基金定投指南），逐步提升投资认知")

        return "\n".join(advice)  # 合并建议为文本

    # 生成指定数量用户的建议（随机抽样，避免分布偏差）
    sample_users = profiles.sample(min(n_users, len(profiles)), random_state=seed)
    advice_list = [generate_advice(user) for _, user in sample_users.iterrows()]

    # 保存建议到文本文件
    with open(os.path.join(outdir, "user_advice.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(advice_list))

    print(f"\n投顾建议生成完成，保存至 {outdir}/user_advice.txt")
    print("示例建议：")
    print(advice_list[0])  # 打印第一条建议作为示例

    # 新增：同步一份到 advisor_design/，并采用模板化风格（数据来源于 M3/M4 输出）
    if ensure_advisor_dir is not None and generate_text_advice is not None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        advisor_dir = ensure_advisor_dir(base_dir)
        profiles_csv = os.path.join(indir, "profiles.csv")
        try:
            out_txt = generate_text_advice(profiles_csv, advisor_dir, max_users=n_users)
            print(f"投顾建议（模板风格）已同步到：{out_txt}")
        except Exception as e:
            print(f"WARN: 写入 advisor_design 失败：{e}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Default to 'data' if it exists, else 'outputs'
    data_dir = os.path.join(project_root, "data")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(project_root, "outputs")

    work_dir = os.path.join(project_root, "work")
    m5_results_dir = os.path.join(project_root, "m5_results")

    m5_analysis(indir=data_dir, workdir=work_dir, outdir=m5_results_dir, n_users=100)  # 生成100位用户的建议
