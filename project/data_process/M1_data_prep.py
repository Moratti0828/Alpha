# M1_data_prep.py
# 数据模拟模块：生成用户画像、市场数据、交易日志、对话数据等。
# 目的：为下游模块提供结构化的合成数据，包括用户特征、行为与市场环境

import os  # 用于目录创建与路径处理
import json
import numpy as np  # 用于数值运算（随机数生成、数组运算）
import pandas as pd  # 用于结构化数据（DataFrame）与保存
import matplotlib.pyplot as plt  # 用于数据可视化
import matplotlib.font_manager as fm  # 引入字体管理模块
import seaborn as sns  # 基于 matplotlib 的高级可视化

# ==========================================
# 强制加载项目根目录下的 SimHei.ttf 字体
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
font_path = os.path.join(project_root, "SimHei.ttf")

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
else:
    print("警告: 当前目录下未找到 SimHei.ttf，中文可能无法显示！")
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "Arial Unicode MS"]

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
# ==========================================


def simulate(n_users=2000, n_days=240, seed=42, outdir="outputs"):
    """
    生成用户与市场相关的合成数据，保存为 CSV 文件，并输出简单可视化结果。

    参数：
        n_users: 模拟的用户数量
        n_days: 模拟的交易日天数
        seed: 随机种子，保证结果可复现
        outdir: 输出目录，保存 CSV 与图像
    """
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)

    # 1. 生成用户画像数据
    ages = rng.integers(20, 65, n_users)

    education_levels = ["高中及以下", "大专", "本科", "硕士及以上"]
    education_probs = [0.2, 0.3, 0.4, 0.1]
    education = rng.choice(education_levels, size=n_users, p=education_probs)

    edu_income_baseline = {"高中及以下": 10, "大专": 14, "本科": 18, "硕士及以上": 28}
    income = np.array([rng.normal(edu_income_baseline[edu], 5, 1)[0] for edu in education]).clip(3, 80)

    edu_asset_multiplier = {"高中及以下": 5, "大专": 6, "本科": 7, "硕士及以上": 9}
    asset_multipliers = np.array([edu_asset_multiplier[edu] for edu in education])
    asset10k = (income * asset_multipliers + rng.normal(0, 15, n_users)).clip(5, 800)

    exp_years = rng.integers(0, 10, n_users)

    def get_children(age):
        if age < 30:
            return rng.choice([0, 1], p=[0.8, 0.2])
        elif 30 <= age < 50:
            return rng.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
        else:
            return rng.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])

    children = np.array([get_children(age) for age in ages])

    debt_ratio = 0.3 + 0.1 * (children > 0) - 0.05 * np.array([edu_asset_multiplier[edu] / 10 for edu in education])
    debt10k = (income * asset_multipliers * debt_ratio + rng.normal(0, 5, n_users)).clip(0, asset10k * 0.7)

    # 风险标签：0=稳健，1=进取
    risk_score = (
        0.2 * (ages < 30)
        + 0.2 * (income > 20)
        + 0.2 * (asset10k > 100)
        + 0.2 * (exp_years > 3)
        + 0.1 * (np.isin(education, ["本科", "硕士及以上"]))
        + 0.1 * (children == 0)
    )
    risk_label = (risk_score > 0.5).astype(int)

    # ===== 新增：用户最近关注的股票/主题 watchlist（作为输入，不是推荐）=====
    watchlists = []
    for i in range(n_users):
        is_aggressive = int(risk_label[i]) == 1
        age = int(ages[i])
        edu = str(education[i])
        kid = int(children[i])

        if is_aggressive:
            pools = [
                ["宁德时代", "比亚迪", "新能源ETF"],
                ["半导体ETF", "AI主题ETF", "科创50ETF"],
                ["创业板ETF", "中证500ETF", "成长风格基金"],
                ["中概互联ETF", "纳指ETF", "海外科技ETF"],
            ]
        else:
            pools = [
                ["沪深300ETF", "中证A500ETF", "宽基指数基金"],
                ["红利ETF", "高股息ETF", "银行板块"],
                ["中短债基金", "货币基金", "国债ETF"],
                ["黄金ETF", "大宗商品ETF", "抗通胀主题"],
            ]

        if (age >= 55) or (kid >= 2):
            pools = pools + [["养老目标基金", "年金/养老金", "稳健配置"]]

        if edu in ["本科", "硕士及以上"]:
            pools = pools + [["全球资产配置", "指数增强", "行业轮动"]]
        else:
            pools = pools + [["理财基础", "基金定投", "风险教育"]]

        wl = pools[int(rng.integers(0, len(pools)))]
        watchlists.append(json.dumps(wl, ensure_ascii=False))

    profiles = pd.DataFrame(
        {
            "user_id": np.arange(n_users),
            "age": ages,
            "education": education,
            "income10k": income,
            "asset10k": asset10k,
            "debt10k": debt10k,
            "children": children,
            "exp_years": exp_years,
            "risk_label": risk_label,
            "watchlist": watchlists,  # 新增列
        }
    )
    profiles.to_csv(os.path.join(outdir, "profiles.csv"), index=False)

    # 2. 市场数据
    mkt = rng.normal(0.0005, 0.01, n_days)
    vol = rng.normal(0.15, 0.05, n_days).clip(0.05, 0.6)
    market = pd.DataFrame({"day": range(n_days), "mkt_ret": mkt, "mkt_vol": vol})
    market.to_csv(os.path.join(outdir, "market.csv"), index=False)

    # 3. 用户交易
    transactions = []
    for uid in range(n_users):
        pref = 0.7 if risk_label[uid] == 1 else 0.3
        for d in range(n_days):
            ret = mkt[d] + rng.normal(0, 0.01)
            p_buy = np.clip(pref * (0.5 + 3 * mkt[d]), 0.05, 0.9)
            p_sell = np.clip((1 - pref) * (0.5 - 3 * mkt[d]), 0.05, 0.9)
            act = rng.choice(
                [-1, 0, 1],
                p=[p_sell * (1 - p_buy), 1 - (p_buy + p_sell * (1 - p_buy)), p_buy],
            )
            transactions.append([uid, d, ret, vol[d], act])

    transactions = pd.DataFrame(transactions, columns=["user_id", "day", "mkt_ret", "mkt_vol", "action"])
    transactions.to_csv(os.path.join(outdir, "transactions.csv"), index=False)

    # 4. 对话数据
    intents = [
        "新能源 基金 长期 定投",
        "AI 半导体 成长 风险承受高",
        "价值 投资 蓝筹 稳健 分红",
        "债券 固收 低波动",
        "黄金 商品 抗通胀",
        "医药 消费 产业轮动",
        "海外 ETF 汇率 风险",
        "短线 交易 高频",
        "长期 持有 价值",
    ]
    dialogs = pd.DataFrame(
        {
            "user_id": np.arange(n_users),
            "q_text": rng.choice(intents, n_users),
            "a_text": rng.choice(intents, n_users),
            "q_sentiment": rng.choice([-1, 0, 1], n_users, p=[0.2, 0.5, 0.3]),
            "a_sentiment": rng.choice([-1, 0, 1], n_users, p=[0.2, 0.5, 0.3]),
        }
    )
    dialogs.to_csv(os.path.join(outdir, "dialogs.csv"), index=False)

    # 5. 可视化
    plt.figure(figsize=(18, 15))

    plt.subplot(3, 2, 1)
    sns.histplot(profiles["age"], kde=True)
    plt.title("用户年龄分布")

    plt.subplot(3, 2, 2)
    sns.countplot(x="education", data=profiles)
    plt.title("学历分布")

    plt.subplot(3, 2, 3)
    sns.histplot(profiles["income10k"], kde=True)
    plt.title("年收入分布（万元）")

    plt.subplot(3, 2, 4)
    sns.boxplot(x="education", y="income10k", data=profiles)
    plt.title("不同学历的收入")

    plt.subplot(3, 2, 5)
    sns.scatterplot(x="income10k", y="asset10k", hue="education", data=profiles)
    plt.title("收入与资产散点图（按学历着色）")

    plt.subplot(3, 2, 6)
    sns.boxplot(x="children", y="debt10k", data=profiles)
    plt.title("不同子女数量的负债")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "user_data_vis.png"))
    plt.close()

    print(f"M1 完成：数据已保存至 {outdir}。profiles.csv 已包含 watchlist 字段。")


if __name__ == "__main__":
    simulate()