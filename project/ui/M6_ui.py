# -*- coding: utf-8 -*-
"""
 原 Gradio UI 已移除：当前文件仅保留投顾分析核心逻辑（风险评估 + 资产配置 + 文本报告）。
如需重新启用 Web 界面，可参考历史版本在此基础上重新封装 Gradio Blocks。
"""
import os
import numpy as np
import pandas as pd
from dataclasses import asdict

try:
    import joblib
except Exception:  # 允许无joblib
    joblib = None


# ----------------------------- 数据结构 -----------------------------

class UserInput:
    age: int
    education: str
    income10k: float
    asset10k: float
    debt10k: float
    children: int
    exp_years: int
    action_mean: float  # 交易倾向（-1卖出，0观望，1买入）
    q_text: str
    q_sentiment: int  # -1/0/1


# ----------------------------- 情绪分析 -----------------------------
def analyze_sentiment(text: str) -> int:
    text = (text or "").lower()
    pos_words = ["盈利", "上涨", "看好", "长期", "稳定", "增长", "机会", "低估", "利好", "乐观"]
    neg_words = ["亏损", "下跌", "担心", "风险", "波动", "崩盘", "踩雷", "不确定", "利空", "悲观"]
    pos = sum(w in text for w in pos_words)
    neg = sum(w in text for w in neg_words)
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0


def mock_answer(text: str) -> str:
    if not text.strip():
        return "您好，您可以提问如：‘现在是否适合定投？’ 我将结合您的风险偏好给出建议。"
    return (
        "感谢提问。我会结合您的画像、交易倾向与情绪，为您提供‘分散配置 + 纪律投资’的策略。"
        "若追求长期稳健，建议控制单次仓位，并使用定投平滑波动。"
    )


# ----------------------------- 风险预测（模型或规则） -----------------------------
def load_model_and_features(workdir: str = "work"):
    model, used_cols = None, None
    model_path = os.path.join(workdir, "risk_model.pkl")
    feat_path = os.path.join(workdir, "used_features_m3.csv")
    if os.path.exists(model_path) and joblib is not None:
        try:
            model = joblib.load(model_path)
        except Exception:
            model = None
    if os.path.exists(feat_path):
        try:
            used_cols = list(pd.read_csv(feat_path)["feature"].astype(str))
        except Exception:
            used_cols = None
    return model, used_cols


def predict_risk(user: UserInput, model=None, used_cols=None):
    # 若有已训练模型，则按其特征构造向量；否则走规则。
    base = {
        "age": user.age,
        "income10k": user.income10k,
        "asset10k": user.asset10k,
        "exp_years": user.exp_years,
        "action_mean": user.action_mean,
        "q_sentiment": user.q_sentiment,
        # 若模型包含下列列名，统一补零（GNN/时序特征没有单人图/序列可直接置0）
        "action_ema_10": 0.0, "action_ema_30": 0.0, "action_vol_30": 0.0,
        "action_trend_30": 0.0, "action_mkt_corr_30": 0.0, "buy_ratio_30": 0.0,
        "action_mean_gnn1": 0.0, "q_sentiment_gnn1": 0.0, "action_ema_10_gnn1": 0.0,
        "action_ema_30_gnn1": 0.0, "action_trend_30_gnn1": 0.0,
        "action_mean_gnn2": 0.0, "q_sentiment_gnn2": 0.0, "action_ema_10_gnn2": 0.0,
        "action_ema_30_gnn2": 0.0, "action_trend_30_gnn2": 0.0,
    }
    if model is not None and used_cols:
        X = np.array([[base.get(c, 0.0) for c in used_cols]], dtype=float)
        try:
            prob = float(model.predict_proba(X)[0, 1])
            pred = int(prob >= 0.5)
            return prob, pred, used_cols
        except Exception:
            pass

    # 规则兜底（与M1相近）
    score = 0.0
    score += 0.2 if user.age < 30 else 0.0
    score += 0.2 if user.income10k > 20 else 0.0
    score += 0.2 if user.asset10k > 100 else 0.0
    score += 0.2 if user.exp_years > 3 else 0.0
    score += 0.1 if user.education in ("本科", "硕士及以上") else 0.0
    score += 0.1 if user.children == 0 else 0.0
    score += 0.1 * user.action_mean  # 行为越偏买入越进取
    score += 0.05 * user.q_sentiment  # 情绪偏正面略微进取
    prob = float(np.clip(score, 0, 1))
    pred = int(prob >= 0.5)
    return prob, pred, [k for k in base.keys()]


# ----------------------------- 资产配置与投顾建议 -----------------------------
def mean_variance(mu, sigma, risk_aversion=2.0):
    inv_sigma = np.linalg.inv(sigma)
    w = (inv_sigma @ mu) / max(1e-8, risk_aversion)
    w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def build_portfolio(risk_prob: float):
    assets = ["股票", "债券", "基金", "REITs", "商品", "现金"]
    mu = np.array([0.08, 0.03, 0.06, 0.045, 0.05, 0.01])
    rng = np.random.default_rng(42)
    base = rng.standard_normal((6, 6))
    sigma = (base @ base.T) / 100.0
    risk_aversion = 4.0 - 2.0 * float(risk_prob)  # 2~4之间
    w = mean_variance(mu, sigma, risk_aversion)
    portfolio_str = ", ".join([f"{a}: {x:.1%}" for a, x in zip(assets, w)])
    return assets, w, portfolio_str


def generate_advice(user: UserInput, risk_prob: float, risk_pred: int, portfolio_str: str) -> str:
    debt_ratio = user.debt10k / user.asset10k if user.asset10k > 0 else 0.0
    is_near_retirement = user.age >= 55
    is_new = user.exp_years < 2
    risk_level_str = '进取型' if risk_pred == 1 else '稳健型'
    date_str = '2025-11-11'

    def section_title(title):
        return f"┏{'━' * 4} {title} {'━' * (56 - len(title))}┓"

    def section_end():
        return f"┗{'━' * 68}┛"

    def kv_format(key, value):
        return f"  {key:<12} : {value}"

    # --- Header ---
    advice = [
        "┌────────────────────────────────────────────────────────────────────┐",
        f"│{'智能投顾分析报告':^68}│",
        "├────────────────────────────────────────────────────────────────────┤",
        f"│ 报告日期: {date_str:<20} 用户类型: {risk_level_str:<10} │",
        "└────────────────────────────────────────────────────────────────────┘",
        "",
    ]

    # --- Section 1: 核心评估 ---
    advice.append(section_title("核心评估"))
    advice.append(kv_format("用户画像", f"{user.age}岁, {user.education}, {user.exp_years}年投资经验"))
    advice.append(kv_format("风险偏好", f"{risk_level_str} (进取概率: {risk_prob:.1%})"))

    # Risk Gauge
    n = 20
    pos = int(round(risk_prob * n))
    gauge = '■' * pos + '□' * (n - pos)
    advice.append(kv_format("风险刻度", f"{gauge}"))

    advice.append(kv_format("资产配置", portfolio_str))
    advice.append(section_end())
    advice.append("")

    # --- Section 2: 详细评估 ---
    advice.append(section_title("详细评估"))
    advice.append("  1) 财务状况:")
    advice.append(f"     - 年收入: {user.income10k:.1f}万, 总资产: {user.asset10k:.1f}万, 负债: {user.debt10k:.1f}万")
    advice.append(f"     - 债务比率: {debt_ratio:.1%}")
    advice.append("  2) 市场洞察:")
    sentiment_str = '😊 正面' if user.q_sentiment == 1 else '😟 负面' if user.q_sentiment == -1 else '😐 中性'
    action_str = '🔼 偏买入' if user.action_mean > 0.1 else '🔽 偏卖出' if user.action_mean < -0.1 else '⏸️ 观望'
    advice.append(f"     - 情绪判断: {sentiment_str}")
    advice.append(f"     - 交易倾向: {action_str}")
    advice.append(section_end())
    advice.append("")

    # --- Section 3: 行动策略建议 ---
    advice.append(section_title("行动策略建议"))
    advice.append("  ▶ 投资组合策略:")
    if risk_pred == 1:  # 进取型
        if is_new:
            advice.append("    - 新手入门：从宽基指数基金（如沪深300）起步，逐步加深。")
        else:
            advice.append("    - 积极增长：采用“核心-卫星”策略，30-40%主动权益 + 20-30%行业ETF。")
        if user.asset10k > 500:
            advice.append("    - 资产增厚：可配10-15%另类资产（REITs、黄金）提升分散度。")
    else:  # 稳健型
        if debt_ratio > 0.3:
            advice.append(f"    - 稳健为先：负债率({debt_ratio:.0%})偏高，建议50%+配置低波动产品。")
        else:
            advice.append("    - 均衡配置：构建“固收+”组合，40-50%中短债为基石，稳中求进。")

    advice.append("\n  ▶ 风险管理计划:")
    if debt_ratio > 0.5:
        advice.append("    - [高负债预警] 负债率>50%，建议每月用20%收入降杠杆。")
    else:
        advice.append(f"    - 财务健康：当前负债/资产比({debt_ratio:.0%})处于舒适区。")

    emergency_need = 6 * (user.debt10k + max(5, user.income10k / 12 * 0.5))
    if user.asset10k * 0.1 < emergency_need:
        gap = emergency_need - user.asset10k * 0.1
        advice.append(f"    - [应急资金不足] 建议储备{emergency_need:.0f}万应急金，当前缺口约{gap:.0f}万。")
    else:
        advice.append("    - 后盾坚实：应急资金充足，为投资计划保驾护航。")
    advice.append(section_end())
    advice.append("")

    # --- Section 4: 长期视角与个人成长 ---
    advice.append(section_title("长期视角与个人成长"))
    advice.append("  ▶ 人生阶段规划:")
    if is_near_retirement:
        advice.append("    - [退休准备] 临近退休，建议启动“防守模式”，逐年下调权益仓位。")
    elif user.age < 35:
        advice.append("    - [黄金投资期] 年轻是最大资本！建议坚持定投，让复利为你工作。")

    if user.children > 0:
        advice.append(f"    - [子女教育] 为{user.children}个子女规划约{user.children * 5}万元教育金。")

    advice.append("\n  ▶ 投资心法:")
    if is_new:
        advice.append("    - [新手修炼] 先用模拟盘练手，小额慢行，把控情绪。")
    else:
        advice.append("    - [纪律为王] 设定并执行止盈与止损纪律。")
    advice.append(section_end())
    advice.append("")

    # --- Footer ---
    advice.append("┌────────────────────────────────────────────────────────────────────┐")
    disclaimer = '免责声明：本报告仅为演示用途，不构成任何投资建议'
    advice.append(f"│{disclaimer:^68}│")
    advice.append("└────────────────────────────────────────────────────────────────────┘")

    return "\n".join(advice)


# ----------------------------- GUI 实现 -----------------------------
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

class AdvisorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("智能投顾演示系统 (M6)")
        self.root.geometry("900x700")

        # 确定项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(current_dir)
        self.work_dir = os.path.join(self.project_root, "work")

        # 加载模型
        self.model, self.used_cols = load_model_and_features(self.work_dir)
        model_status = "✅ 已加载训练模型" if self.model else "⚠️ 未找到模型 (使用规则兜底)"

        # 顶部状态栏
        status_frame = tk.Frame(root, bg="#f0f0f0", pady=5)
        status_frame.pack(fill=tk.X)
        tk.Label(status_frame, text=model_status, bg="#f0f0f0", fg="#333").pack(side=tk.RIGHT, padx=10)
        tk.Label(status_frame, text="💡 输入您的信息，获取个性化投资建议", bg="#f0f0f0", font=("微软雅黑", 10, "bold")).pack(side=tk.LEFT, padx=10)

        # 主容器
        main_paned = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧：输入区
        input_frame = tk.LabelFrame(main_paned, text="用户画像输入", padx=10, pady=10)
        main_paned.add(input_frame, width=300)

        self.create_inputs(input_frame)

        # 右侧：输出区
        output_frame = tk.LabelFrame(main_paned, text="投顾建议报告", padx=10, pady=10)
        main_paned.add(output_frame)

        self.advice_text = scrolledtext.ScrolledText(output_frame, font=("Consolas", 10), state=tk.DISABLED)
        self.advice_text.pack(fill=tk.BOTH, expand=True)

    def create_inputs(self, parent):
        # 辅助函数：创建带标签的输入框
        def add_row(label, var, row, tooltip=None):
            tk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)
            entry = tk.Entry(parent, textvariable=var)
            entry.grid(row=row, column=1, sticky=tk.EW, pady=5)
            if tooltip:
                tk.Label(parent, text=tooltip, fg="gray", font=("Arial", 8)).grid(row=row, column=2, sticky=tk.W, padx=5)

        self.var_age = tk.IntVar(value=30)
        add_row("年龄:", self.var_age, 0)

        tk.Label(parent, text="学历:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.var_edu = tk.StringVar(value="本科")
        edu_cb = ttk.Combobox(parent, textvariable=self.var_edu, values=["高中及以下", "大专", "本科", "硕士及以上"])
        edu_cb.grid(row=1, column=1, sticky=tk.EW, pady=5)

        self.var_income = tk.DoubleVar(value=20.0)
        add_row("年收入(万):", self.var_income, 2)

        self.var_asset = tk.DoubleVar(value=50.0)
        add_row("总资产(万):", self.var_asset, 3)

        self.var_debt = tk.DoubleVar(value=10.0)
        add_row("总负债(万):", self.var_debt, 4)

        self.var_children = tk.IntVar(value=0)
        add_row("子女数量:", self.var_children, 5)

        self.var_exp = tk.IntVar(value=3)
        add_row("投资经验(年):", self.var_exp, 6)

        # 交易倾向滑块
        tk.Label(parent, text="近期交易倾向:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.var_action = tk.DoubleVar(value=0.0)
        scale = tk.Scale(parent, variable=self.var_action, from_=-1.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, label="卖出(-1) <-> 买入(1)")
        scale.grid(row=7, column=1, columnspan=2, sticky=tk.EW)

        # 问答输入
        tk.Label(parent, text="投资疑问/想法:").grid(row=8, column=0, sticky=tk.NW, pady=5)
        self.txt_question = tk.Text(parent, height=4, width=20, font=("Arial", 9))
        self.txt_question.grid(row=8, column=1, columnspan=2, sticky=tk.EW, pady=5)
        self.txt_question.insert("1.0", "最近市场波动大，我该怎么办？")

        # 按钮
        btn_gen = tk.Button(parent, text="生成建议 (Generate)", command=self.on_generate, bg="#0078d7", fg="white", font=("Arial", 10, "bold"))
        btn_gen.grid(row=9, column=0, columnspan=3, sticky=tk.EW, pady=15)

        parent.columnconfigure(1, weight=1)

    def on_generate(self):
        try:
            # 1. 收集输入
            user = UserInput()
            user.age = self.var_age.get()
            user.education = self.var_edu.get()
            user.income10k = self.var_income.get()
            user.asset10k = self.var_asset.get()
            user.debt10k = self.var_debt.get()
            user.children = self.var_children.get()
            user.exp_years = self.var_exp.get()
            user.action_mean = self.var_action.get()
            user.q_text = self.txt_question.get("1.0", tk.END).strip()
            user.q_sentiment = analyze_sentiment(user.q_text)

            # 2. 预测风险
            risk_prob, risk_pred, _ = predict_risk(user, self.model, self.used_cols)

            # 3. 生成配置
            assets, weights, portfolio_str = build_portfolio(risk_prob)

            # 4. 生成报告
            report = generate_advice(user, risk_prob, risk_pred, portfolio_str)

            # 5. 显示
            self.advice_text.config(state=tk.NORMAL)
            self.advice_text.delete("1.0", tk.END)
            self.advice_text.insert(tk.END, report)
            self.advice_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("错误", f"生成失败: {str(e)}")


if __name__ == "__main__":
    # 确保 utils 在 path 中 (如果需要)
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    root = tk.Tk()
    app = AdvisorGUI(root)
    root.mainloop()


