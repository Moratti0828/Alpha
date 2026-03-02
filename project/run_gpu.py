"""
run_gpu.py (最终全能版)

用途：
- 基于 PyTorch + LSTM 在 GPU 上训练金融预测模型。
- 自动处理：数据生成 -> 模型训练 -> 结果评估 -> 权重保存(.pth) -> 报告生成。
- 包含字体自动加载与类别权重优化。

运行方式：
python run_gpu.py
"""

import os
import sys
sys.path.extend([
    os.path.join(os.path.dirname(__file__), 'data_process'),
    os.path.join(os.path.dirname(__file__), 'models'),
    os.path.join(os.path.dirname(__file__), 'advisor'),
    os.path.join(os.path.dirname(__file__), 'ui'),
    os.path.join(os.path.dirname(__file__), 'utils'),
])

import json
import time
import base64
import io
import html
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 字体管理
import seaborn as sns

# ================= 1. 环境与字体设置 =================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 核心修复：强制加载本地字体 ---
base_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(base_dir, 'SimHei.ttf')

if os.path.exists(font_path):
    # print(f"✅ [Font] 加载本地字体: {font_path}")
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    print("⚠️ [Font] 未找到 SimHei.ttf，回退到系统字体")
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
# --------------------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler

# 复用现有模块
from M1_data_prep import simulate
from temporal_model import build_sequences
from advisor_llm import generate_text_advice

# ================= 2. 定义 LSTM 模型 =================
# 注意：这个类定义必须与 api_server.py 中的保持完全一致
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob=0.2):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, 
            batch_first=True, dropout=dropout_prob if layer_dim > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.out(out)
        return out

# ================= 3. 训练与辅助函数 =================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15):
    print(f" -> 开始训练 (设备: {device})，共 {num_epochs} 轮...")
    model = model.to(device)
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 简单验证
        if (epoch + 1) % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
            print(f"    Epoch [{epoch+1}/{num_epochs}] | Val Acc: {acc:.2f}%")

    print(f" -> 训练结束，总耗时: {time.time() - start_time:.2f}秒")
    return model

def get_predictions(model, loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = softmax(outputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def print_status(step_name):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"[{ts}] 🚀 [GPU 任务] {step_name}")
    print(f"{'='*60}")

def ensure_dirs(base_dir):
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    advisor_dir = os.path.join(base_dir, "advisor_design")
    work_dir = os.path.join(base_dir, "work") 
    for d in [data_dir, results_dir, advisor_dir, work_dir]:
        os.makedirs(d, exist_ok=True)
    return data_dir, results_dir, advisor_dir, work_dir

# 绘图函数
def save_confusion_matrix(y_true, y_pred, classes, out_png, title="混淆矩阵"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("预测")
    plt.ylabel("真实")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def save_roc_pr_curves(y_true, y_prob, out_png_prefix):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC 曲线")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(rec, prec, label=f"AUC={pr_auc:.3f}")
    plt.title("P-R 曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_prefix + "_roc_pr.png")
    plt.close()

# ================= 4. 素材生成函数 (集成自 make_report.py) =================
def generate_advisor_assets(advisor_dir, profiles_path, predictions_path):
    print(" -> 生成可视化素材与 HTML 报告...")
    
    # 1. 静态图表
    colors = ["#2E86AB", "#F6F5AE", "#F26419", "#1B998B", "#C5D86D", "#6C5B7B"]
    plt.figure(figsize=(8, 2))
    for i, c in enumerate(colors):
        plt.bar(i, 1, color=c)
        plt.text(i, 1.02, c, ha="center", va="bottom", fontsize=9)
    plt.axis('off')
    plt.title("配色方案")
    plt.savefig(os.path.join(advisor_dir, "palette.png"))
    plt.close()

    # 风险仪表盘示例
    from matplotlib.patches import Wedge
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_aspect('equal')
    ax.add_patch(Wedge((0, 0), 1, 180, 0, facecolor="#E0E0E0"))
    ax.add_patch(Wedge((0, 0), 1, 180, 120, facecolor="#2ecc71"))
    ax.add_patch(Wedge((0, 0), 1, 120, 60, facecolor="#f1c40f"))
    ax.add_patch(Wedge((0, 0), 1, 60, 0, facecolor="#e74c3c"))
    # 指针 (示例 0.7)
    theta = 180 * (1 - 0.7)
    x = 0.8 * np.cos(np.deg2rad(theta))
    y = 0.8 * np.sin(np.deg2rad(theta))
    ax.plot([0, x], [0, y], color="black", linewidth=2)
    ax.scatter([0], [0], color="black", s=20)
    ax.text(0, -0.2, "市场风险评分：0.70", ha="center", va="center", fontsize=12)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.2, 1.05); ax.axis('off')
    ax.set_title("风险仪表盘")
    plt.savefig(os.path.join(advisor_dir, "risk_gauge.png"))
    plt.close()

    # Banner
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_facecolor("#2E86AB")
    ax.text(0.05, 0.5, "智能投顾实验报告 (LSTM版)", color="white", fontsize=24, va="center", ha="left")
    ax.text(0.05, 0.2, "Powered by NVIDIA RTX 3090", color="white", fontsize=12, va="center", ha="left")
    ax.axis('off')
    plt.savefig(os.path.join(advisor_dir, "banner.png"))
    plt.close()

    # 2. 生成 HTML
    try:
        # if not os.getenv("SILICONFLOW_API_KEY"):
        #     raise ValueError("Missing SILICONFLOW_API_KEY")

        print(" -> 调用 LLM 生成投顾建议...")
        generate_text_advice(
            profiles_csv=profiles_path,
            advisor_dir=advisor_dir,
            predictions_csv=predictions_path,
            max_users=20  # 限制用户数以节省时间和成本
        )
        print(f"✅ HTML 报告已生成: {os.path.join(advisor_dir, 'advice_all.html')}")

    except Exception as e:
        print(f"⚠️ LLM 生成不可用 ({e})，回退到本地规则模板...")

        try:
            df = pd.read_csv(profiles_path)
            pred_df = pd.read_csv(predictions_path)
            if 'risk_prob' in pred_df.columns:
                risk_map = dict(zip(pred_df['user_id'], pred_df['risk_prob']))
                df['risk_prob'] = df['user_id'].map(risk_map).fillna(0.5)
            else:
                df['risk_prob'] = 0.5

            html_parts = [
                "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'><title>智能投顾建议</title>",
                "<style>body{font-family:'Microsoft YaHei','SimHei',sans-serif;background:#f0f2f5;padding:20px;} .card{background:#fff;padding:15px;margin-bottom:15px;border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,0.05);} .badge{padding:3px 8px;border-radius:4px;color:#fff;font-size:12px;} .bg-red{background:#e74c3c;} .bg-green{background:#2ecc71;} .bg-blue{background:#3498db;}</style></head><body>",
                "<h1>🚀 智能投顾建议书 (GPU加速版 - 本地备用)</h1>",
                f"<p>数据来源：LSTM 二分类预测模型 | 覆盖用户数：{min(200, len(df))}</p>"
            ]

            for _, row in df.head(200).iterrows():
                uid = int(row.get('user_id', 0))
                prob = float(row.get('risk_prob', 0.5))

                if prob > 0.6:
                    risk_str, color, advice = "进取型", "bg-red", "建议关注科技、新能源等高弹性板块，适当增加仓位。"
                elif prob < 0.4:
                    risk_str, color, advice = "保守型", "bg-green", "市场风险较高，建议止盈离场，转投国债或货币基金。"
                else:
                    risk_str, color, advice = "稳健型", "bg-blue", "市场方向不明，建议保持现有仓位，定投沪深300指数。"

                gauge_len = int(prob * 20)
                gauge = "█" * gauge_len + "░" * (20 - gauge_len)

                html_parts.append(f"""
                <div class='card'>
                    <h3>用户 {uid} <span class='badge {color}'>{risk_str}</span></h3>
                    <p><strong>资产状况：</strong>年收入 {row.get('income10k',0):.1f}万，总资产 {row.get('asset10k',0):.1f}万</p>
                    <p><strong>AI 预测 (买入概率)：</strong> <span style='font-family:monospace;color:#2E86AB'>{gauge} {prob:.1%}</span></p>
                    <div style='background:#f9f9f9;padding:10px;border-left:4px solid #2E86AB'><strong>🤖 投资建议：</strong><br>{advice}</div>
                </div>
                """)

            html_parts.append("</body></html>")
            with open(os.path.join(advisor_dir, "advice_all.html"), "w", encoding="utf-8") as f:
                f.write("\n".join(html_parts))
            print(f"✅ (备用) HTML 报告已生成: {os.path.join(advisor_dir, 'advice_all.html')}")

        except Exception as e2:
            print(f"❌ HTML 生成彻底失败: {e2}")

# ================= 5. 主程序 =================
def main():
    print(f"\n硬件检测: Python {sys.version.split()[0]}, PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ 正在使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 警告: 未检测到 GPU，将在 CPU 上运行")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir, results_dir, advisor_dir, work_dir = ensure_dirs(base_dir)

    # --- 步骤 1: 生成数据 (CPU) ---
    n_users = 2000
    n_days = 240
    max_samples = 200000 
    
    print_status(f"M1: 生成合成数据 (N_USERS={n_users}, N_DAYS={n_days})")
    start_time = time.time()
    simulate(n_users=n_users, n_days=n_days, seed=42, outdir=data_dir)
    print(f" -> 数据生成耗时: {time.time() - start_time:.2f}秒")

    tx_path = os.path.join(data_dir, "transactions.csv")
    mkt_path = os.path.join(data_dir, "market.csv")

    # --- 步骤 2: 训练多分类 LSTM ---
    print_status("M2: 训练多分类 LSTM (卖出/观望/买入)")
    seq_len = 60
    X, y, feat_cols = build_sequences(tx_path, mkt_path, seq_len=seq_len, max_samples=max_samples, add_rolling=True, binary=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 归一化
    scaler = StandardScaler()
    N, L, F = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N, L, F)
    X_test = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape[0], L, F)
    
    # 训练
    batch_size = 512
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=batch_size)
    
    model_mc = LSTMClassifier(F, 128, 2, 3)
    # 优化：类别权重 (解决样本不平衡)
    weights_mc = torch.tensor([2.0, 1.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_mc)
    optimizer = optim.Adam(model_mc.parameters(), lr=0.001)
    
    train_model(model_mc, train_loader, val_loader, criterion, optimizer, num_epochs=20)
    
    # 评估与保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    y_true, y_pred, _ = get_predictions(model_mc, val_loader)
    save_confusion_matrix(y_true, y_pred, classes=['卖出', '观望', '买入'], out_png=os.path.join(results_dir, f"cm_lstm_mc_{ts}.png"))

    # --- 步骤 3: 训练二分类 LSTM ---
    print_status("M3: 训练二分类 LSTM (买入 vs 其他)")
    Xb, yb, feat_cols_b = build_sequences(tx_path, mkt_path, seq_len=seq_len, max_samples=max_samples, add_rolling=True, binary=True)
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.2, random_state=42, stratify=yb)
    
    # 归一化
    scaler_b = StandardScaler()
    N, L, F = Xb_train.shape
    Xb_train = scaler_b.fit_transform(Xb_train.reshape(-1, F)).reshape(N, L, F)
    Xb_test = scaler_b.transform(Xb_test.reshape(-1, F)).reshape(Xb_test.shape[0], L, F)
    
    train_loader_b = DataLoader(TensorDataset(torch.FloatTensor(Xb_train), torch.LongTensor(yb_train)), batch_size=batch_size, shuffle=True)
    val_loader_b = DataLoader(TensorDataset(torch.FloatTensor(Xb_test), torch.LongTensor(yb_test)), batch_size=batch_size)
    
    model_bin = LSTMClassifier(F, 128, 2, 2)
    # 优化：二分类权重
    weights_bin = torch.tensor([1.0, 4.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_bin)
    optimizer = optim.Adam(model_bin.parameters(), lr=0.001)
    
    train_model(model_bin, train_loader_b, val_loader_b, criterion, optimizer, num_epochs=20)
    
    # 评估
    y_true_b, y_pred_b, y_probs_b = get_predictions(model_bin, val_loader_b)
    save_confusion_matrix(y_true_b, y_pred_b, classes=['非买入', '买入'], out_png=os.path.join(results_dir, f"cm_lstm_bin_{ts}.png"))
    save_roc_pr_curves(y_true_b, y_probs_b[:, 1], os.path.join(results_dir, f"binary_lstm_{ts}"))

    # --- 步骤 4: 保存权重与生成素材 ---
    print_status("M4: 保存模型与生成建议")
    
    # 1. 保存 PyTorch 权重文件 (供 API 使用)
    model_path = os.path.join(work_dir, "best_model_binary.pth")
    torch.save(model_bin.state_dict(), model_path)
    print(f"✅ 模型权重已保存: {model_path}")

    # 2. 保存预测概率供生成报告使用
    pred_df = pd.DataFrame({
        'user_id': range(len(y_probs_b)), # 模拟 ID
        'risk_prob': y_probs_b[:, 1]
    })
    pred_path = os.path.join(work_dir, 'predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    
    # 3. 调用内置的素材生成函数
    profiles_path = os.path.join(data_dir, "profiles.csv")
    generate_advisor_assets(advisor_dir, profiles_path, pred_path)
    
    print(f"\n✅ 全部任务完成！")
    print(f" - [模型] 权重文件: {model_path}")
    print(f" - [演示] HTML报告: {os.path.join(advisor_dir, 'advice_all.html')}")

if __name__ == "__main__":
    main()