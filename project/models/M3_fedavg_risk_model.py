# 联邦学习风险预测模型（增强版）：加入GNN特征与时序特征后进行联邦概率集成
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from federated_ensemble import FederatedEnsemble
import importlib
import subprocess


def _safe_read_csv(path):
    """安全读取CSV，不存在则返回None。"""
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception as e:
        print(f"读取失败 {path}: {e}")
    return None


def _build_user_graph(all_features: pd.DataFrame, k: int = 10) -> np.ndarray:
    """
    基于用户画像构建kNN图，返回对称归一化邻接矩阵 A_hat = D^{-1/2}(A+I)D^{-1/2}
    仅使用数值画像特征以避免文本处理依赖。
    """
    # 数值画像特征（与风险偏好相关且在M1中稳定生成）
    num_cols = [c for c in ["age", "income10k", "asset10k", "exp_years", "children", "debt10k"] if c in all_features.columns]
    if not num_cols:
        raise ValueError("构图失败：缺少数值画像特征列")

    X = all_features[num_cols].astype(float).values
    X = StandardScaler().fit_transform(X)

    # 构建k近邻无向图（包含自环）
    A = kneighbors_graph(X, n_neighbors=min(k, max(1, len(X) - 1)), mode="connectivity", include_self=True)
    A = A.maximum(A.T)  # 对称化

    # A_hat 归一化
    deg = np.array(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    inv_sqrt_deg = 1.0 / np.sqrt(deg)
    D_inv_sqrt = inv_sqrt_deg[:, None] * inv_sqrt_deg[None, :]

    # 稀疏矩阵点乘（转成稠密会更简单，n<=几千可承受）
    A_dense = A.toarray()
    A_hat = A_dense * D_inv_sqrt
    return A_hat


def _graph_propagate_features(all_features: pd.DataFrame, A_hat: np.ndarray, cols: list, steps: int = 2) -> pd.DataFrame:
    """
    进行GCN式特征传播：H^{(l+1)} = A_hat H^{(l)}，返回带有_gnn{l}后缀的新特征。
    """
    out = all_features.copy()
    H0 = all_features[cols].astype(float).values
    H = H0
    for l in range(1, steps + 1):
        H = A_hat.dot(H)
        for i, c in enumerate(cols):
            out[f"{c}_gnn{l}"] = H[:, i]
    return out


def _compute_temporal_features(transactions: pd.DataFrame, market: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    从逐日交易数据构造用户级时序特征（无需深度学习依赖的轻量级时序模块）。
    特征示例：EMA、波动、近窗趋势、与市场收益的相关性、近窗买入占比等。
    返回包含 user_id 的DataFrame。
    """
    required_cols = {"user_id", "day", "action"}
    if not required_cols.issubset(set(transactions.columns)):
        raise ValueError("transactions.csv 缺少必要列：user_id/day/action")

    # 优先使用transactions自带的mkt_ret；若不存在再从market映射，否则置0
    tx = transactions.copy()
    if "mkt_ret" not in tx.columns:
        if market is not None and {"day", "mkt_ret"}.issubset(set(market.columns)):
            tx = tx.merge(market[["day", "mkt_ret"]], on="day", how="left")
        else:
            tx["mkt_ret"] = 0.0

    # 仅保留需要列，确保类型正确
    tx = tx[["user_id", "day", "action", "mkt_ret"]].sort_values(["user_id", "day"]).reset_index(drop=True)

    feats = []
    # 分组计算
    for uid, g in tx.groupby("user_id", sort=False):
        a = g["action"].astype(float).values
        r = g["mkt_ret"].astype(float).values
        n = len(a)
        if n == 0:
            continue
        w = min(window, n)

        # 近窗切片
        a_win = a[-w:]
        r_win = r[-w:]
        days = np.arange(w)

        # EMA（平滑买卖倾向）
        def ema(x, span):
            alpha = 2.0 / (span + 1)
            out = 0.0
            for xi in x:
                out = alpha * xi + (1 - alpha) * out
            return out

        ema_10 = ema(a_win, min(10, w))
        ema_30 = ema(a_win, min(30, w))

        # 波动（行为稳定性）
        vol_30 = float(np.std(a_win)) if w > 1 else 0.0

        # 趋势（线性回归斜率）
        slope = float(np.polyfit(days, a_win, 1)[0]) if w > 1 else 0.0

        # 与市场相关性（反映顺势/逆势）
        if w > 1 and np.std(a_win) > 1e-8 and np.std(r_win) > 1e-8:
            corr = float(np.corrcoef(a_win, r_win)[0, 1])
        else:
            corr = 0.0

        # 近窗买入占比
        buy_ratio = float((a_win > 0).mean())

        feats.append({
            "user_id": uid,
            "action_ema_10": ema_10,
            "action_ema_30": ema_30,
            "action_vol_30": vol_30,
            "action_trend_30": slope,
            "action_mkt_corr_30": corr,
            "buy_ratio_30": buy_ratio,
        })

    return pd.DataFrame(feats)


def _ensure_data_ready(workdir: str = "work", outputs_dir: str = "outputs"):
    """若缺少前置数据，自动运行 M1/M2 生成。"""
    os.makedirs(workdir, exist_ok=True)
    need_m1 = not (os.path.exists(os.path.join(outputs_dir, "profiles.csv")) and
                   os.path.exists(os.path.join(outputs_dir, "transactions.csv")) and
                   os.path.exists(os.path.join(outputs_dir, "market.csv")))
    need_m2 = not (os.path.exists(os.path.join(workdir, "train_features.csv")) and
                   os.path.exists(os.path.join(workdir, "test_features.csv")))
    if need_m1:
        print("检测到缺少 M1 数据，正在自动生成…")
        try:
            m1 = importlib.import_module("data_process.M1_data_prep")
            m1.simulate()
        except Exception:
            subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), "..", "data_process", "M1_data_prep.py")])
    if need_m2:
        print("检测到缺少 M2 特征，正在自动生成…")
        try:
            m2 = importlib.import_module("data_process.M2_features_and_split")
            m2.process_features(m1_dir=outputs_dir, workdir=workdir)
        except Exception:
            subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), "..", "data_process", "M2_features_and_split.py")])


def load_data(workdir="work", outputs_dir="outputs", use_gnn=True, use_temporal=True):
    """
    读取M2生成的训练/测试数据，并按需融合GNN与时序特征。

    返回：
        X_train, y_train, X_test, y_test, test_user_ids
    """
    _ensure_data_ready(workdir=workdir, outputs_dir=outputs_dir)
    train = pd.read_csv(os.path.join(workdir, "train_features.csv"))
    test = pd.read_csv(os.path.join(workdir, "test_features.csv"))
    all_features = pd.read_csv(os.path.join(workdir, "all_features.csv")) if os.path.exists(os.path.join(workdir, "all_features.csv")) else pd.concat([train, test], axis=0, ignore_index=True)

    # 基础特征
    base_cols = ["age", "income10k", "asset10k", "exp_years", "action_mean", "q_sentiment"]
    present_base = [c for c in base_cols if c in all_features.columns]

    # 可选：时序特征
    if use_temporal:
        transactions = _safe_read_csv(os.path.join(outputs_dir, "transactions.csv"))
        market = _safe_read_csv(os.path.join(outputs_dir, "market.csv"))
        if transactions is not None:
            tfeat = _compute_temporal_features(transactions, market)
            all_features = all_features.merge(tfeat, on="user_id", how="left").fillna(0)
        else:
            print("未找到transactions.csv，跳过时序特征")

    # 可选：GNN特征（在全量用户级上构图并传播，避免拆分带来的图断裂）
    if use_gnn:
        try:
            A_hat = _build_user_graph(all_features)
            # 选择传播的列：基础行为/情绪以及时序平滑项（若存在）
            gnn_cols = [c for c in [
                "action_mean", "q_sentiment", "action_ema_10", "action_ema_30", "action_trend_30"
            ] if c in all_features.columns]
            if gnn_cols:
                all_features = _graph_propagate_features(all_features, A_hat, gnn_cols, steps=2)
            else:
                print("GNN传播列为空，跳过图传播")
        except Exception as e:
            print(f"构图或传播失败，跳过GNN特征：{e}")

    # 切回训练/测试集
    cols_in_all = set(all_features.columns)
    # 汇总可用特征列：基础 + 时序 + GNN
    extra_cols = [c for c in [
        "action_ema_10", "action_ema_30", "action_vol_30", "action_trend_30", "action_mkt_corr_30", "buy_ratio_30",
        "action_mean_gnn1", "q_sentiment_gnn1", "action_ema_10_gnn1", "action_ema_30_gnn1", "action_trend_30_gnn1",
        "action_mean_gnn2", "q_sentiment_gnn2", "action_ema_10_gnn2", "action_ema_30_gnn2", "action_trend_30_gnn2",
    ] if c in cols_in_all]

    X_cols = [c for c in present_base + extra_cols if c in cols_in_all]

    # 将增强后的列按user_id回灌到train/test
    all_idx = all_features[["user_id"] + X_cols].copy()
    train = train.drop(columns=[c for c in X_cols if c in train.columns], errors="ignore").merge(all_idx, on="user_id", how="left").fillna(0)
    test = test.drop(columns=[c for c in X_cols if c in test.columns], errors="ignore").merge(all_idx, on="user_id", how="left").fillna(0)

    return (
        train[X_cols], train["risk_label"],
        test[X_cols], test["risk_label"],
        test["user_id"], X_cols
    )


def fedavg_train(workdir="work", outputs_dir="outputs", n_clients=500, rounds=1, seed=42, use_gnn=True, use_temporal=True,
                 trees_per_client=80, max_depth=None):
    """
    用"联邦概率集成"训练随机森林基学习器：
      - 将训练数据划分为n个客户端；每个客户端训练一个随机森林（trees_per_client棵树）。
      - 使用样本量作为权重，对各客户端模型的预测概率进行加权平均，形成可持久化的FederatedEnsemble。
    参数：
      - outputs_dir: 原始逐日数据目录（默认"outputs"）
      - use_gnn/use_temporal: 是否启用增强特征
      - rounds: 逻辑回归的FedAvg回合在此不再生效，保留参数仅为兼容。随机森林直接一次训练即可。
      - trees_per_client: 每个客户端的树数（默认80）
      - max_depth: 树最大深度（None表示不限制）
    """
    # 1) 加载数据
    X_train, y_train, X_test, y_test, test_user_ids, X_cols = load_data(
        workdir=workdir, outputs_dir=outputs_dir, use_gnn=use_gnn, use_temporal=use_temporal
    )

    if len(X_train.columns) == 0:
        raise ValueError("没有可用的特征列，检查前置模块输出或开关设置")

    rng = np.random.default_rng(seed)

    # 2) 客户端划分
    indices = np.arange(len(X_train))
    rng.shuffle(indices)
    client_indices = np.array_split(indices, n_clients)

    # 3) 各客户端训练随机森林
    client_models = []
    client_weights = []
    for idx in client_indices:
        Xi, yi = X_train.iloc[idx], y_train.iloc[idx]
        # 处理标签单一的客户端：跳过（不给权重）
        if len(np.unique(yi)) < 2 or len(Xi) == 0:
            continue
        rf = RandomForestClassifier(
            n_estimators=trees_per_client,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=-1,
            class_weight=None
        )
        rf.fit(Xi, yi)
        client_models.append(rf)
        client_weights.append(len(Xi))

    if not client_models:
        raise RuntimeError("所有客户端训练失败或样本不足，无法构建联邦集成模型")

    # 4) 组装联邦集成器
    ensemble = FederatedEnsemble(client_models, client_weights)

    # 5) 评估
    y_prob = ensemble.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"M3联邦集成(随机森林) 准确率：{acc:.4f}，特征数={len(X_cols)}，客户端={len(client_models)}")

    # 6) 保存
    os.makedirs(workdir, exist_ok=True)
    joblib.dump(ensemble, os.path.join(workdir, "risk_model.pkl"))
    pd.Series(X_cols, name="feature").to_csv(os.path.join(workdir, "used_features_m3.csv"), index=False)

    predictions = pd.DataFrame({
        "user_id": test_user_ids,
        "risk_pred": y_pred,
        "risk_prob": y_prob
    })
    predictions.to_csv(os.path.join(workdir, "predictions.csv"), index=False)
    print(f"M3完成：联邦集成模型与预测已保存至 {workdir}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    work_dir = os.path.join(project_root, "work")
    # Default to 'data' if it exists, else 'outputs'
    data_dir = os.path.join(project_root, "data")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(project_root, "outputs")

    # 默认启用GNN与时序特征
    fedavg_train(workdir=work_dir, outputs_dir=data_dir)
