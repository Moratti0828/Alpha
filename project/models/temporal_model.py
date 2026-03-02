"""
temporal_model.py

构建时间序列样本：将历史用户行为和市场信号（mkt_ret, mkt_vol）按窗口（seq_len / n_days）组织成序列，用于预测下一天的用户行为。

增强功能：
 - 增加市场滚动特征（例如 3/7/14 日均值）
 - 对缺失日期具有鲁棒性（缺失行为填充为无操作）
 - 支持可选的二分类目标（买入 vs 非买入）
 - 支持特征缩放（StandardScaler），并可保存到磁盘
 - 包含用于序列构造的 pytest 单元测试（位于独立文件）

使用示例（Windows cmd）：

python c:/Users/13412/Desktop/temporal_model.py --seq_len 30 --max_samples 20000 --epochs 10 --model auto --early_stop --save_scaler

python c:/Users/13412/Desktop/temporal_model.py --seq_len 30 --model rf --binary --max_samples 10000

"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# 尝试导入 TensorFlow；如果不可用则回退到 RandomForest
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


def load_and_enrich_market(market_path, rolling_windows=(3, 7, 14)):
    """读取 market CSV 并计算滚动统计量（针对 mkt_ret 与 mkt_vol）。
    返回按 day 排序并包含额外滚动列的 market DataFrame（例如 mkt_ret_roll_3）。
    """
    market = pd.read_csv(market_path)
    market = market.sort_values('day').reset_index(drop=True)
    for w in rolling_windows:
        market[f'mkt_ret_roll_{w}'] = market['mkt_ret'].rolling(window=w, min_periods=1).mean()
        market[f'mkt_vol_roll_{w}'] = market['mkt_vol'].rolling(window=w, min_periods=1).mean()
    return market


def build_sequences(transactions_path, market_path, seq_len=30, max_samples=None, add_rolling=True, rolling_windows=(3,7,14), binary=False):
    """返回 X, y，形状为 X.shape=(N, seq_len, n_features)，y 为整数标签。

    每个时间步的特征： [action_float, mkt_ret, mkt_vol, (可选的滚动特征...)]
    标签：下一天的 action，映射为 {-1->0, 0->1, 1->2}，或在 binary=True 时为二分类（1=买入，0=非买入）。

    鲁棒性：若用户在某些天缺失记录，函数会按 market 的 day 全量 reindex，并将缺失的 action 填为 0（无操作）。
    """
    if not os.path.exists(transactions_path):
        raise FileNotFoundError(f"未找到交易文件: {transactions_path}")
    if not os.path.exists(market_path):
        raise FileNotFoundError(f"未找到市场文件: {market_path}")

    tx = pd.read_csv(transactions_path)
    market = load_and_enrich_market(market_path, rolling_windows=rolling_windows if add_rolling else ())

    # 合并市场数据以确保每个 day 都有市场特征
    tx = tx.merge(market, on='day', how='right', suffixes=(None, '_mkt'))

    # 强制类型
    tx['user_id'] = tx['user_id'].astype('Int64')

    # 将动作映射为类别：-1->0, 0->1, 1->2；缺失动作视为 0（无操作）
    tx['action'] = tx['action'].fillna(0).astype(float)
    tx['action_cat'] = tx['action'].map({-1: 0, 0: 1, 1: 2}).astype(int)

    X_list, y_list = [], []

    # 用于保持连续性的日期范围
    all_days = market['day'].unique()
    min_day, max_day = int(all_days.min()), int(all_days.max())

    # 预先计算特征列名，避免在无样本时未定义 feat_cols
    feat_cols = ['action', 'mkt_ret', 'mkt_vol']
    if add_rolling:
        for w in rolling_windows:
            feat_cols.append(f'mkt_ret_roll_{w}')
            feat_cols.append(f'mkt_vol_roll_{w}')

    # 对每个用户按完整日期重索引以处理缺失日期
    for uid, g in tx.groupby('user_id'):
        # 如果 user_id 为 NaN（合并产生）则跳过
        if pd.isna(uid):
            continue
        g = g.set_index('day')
        # 重新索引为完整的市场日期范围：缺失的行会有 NaN 的 action
        g = g.reindex(range(min_day, max_day+1))
        # 填充市场列（如果存在缺失），通常合并时市场列已存在，这里做保护性填充
        market_cols = [c for c in market.columns if c not in ['day']]
        for c in market_cols:
            if c in g.columns and g[c].isna().any():
                # 从 market 表中按 day 对齐填充
                g[c] = g[c].fillna(pd.Series(market.set_index('day')[c]))
        # 缺失的用户动作填为 0（无操作）
        if 'action' not in g.columns:
            g['action'] = 0.0
        g['action'] = g['action'].fillna(0.0)
        # 保证 action_cat 一致
        g['action_cat'] = g['action'].map({-1: 0, 0: 1, 1: 2}).fillna(1).astype(int)

        # 如果长度不足以构造一个样本（seq_len + 1）则跳过
        if len(g) < seq_len + 1:
            continue

        # 构建每个时间步的特征（使用预先计算的 feat_cols）
        feats = g[feat_cols].values.astype(float)
        cats = g['action_cat'].values.astype(int)

        # 滑动窗口生成样本
        for i in range(0, len(g) - seq_len):
            X_list.append(feats[i:i+seq_len])
            label = cats[i+seq_len]
            if binary:
                # 二分类：若下一天为买入（类别 2）则标记为 1，否则为 0
                y_list.append(1 if label == 2 else 0)
            else:
                y_list.append(label)
            if max_samples and len(X_list) >= max_samples:
                break
        if max_samples and len(X_list) >= max_samples:
            break

    if len(X_list) == 0:
        return np.empty((0, seq_len, len(feat_cols))), np.empty((0,), dtype=int), feat_cols

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, feat_cols


def build_lstm(seq_len, n_features, n_classes=3, latent=64, dropout=0.2):
    """构建并返回一个简单的 LSTM 模型用于序列分类。"""
    model = Sequential([
        LSTM(latent, input_shape=(seq_len, n_features)),
        Dropout(dropout),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train(args):
    # Determine project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    work_dir = os.path.join(project_root, 'work')
    os.makedirs(work_dir, exist_ok=True)

    # Default to 'data' if it exists, else 'outputs'
    data_dir = os.path.join(project_root, "data")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(project_root, "outputs")

    tx_path = args.transactions or os.path.join(data_dir, 'transactions.csv')
    market_path = args.market or os.path.join(data_dir, 'market.csv')

    X, y, feat_cols = build_sequences(tx_path, market_path, seq_len=args.seq_len, max_samples=args.max_samples, add_rolling=args.add_rolling, rolling_windows=tuple(args.rolling_windows), binary=args.binary)
    if X.size == 0:
        raise RuntimeError('没有足够的数据构建序列，请先运行 M1_data_prep.simulate() 或检查 CSV 文件')

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    if args.model == 'auto':
        model_choice = 'lstm' if TF_AVAILABLE else 'rf'
    else:
        model_choice = args.model

    print(f"样本数: {len(X)}, 训练: {len(X_train)}, 测试: {len(X_test)}, 模型: {model_choice}, 特征: {feat_cols}")

    # 对于基于树的模型需要将序列展开为向量；若使用 LSTM 则按时间步保留三维
    if args.scale:
        # 在展平的训练集上拟合 scaler（保留各时间步分布）
        n_samples, seq_len, n_feat = X_train.shape
        scaler = StandardScaler()
        X_train_flat = X_train.reshape((n_samples * seq_len, n_feat))
        scaler.fit(X_train_flat)
        # 应用缩放
        X_train = scaler.transform(X_train_flat).reshape((n_samples, seq_len, n_feat))
        X_test = scaler.transform(X_test.reshape((X_test.shape[0] * seq_len, n_feat))).reshape((X_test.shape[0], seq_len, n_feat))
        if args.save_scaler:
            joblib.dump({'scaler': scaler, 'feat_cols': feat_cols, 'seq_len': seq_len}, os.path.join(work_dir, 'scaler.joblib'))
            print(f'已保存 scaler 到 {os.path.join(work_dir, "scaler.joblib")}')
    else:
        n_feat = X_train.shape[2]

    if model_choice == 'lstm':
        if not TF_AVAILABLE:
            raise RuntimeError('TensorFlow 未安装，无法训练 LSTM，请安装 tensorflow 或使用 --model rf')
        model = build_lstm(args.seq_len, X.shape[2], n_classes=len(np.unique(y)), latent=args.latent, dropout=args.dropout)
        callbacks = []
        if args.early_stop:
            callbacks.append(tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True))
        model.fit(X_train, y_train, validation_split=0.1, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=2)
        y_pred = np.argmax(model.predict(X_test, batch_size=args.batch_size), axis=1)
        out_path = os.path.join(work_dir, 'temporal_model_lstm.h5')
        model.save(out_path)

    elif model_choice == 'rf':
        # 展平序列为向量以供决策树/随机森林使用
        n_samples, seq_len, n_feat = X_train.shape
        X_train_flat = X_train.reshape((n_samples, seq_len * n_feat))
        X_test_flat = X_test.reshape((X_test.shape[0], seq_len * n_feat))
        clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed, n_jobs=-1)
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        out_path = os.path.join(work_dir, 'temporal_model_rf.joblib')
        joblib.dump({'model': clf, 'seq_len': args.seq_len, 'n_feat': n_feat, 'feat_cols': feat_cols, 'binary': args.binary}, out_path)
    else:
        raise ValueError('未知的模型选择: ' + str(model_choice))

    print('\n分类报告:')
    print(classification_report(y_test, y_pred, digits=4))
    print('\n混淆矩阵:')
    print(confusion_matrix(y_test, y_pred))
    print(f'模型已保存到: {out_path}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--transactions', type=str, default=None, help='交易文件路径（transactions.csv）')
    p.add_argument('--market', type=str, default=None, help='市场文件路径（market.csv）')
    p.add_argument('--seq_len', type=int, default=30, help='序列长度（n_days）')
    # 提高默认的 max_samples，以便使用更多数据进行训练（若机器资源允许）
    p.add_argument('--max_samples', type=int, default=100000, help='限制训练序列数量')
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    # 增加默认 epochs，LSTM 训练使用更多轮次以收敛
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--model', type=str, default='auto', choices=['auto', 'lstm', 'rf'], help='模型类型（auto/lstm/rf）')
    p.add_argument('--latent', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--n_estimators', type=int, default=200)
    p.add_argument('--early_stop', action='store_true')
    p.add_argument('--add_rolling', dest='add_rolling', action='store_true')
    p.add_argument('--no_rolling', dest='add_rolling', action='store_false')
    p.set_defaults(add_rolling=True)
    p.add_argument('--rolling_windows', nargs='+', type=int, default=[3,7,14], help='要计算的滚动窗口列表，例如 3 7 14')
    p.add_argument('--binary', action='store_true', help='训练二分类：买入 vs 非买入')
    p.add_argument('--scale', action='store_true', help='对特征应用 StandardScaler')
    p.add_argument('--save_scaler', action='store_true', help='保存拟合的 scaler 到 work/scaler.joblib')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
