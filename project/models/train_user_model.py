"""
train_user_model.py (升级版)
功能：训练“用户画像塔” (Random Forest)
升级点：
1. 纳入 'education' (学历) -> 需转为数字
2. 纳入 'action_mean' (交易倾向) -> 需从 M2 特征中读取
"""
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train():
    # Determine project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    work_dir = os.path.join(project_root, "work")
    os.makedirs(work_dir, exist_ok=True)

    # Default to 'data' if it exists, else 'outputs'
    data_dir = os.path.join(project_root, "data")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(project_root, "outputs")

    # 1. 读取数据
    # 优先读取 M2 生成的 all_features.csv，因为它已经合并了交易特征(action_mean)
    data_path = os.path.join(work_dir, "all_features.csv")

    # 兜底：如果 M2 没运行，尝试读 M1 的 profiles 并不全默认值
    if not os.path.exists(data_path):
        print(f"⚠️ 未找到 {data_path}，尝试读取 profiles.csv...")
        data_path = os.path.join(data_dir, "profiles.csv")
        if not os.path.exists(data_path):
            print("❌ 数据缺失！请先运行: python M1_data_prep.py && python M2_features_and_split.py")
            return
        df = pd.read_csv(data_path)
        # 补全缺失列 (如果是只运行了 M1)
        if 'action_mean' not in df.columns:
            print("⚠️ 补全 action_mean 为默认值 0")
            df['action_mean'] = 0.0
    else:
        print(f"✅ 读取特征文件: {data_path}")
        df = pd.read_csv(data_path)

    # 2. 特征工程 (必须与 API 端的处理逻辑严格一致！)
    
    # A. 学历映射 (String -> Int)
    # 这种手动映射比 OneHot 更适合树模型处理有序关系 (高中 < 本科 < 硕士)
    edu_map = {
        "高中及以下": 0,
        "大专": 1,
        "本科": 2,
        "硕士及以上": 3
    }
    # 如果数据里有未知的学历，默认为 2 (本科)
    df['education_int'] = df['education'].map(edu_map).fillna(2).astype(int)

    # B. 定义特征列表 (注意顺序！)
    features = [
        'age',          # 年龄
        'income10k',    # 收入
        'asset10k',     # 资产
        'debt10k',      # 负债
        'children',     # 子女
        'exp_years',    # 经验
        'action_mean',  # 交易倾向 (-1~1)
        'education_int' # 学历 (0~3)
    ]
    target = 'risk_label' # 0:稳健, 1:激进

    print(f"🚀 特征列表: {features}")

    X = df[features].fillna(0)
    y = df[target]

    # 3. 训练模型
    print(f"🚀 开始训练随机森林 (样本数: {len(df)})...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 增加树的数量和深度，提高拟合能力
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    # 4. 评估
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ 训练完成！测试集准确率: {acc:.2%}")

    # 5. 保存
    save_path = os.path.join(work_dir, "user_model.pkl")
    joblib.dump(clf, save_path)
    print(f"💾 模型已保存至: {save_path}")
    print("👉 请务必将此文件上传到腾讯云，覆盖旧模型！")

if __name__ == "__main__":
    train()