"""
HF 风险预测 baseline（中文分类模型）
- 输入：outputs/profiles.csv  (必须包含 user_id 与 risk_label)
- 输出：
  - work/predictions_hf_{alias}.csv   (user_id,risk_pred,risk_prob)
  - results/metrics_hf_risk_{alias}_{timestamp}.json

用法示例：
  python hf_risk_baseline.py --model hfl/chinese-macbert-base --alias macbert
  python hf_risk_baseline.py --model hfl/chinese-roberta-wwm-ext --alias roberta

依赖：
  pip install -U transformers datasets accelerate evaluate scikit-learn pandas numpy
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


DEFAULT_FEATURE_COLS = [
    "age",
    "education",
    "income10k",
    "asset10k",
    "debt10k",
    "children",
    "exp_years",
]


def serialize_profile_row(row: pd.Series, feature_cols: List[str]) -> str:
    """把结构化画像转成中文文本，让 HF 文本分类模型能吃进去。"""
    # 允许缺失：缺失就跳过
    parts = []
    if "age" in feature_cols and pd.notna(row.get("age")):
        parts.append(f"年龄{int(row['age'])}岁")
    if "education" in feature_cols and pd.notna(row.get("education")):
        parts.append(f"学历{str(row['education'])}")
    if "income10k" in feature_cols and pd.notna(row.get("income10k")):
        parts.append(f"年收入{float(row['income10k']):.1f}万")
    if "asset10k" in feature_cols and pd.notna(row.get("asset10k")):
        parts.append(f"总资产{float(row['asset10k']):.1f}万")
    if "debt10k" in feature_cols and pd.notna(row.get("debt10k")):
        parts.append(f"负债{float(row['debt10k']):.1f}万")
    if "children" in feature_cols and pd.notna(row.get("children")):
        parts.append(f"子女{int(row['children'])}人")
    if "exp_years" in feature_cols and pd.notna(row.get("exp_years")):
        parts.append(f"投资经验{int(row['exp_years'])}年")

    # 兜底：如果全缺失
    if not parts:
        return "用户画像信息缺失"
    return " ".join(parts)


def build_dataset_from_profiles(
    profiles_csv: str,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, Dataset]:
    df = pd.read_csv(profiles_csv)

    required = {"user_id", "risk_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{profiles_csv} 缺少列: {sorted(missing)}")

    df["text"] = df.apply(lambda r: serialize_profile_row(r, feature_cols), axis=1)
    df["label"] = df["risk_label"].astype(int)

    # HF datasets
    ds = Dataset.from_pandas(df[["user_id", "text", "label"]], preserve_index=False)
    return df, ds


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    probs = softmax_np(logits)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    # roc_auc 需要同时存在正负样本
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")

    return {"accuracy": acc, "f1": f1, "roc_auc": auc}


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def main():
    # Determine project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    work_dir = os.path.join(project_root, "work")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Default to 'data' if it exists, else 'outputs'
    data_dir = os.path.join(project_root, "data")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(project_root, "outputs")

    p = argparse.ArgumentParser()
    p.add_argument("--profiles_csv", default=os.path.join(data_dir, "profiles.csv"))
    p.add_argument("--model", required=True, help="HF model repo id, e.g. hfl/chinese-macbert-base")
    p.add_argument("--alias", required=True, help="short name used in output file names, e.g. macbert")
    p.add_argument("--feature_cols", default=",".join(DEFAULT_FEATURE_COLS))
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    # training hyperparams (small dataset 1000)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)

    args = p.parse_args()
    feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]

    os.makedirs("work", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    df, ds = build_dataset_from_profiles(args.profiles_csv, feature_cols)

    # 固定切分
    train_ids, test_ids = train_test_split(
        df["user_id"].values,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["risk_label"].values,
    )
    train_set = ds.filter(lambda x: x["user_id"] in set(train_ids))
    test_set = ds.filter(lambda x: x["user_id"] in set(test_ids))

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_set = train_set.map(tok, batched=True)
    test_set = test_set.map(tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("work", f"hf_risk_runs_{args.alias}_{ts}")

    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    # 生成 predictions（对齐你现有格式）
    pred_out = trainer.predict(test_set)
    logits = pred_out.predictions
    probs = softmax_np(logits)[:, 1]
    pred = (probs >= 0.5).astype(int)

    test_user_ids = np.array(test_set["user_id"])
    pred_df = pd.DataFrame(
        {"user_id": test_user_ids.astype(int), "risk_pred": pred.astype(int), "risk_prob": probs.astype(float)}
    ).sort_values("user_id")

    pred_path = os.path.join(work_dir, f"predictions_hf_{args.alias}.csv")
    pred_df.to_csv(pred_path, index=False)

    metrics_path = os.path.join(results_dir, f"metrics_hf_risk_{args.alias}_{ts}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "alias": args.alias,
                "profiles_csv": args.profiles_csv,
                "feature_cols": feature_cols,
                "test_size": args.test_size,
                "seed": args.seed,
                "train_args": {
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "max_length": args.max_length,
                },
                "eval": eval_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved:")
    print(" - predictions:", pred_path)
    print(" - metrics:", metrics_path)


if __name__ == "__main__":
    main()