"""
联邦集成器（FederatedEnsemble）

用途：
- 将多个已训练好的二分类基学习器（如随机森林、逻辑回归等）与其样本权重进行组合，
  通过对各模型的 predict_proba 概率进行加权平均，得到集成预测概率与标签。
- 可被 joblib/pickle 序列化，便于在 M3 训练后持久化并在其它脚本中加载使用。

使用方式：
- 创建：ens = FederatedEnsemble([clf1, clf2, ...], weights=[n1, n2, ...])
- 预测概率：proba = ens.predict_proba(X)  # 返回形状 (n_samples, 2)
- 预测类别：pred = ens.predict(X)        # 返回 0/1
"""

import numpy as np

class FederatedEnsemble:
    """
    简单的联邦集成模型。

    参数：
    - estimators: 可迭代的已训练分类器，每个应至少实现 predict_proba(X) 或 decision_function(X)
    - weights: 与 estimators 等长的权重序列（通常使用各客户端样本数），若为 None 则等权重

    属性：
    - estimators: 存储的基学习器列表
    - weights:   numpy.ndarray 形式的非负权重（为0的将被置为1避免退化）
    - classes_:  推断的类别标签数组（优先读取第一个含 classes_ 的学习器；若都无，则默认为 [0,1]）

    使用示例：
    >>> ens = FederatedEnsemble([rf_a, rf_b], weights=[1000, 800])
    >>> proba = ens.predict_proba(X_test)
    >>> yhat = ens.predict(X_test)
    """
    def __init__(self, estimators, weights=None):
        self.estimators = list(estimators)
        self.weights = (np.array(weights, dtype=float)
                        if weights is not None else np.ones(len(self.estimators), dtype=float))
        self.weights = np.where(self.weights > 0, self.weights, 1.0)
        self.classes_ = None
        for est in self.estimators:
            if hasattr(est, "classes_"):
                self.classes_ = np.array(est.classes_)
                break
        if self.classes_ is None:
            self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        """
        集成概率预测：对各基学习器的概率进行加权平均。

        参数：
        - X: 形状 (n_samples, n_features) 的特征矩阵

        返回：
        - probs: 形状 (n_samples, 2) 的类别概率，第二列为正类(1)概率

        说明：
        - 若某基学习器无 predict_proba，则尝试使用 decision_function 并通过 sigmoid 近似概率。
        - 若无学习器，返回常数 0.5 概率。
        """
        if len(self.estimators) == 0:
            return np.full((len(X), 2), 0.5, dtype=float)
        probs = None
        wsum = float(self.weights.sum())
        for w, est in zip(self.weights, self.estimators):
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(X)
            else:
                s = est.decision_function(X)
                p1 = 1.0 / (1.0 + np.exp(-s))
                p = np.vstack([1 - p1, p1]).T
            probs = p * (w / wsum) if probs is None else probs + p * (w / wsum)
        return probs

    def predict(self, X):
        """
        集成类别预测：
        - 以 predict_proba(X) 的第二列（正类）概率 >= 0.5 为阈值，输出 0/1 标签。

        参数：
        - X: 形状 (n_samples, n_features) 的特征矩阵

        返回：
        - y_pred: 形状 (n_samples,) 的 0/1 预测标签
        """
        p = self.predict_proba(X)
        return (p[:, 1] >= 0.5).astype(int)
