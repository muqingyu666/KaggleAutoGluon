# -*- coding: utf-8 -*-
"""数据加载与预处理模块。

提供 CSV 数据的加载、特征/目标分离、缺失值填充、类别编码等基础预处理功能。
用户可根据具体比赛需求在此基础上修改或扩展。
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(cfg):
    """读取训练集、测试集和可选的 sample_submission。

    Args:
        cfg: 配置字典，需包含 train_path, test_path, sample_submission_path。

    Returns:
        train_df, test_df, sample_sub (DataFrame or None)
    """
    train_df = pd.read_csv(cfg["train_path"])
    test_df = pd.read_csv(cfg["test_path"])

    sample_sub = None
    sub_path = cfg.get("sample_submission_path")
    if sub_path:
        try:
            sample_sub = pd.read_csv(sub_path)
        except FileNotFoundError:
            print(f"[警告] sample_submission 未找到: {sub_path}")

    print(f"训练集: {train_df.shape}  测试集: {test_df.shape}")
    return train_df, test_df, sample_sub


def preprocess(train_df, test_df, cfg):
    """基础预处理：分离特征/目标、编码类别特征、填充缺失值。

    处理逻辑：
    1. 分离目标列和 ID 列
    2. 对齐训练集和测试集的列
    3. LabelEncoder 编码 object/category 列
    4. 用训练集中位数填充数值型缺失值

    Args:
        train_df: 训练数据 DataFrame
        test_df: 测试数据 DataFrame
        cfg: 配置字典

    Returns:
        X_train, y_train, X_test, feature_names
    """
    target = cfg["target"]
    id_col = cfg.get("id_col")

    y_train = train_df[target].copy()

    # 确定要从特征中移除的列
    drop_cols = [target]
    if id_col and id_col in train_df.columns:
        drop_cols.append(id_col)

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    # 对齐列（测试集可能缺少某些列）
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()

    # 编码类别特征
    for col in X_train.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        # 在 train + test 合并值上 fit，避免测试集出现未知类别
        combined = pd.concat([X_train[col], X_test[col]]).astype(str).fillna("_missing_")
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str).fillna("_missing_"))
        X_test[col] = le.transform(X_test[col].astype(str).fillna("_missing_"))

    # 用训练集中位数填充数值型缺失值
    for col in X_train.columns:
        if X_train[col].isnull().any() or X_test[col].isnull().any():
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

    feature_names = list(common_cols)
    print(f"特征数: {len(feature_names)} 列")
    return X_train, y_train, X_test, feature_names
