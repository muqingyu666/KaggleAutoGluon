# -*- coding: utf-8 -*-
"""EDA（探索性数据分析）工具模块。

提供数据概览、目标分布、相关性、特征重要性等常用可视化函数。
所有函数独立可用，可在 run.py 中按需调用，也可 `from eda import summarize` 单独使用。
用户可在此文件中自由添加自己的 EDA 分析函数。
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境兼容
import matplotlib.pyplot as plt


def summarize(df, target=None):
    """打印数据概览信息。

    Args:
        df: DataFrame
        target: 目标列名（可选，如提供则打印目标分布信息）
    """
    print("=" * 50)
    print(f"数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"\n列类型:")
    print(df.dtypes.value_counts().to_string())
    print(f"\n缺失值:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing.to_string())
    else:
        print("  无缺失值")

    if target and target in df.columns:
        print(f"\n目标列 [{target}] 分布:")
        if df[target].nunique() <= 20:
            print(df[target].value_counts().to_string())
        else:
            print(df[target].describe().to_string())
    print("=" * 50)


def plot_target(df, target, output_dir="results"):
    """绘制目标变量分布图。

    Args:
        df: 包含目标列的 DataFrame
        target: 目标列名
        output_dir: 图片保存目录
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    if df[target].nunique() <= 20:
        df[target].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"目标分布: {target}")
    else:
        df[target].hist(bins=30, ax=ax)
        ax.set_title(f"目标分布: {target}")

    ax.set_xlabel(target)
    ax.set_ylabel("计数")
    plt.tight_layout()
    path = os.path.join(output_dir, "target_distribution.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"目标分布图已保存: {path}")


def plot_correlations(df, output_dir="results"):
    """绘制数值特征相关性热力图。

    Args:
        df: DataFrame
        output_dir: 图片保存目录
    """
    os.makedirs(output_dir, exist_ok=True)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        print("[跳过] 数值特征不足，无法绘制相关性图")
        return

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im)
    ax.set_title("特征相关性")
    plt.tight_layout()
    path = os.path.join(output_dir, "correlations.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"相关性热力图已保存: {path}")


def plot_feature_importance(model, feature_names, output_dir="results"):
    """绘制特征重要性条形图（支持 tree-based 模型）。

    Args:
        model: 已训练的模型（需有 feature_importances_ 属性）
        feature_names: 特征名列表
        output_dir: 图片保存目录
    """
    if not hasattr(model, "feature_importances_"):
        print("[跳过] 该模型不支持 feature_importances_")
        return

    os.makedirs(output_dir, exist_ok=True)
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=(10, max(5, len(feature_names) * 0.3)))
    y_pos = range(len(feature_names))
    ax.barh(y_pos, importance[indices[::-1]])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=8)
    ax.set_xlabel("重要性")
    ax.set_title("特征重要性")
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"特征重要性图已保存: {path}")


# ========================================
# 在下方添加你的自定义 EDA 函数
# ========================================
# 例如:
# def plot_missing_pattern(df, output_dir="results"):
#     """绘制缺失值模式图。"""
#     ...
#
# def analyze_feature(df, col, target=None):
#     """分析单个特征与目标的关系。"""
#     ...
