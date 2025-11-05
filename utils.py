# -*- coding: utf-8 -*-
"""
工具函数模块
============

提供配置验证、数据验证等辅助功能

作者: Muqy
创建日期: 2025-03-06
"""

import os
import sys
from typing import Dict, Any, List, Optional


def validate_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    验证配置文件的有效性

    参数：
    ----------
    config : dict
        从config.yaml加载的配置字典

    返回：
    ----------
    tuple : (is_valid, error_messages)
        - is_valid: bool, 配置是否有效
        - error_messages: list, 错误信息列表
    """
    errors = []

    # 检查必需的配置项
    required_fields = ['SEED', 'LABEL', 'TRAIN_DATA_PATH', 'TEST_DATA_PATH',
                       'OUTPUT_DIR', 'MODELS_DIR', 'TIME_LIMIT',
                       'EVAL_METRIC', 'PRESETS', 'PREDICTION_TYPE']

    for field in required_fields:
        if field not in config:
            errors.append(f"缺少必需配置项: {field}")

    # 验证数据类型
    if 'SEED' in config and not isinstance(config['SEED'], int):
        errors.append("SEED 必须是整数")

    if 'TIME_LIMIT' in config and not isinstance(config['TIME_LIMIT'], (int, float)):
        errors.append("TIME_LIMIT 必须是数字")

    if 'LABEL' in config and not isinstance(config['LABEL'], str):
        errors.append("LABEL 必须是字符串")

    # 验证文件路径
    if 'TRAIN_DATA_PATH' in config:
        train_path = config['TRAIN_DATA_PATH']
        if not os.path.exists(train_path):
            errors.append(f"训练数据文件不存在: {train_path}")

    if 'TEST_DATA_PATH' in config:
        test_path = config['TEST_DATA_PATH']
        if not os.path.exists(test_path):
            errors.append(f"测试数据文件不存在: {test_path}")

    # 验证评估指标
    if 'EVAL_METRIC' in config:
        valid_metrics = [
            # 分类指标
            'roc_auc', 'accuracy', 'f1', 'precision', 'recall', 'log_loss', 'pac_score',
            # 回归指标
            'rmse', 'mse', 'mae', 'r2',
            'root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error'
        ]
        if config['EVAL_METRIC'].lower() not in valid_metrics:
            errors.append(f"不支持的评估指标: {config['EVAL_METRIC']}. "
                         f"支持的指标: {', '.join(valid_metrics)}")

    # 验证预设
    if 'PRESETS' in config:
        valid_presets = ['best_quality', 'high_quality', 'medium_quality',
                        'good_quality', 'optimize_for_deployment']
        if config['PRESETS'] not in valid_presets:
            errors.append(f"不支持的预设: {config['PRESETS']}. "
                         f"支持的预设: {', '.join(valid_presets)}")

    # 验证预测类型
    if 'PREDICTION_TYPE' in config:
        valid_types = ['class', 'prob', 'both']
        if config['PREDICTION_TYPE'] not in valid_types:
            errors.append(f"不支持的预测类型: {config['PREDICTION_TYPE']}. "
                         f"支持的类型: {', '.join(valid_types)}")

    return len(errors) == 0, errors


def validate_data(train_df, test_df, label_col: str) -> tuple[bool, List[str]]:
    """
    验证数据的有效性

    参数：
    ----------
    train_df : pd.DataFrame
        训练数据
    test_df : pd.DataFrame
        测试数据
    label_col : str
        目标列名称

    返回：
    ----------
    tuple : (is_valid, warning_messages)
        - is_valid: bool, 数据是否有效
        - warning_messages: list, 警告信息列表
    """
    warnings = []

    # 检查训练集是否包含目标列
    if label_col not in train_df.columns:
        warnings.append(f"错误: 训练数据中不存在目标列 '{label_col}'")
        return False, warnings

    # 检查测试集是否包含目标列（不应该包含）
    if label_col in test_df.columns:
        warnings.append(f"警告: 测试数据中包含目标列 '{label_col}'，这可能不是预期的行为")

    # 检查训练集是否为空
    if len(train_df) == 0:
        warnings.append("错误: 训练数据为空")
        return False, warnings

    # 检查测试集是否为空
    if len(test_df) == 0:
        warnings.append("错误: 测试数据为空")
        return False, warnings

    # 检查目标列是否有缺失值
    if train_df[label_col].isnull().any():
        missing_count = train_df[label_col].isnull().sum()
        warnings.append(f"警告: 目标列包含 {missing_count} 个缺失值")

    # 检查列名是否一致（除了目标列）
    train_cols = set(train_df.columns) - {label_col}
    test_cols = set(test_df.columns)

    only_in_train = train_cols - test_cols
    only_in_test = test_cols - train_cols

    if only_in_train:
        warnings.append(f"警告: 以下列只存在于训练集: {', '.join(only_in_train)}")

    if only_in_test:
        warnings.append(f"警告: 以下列只存在于测试集: {', '.join(only_in_test)}")

    # 检查数据大小
    if len(train_df) < 100:
        warnings.append(f"警告: 训练数据量较小 ({len(train_df)} 行)，可能影响模型性能")

    return True, warnings


def print_data_info(train_df, test_df, label_col: str):
    """
    打印数据集的基本信息

    参数：
    ----------
    train_df : pd.DataFrame
        训练数据
    test_df : pd.DataFrame
        测试数据
    label_col : str
        目标列名称
    """
    print("\n" + "="*60)
    print("数据集信息")
    print("="*60)

    print(f"\n训练集:")
    print(f"  - 样本数量: {len(train_df)}")
    print(f"  - 特征数量: {len(train_df.columns) - 1}")  # 减去目标列
    print(f"  - 目标列: {label_col}")

    # 打印目标列的统计信息
    if train_df[label_col].dtype == 'object' or len(train_df[label_col].unique()) < 20:
        print(f"  - 目标分布:")
        value_counts = train_df[label_col].value_counts()
        for value, count in value_counts.items():
            print(f"      {value}: {count} ({count/len(train_df)*100:.2f}%)")
    else:
        print(f"  - 目标统计:")
        print(f"      均值: {train_df[label_col].mean():.4f}")
        print(f"      中位数: {train_df[label_col].median():.4f}")
        print(f"      标准差: {train_df[label_col].std():.4f}")
        print(f"      最小值: {train_df[label_col].min():.4f}")
        print(f"      最大值: {train_df[label_col].max():.4f}")

    print(f"\n测试集:")
    print(f"  - 样本数量: {len(test_df)}")
    print(f"  - 特征数量: {len(test_df.columns)}")

    # 打印数据类型分布
    train_dtypes = train_df.dtypes.value_counts()
    print(f"\n特征类型分布 (训练集):")
    for dtype, count in train_dtypes.items():
        print(f"  - {dtype}: {count} 列")

    # 打印缺失值信息
    train_missing = train_df.isnull().sum().sum()
    test_missing = test_df.isnull().sum().sum()

    if train_missing > 0 or test_missing > 0:
        print(f"\n缺失值统计:")
        print(f"  - 训练集: {train_missing} 个缺失值")
        print(f"  - 测试集: {test_missing} 个缺失值")

    print("="*60 + "\n")
