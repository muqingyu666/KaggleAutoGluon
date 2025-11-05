#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速开始脚本
============

帮助用户快速配置和启动AutoGluon训练流程

使用方法:
    python quick_start.py

作者: Muqy
创建日期: 2025-03-06
"""

import os
import sys
import yaml
import shutil
from pathlib import Path


def print_header():
    """打印欢迎信息"""
    print("\n" + "="*80)
    print("Kaggle AutoGluon 快速开始向导".center(80))
    print("="*80)
    print("\n本向导将帮助你配置AutoGluon训练流程\n")


def check_environment():
    """检查环境和依赖"""
    print("[ 1 ] 检查环境...")

    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要Python 3.8+")
        return False

    print(f"  ✓ Python版本: {sys.version.split()[0]}")

    # 检查关键依赖
    required_packages = ['pandas', 'numpy', 'yaml', 'autogluon']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package} 已安装")
        except ImportError:
            print(f"  ❌ {package} 未安装")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n请先安装缺失的依赖:")
        print(f"  pip install -r requirements.txt\n")
        return False

    return True


def get_user_input():
    """获取用户输入的配置信息"""
    print("\n[ 2 ] 配置参数...")

    config = {}

    # 数据文件路径
    print("\n数据文件路径:")
    print("  提示: 请将数据文件放入 data/ 目录，或提供完整路径")

    train_path = input("  训练数据文件路径 [data/train_data.csv]: ").strip()
    config['TRAIN_DATA_PATH'] = train_path if train_path else "data/train_data.csv"

    test_path = input("  测试数据文件路径 [data/test_data.csv]: ").strip()
    config['TEST_DATA_PATH'] = test_path if test_path else "data/test_data.csv"

    sample_path = input("  样本提交文件路径 (可选，直接回车跳过) [sample_submission.csv]: ").strip()
    config['SAMPLE_SUBMISSION_PATH'] = sample_path if sample_path else "sample_submission.csv"

    # 检查文件是否存在
    if not os.path.exists(config['TRAIN_DATA_PATH']):
        print(f"  ⚠️  警告: 训练数据文件不存在: {config['TRAIN_DATA_PATH']}")

    if not os.path.exists(config['TEST_DATA_PATH']):
        print(f"  ⚠️  警告: 测试数据文件不存在: {config['TEST_DATA_PATH']}")

    # 目标列名称
    print("\n目标列配置:")
    label = input("  目标列名称 (例如: income, price, target) [income]: ").strip()
    config['LABEL'] = label if label else "income"

    # 任务类型和评估指标
    print("\n任务类型:")
    print("  1. 分类任务 (Classification)")
    print("  2. 回归任务 (Regression)")

    task_choice = input("  选择任务类型 [1]: ").strip()

    if task_choice == "2":
        print("\n评估指标 (回归):")
        print("  1. RMSE (均方根误差)")
        print("  2. MAE (平均绝对误差)")
        print("  3. R² (决定系数)")
        metric_choice = input("  选择评估指标 [1]: ").strip()

        metrics_map = {"1": "rmse", "2": "mae", "3": "r2"}
        config['EVAL_METRIC'] = metrics_map.get(metric_choice, "rmse")
        config['PREDICTION_TYPE'] = "both"  # 回归任务此项无效

    else:  # 分类任务
        print("\n评估指标 (分类):")
        print("  1. ROC AUC (推荐用于二分类)")
        print("  2. Accuracy (准确率)")
        print("  3. F1 Score")
        metric_choice = input("  选择评估指标 [1]: ").strip()

        metrics_map = {"1": "roc_auc", "2": "accuracy", "3": "f1"}
        config['EVAL_METRIC'] = metrics_map.get(metric_choice, "roc_auc")

        print("\n预测输出类型:")
        print("  1. both - 同时输出类别和概率 (推荐)")
        print("  2. class - 仅输出类别")
        print("  3. prob - 仅输出概率")
        pred_choice = input("  选择输出类型 [1]: ").strip()

        pred_map = {"1": "both", "2": "class", "3": "prob"}
        config['PREDICTION_TYPE'] = pred_map.get(pred_choice, "both")

    # 训练时间
    print("\n训练时间:")
    print("  建议值:")
    print("    - 快速测试: 300-600 秒 (5-10分钟)")
    print("    - 正式训练: 3600 秒 (1小时)")
    print("    - 高质量模型: 7200+ 秒 (2小时以上)")

    time_limit = input("  训练时间限制（秒） [3600]: ").strip()
    config['TIME_LIMIT'] = int(time_limit) if time_limit.isdigit() else 3600

    # 质量预设
    print("\n模型质量预设:")
    print("  1. best_quality - 最高质量 (训练时间最长)")
    print("  2. high_quality - 高质量")
    print("  3. medium_quality - 中等质量 (平衡质量和速度)")
    preset_choice = input("  选择质量预设 [1]: ").strip()

    preset_map = {"1": "best_quality", "2": "high_quality", "3": "medium_quality"}
    config['PRESETS'] = preset_map.get(preset_choice, "best_quality")

    # 其他固定配置
    config['SEED'] = 0
    config['OUTPUT_DIR'] = "results"
    config['MODELS_DIR'] = "autogluon_models"

    return config


def save_config(config):
    """保存配置到config.yaml"""
    print("\n[ 3 ] 保存配置...")

    # 备份原配置文件（如果存在）
    if os.path.exists("config.yaml"):
        backup_path = "config.yaml.backup"
        shutil.copy("config.yaml", backup_path)
        print(f"  ✓ 原配置文件已备份至: {backup_path}")

    # 写入新配置
    with open("config.yaml", 'w', encoding='utf-8') as f:
        f.write("# ==============================\n")
        f.write("# Kaggle AutoGluon 配置文件\n")
        f.write("# ==============================\n")
        f.write("# 由快速开始向导自动生成\n\n")

        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print("  ✓ 配置已保存到 config.yaml")


def print_summary(config):
    """打印配置摘要"""
    print("\n[ 4 ] 配置摘要:")
    print("-" * 80)
    print(f"  训练数据:       {config['TRAIN_DATA_PATH']}")
    print(f"  测试数据:       {config['TEST_DATA_PATH']}")
    print(f"  目标列:         {config['LABEL']}")
    print(f"  评估指标:       {config['EVAL_METRIC']}")
    print(f"  训练时间:       {config['TIME_LIMIT']} 秒")
    print(f"  质量预设:       {config['PRESETS']}")
    print(f"  预测类型:       {config['PREDICTION_TYPE']}")
    print("-" * 80)


def main():
    """主函数"""
    print_header()

    # 检查环境
    if not check_environment():
        return

    # 获取用户配置
    config = get_user_input()

    # 保存配置
    save_config(config)

    # 打印摘要
    print_summary(config)

    # 询问是否立即开始训练
    print("\n" + "="*80)
    start_training = input("\n是否立即开始训练? [y/N]: ").strip().lower()

    if start_training == 'y':
        print("\n开始训练...")
        print("-" * 80)
        os.system("python autogluon_pipeline.py")
    else:
        print("\n配置完成！")
        print("\n你可以随时运行以下命令开始训练:")
        print("  python autogluon_pipeline.py")

    print("\n" + "="*80)
    print("感谢使用 Kaggle AutoGluon!".center(80))
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户取消操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1)
