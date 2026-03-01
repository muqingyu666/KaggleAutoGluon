# -*- coding: utf-8 -*-
"""KaggleML — 通用机器学习竞赛快速测试框架。

用法:
    python run.py                  # 使用默认 config.yaml
    python run.py --config my.yaml # 使用自定义配置文件

流程:
    加载数据 → (可选 EDA) → 预处理 → 训练多模型 CV 对比 → 最佳模型预测 → 保存提交文件
"""

import os
import argparse
import yaml
import pandas as pd

from preprocess import load_data, preprocess
from models import train_and_evaluate, predict_test, run_autogluon
from eda import summarize, plot_target, plot_correlations, plot_feature_importance


def save_submission(predictions, test_df, cfg, sample_sub=None):
    """保存预测结果为提交文件。

    Args:
        predictions: 模型预测结果
        test_df: 原始测试数据（用于提取 ID 列）
        cfg: 配置字典
        sample_sub: 可选的 sample_submission DataFrame
    """
    output_dir = cfg.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)

    target = cfg["target"]
    id_col = cfg.get("id_col")

    if sample_sub is not None:
        submission = sample_sub.copy()
        submission[target] = predictions
    elif id_col and id_col in test_df.columns:
        submission = pd.DataFrame({id_col: test_df[id_col], target: predictions})
    else:
        submission = pd.DataFrame({target: predictions})

    path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(path, index=False)
    print(f"\n提交文件已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="KaggleML 快速测试框架")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"问题类型: {cfg['problem_type']}  评估指标: {cfg['eval_metric']}")
    print(f"模型列表: {cfg['models']}\n")

    # 1. 加载数据
    train_df, test_df, sample_sub = load_data(cfg)

    # 2. EDA（取消注释以启用）
    # summarize(train_df, cfg["target"])
    # plot_target(train_df, cfg["target"], cfg.get("output_dir", "results"))
    # plot_correlations(train_df, cfg.get("output_dir", "results"))

    # 3. 预处理
    X_train, y_train, X_test, feature_names = preprocess(train_df, test_df, cfg)

    # 4. 训练并对比模型
    print("\n模型交叉验证结果:")
    print("-" * 60)
    results_df, best_model = train_and_evaluate(X_train, y_train, cfg)

    print(f"\n{'='*60}")
    print("模型排行榜:")
    print(results_df.to_string(index=False))
    print(f"{'='*60}")

    # 5. 特征重要性
    output_dir = cfg.get("output_dir", "results")
    plot_feature_importance(best_model, feature_names, output_dir)

    # 6. 测试集预测
    predictions = predict_test(best_model, X_test, cfg)

    # 7. 保存提交文件
    save_submission(predictions, test_df, cfg, sample_sub)

    # 8. AutoGluon（可选）
    if "autogluon" in cfg.get("models", []):
        ag_preds = run_autogluon(train_df, test_df, cfg)
        if ag_preds is not None:
            ag_sub_path = os.path.join(output_dir, "submission_autogluon.csv")
            id_col = cfg.get("id_col")
            target = cfg["target"]
            if id_col and id_col in test_df.columns:
                ag_sub = pd.DataFrame({id_col: test_df[id_col], target: ag_preds})
            else:
                ag_sub = pd.DataFrame({target: ag_preds})
            ag_sub.to_csv(ag_sub_path, index=False)
            print(f"AutoGluon 提交文件已保存: {ag_sub_path}")

    # 保存模型对比结果
    results_path = os.path.join(output_dir, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"模型对比结果已保存: {results_path}")


if __name__ == "__main__":
    main()
