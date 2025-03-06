# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2025-03-03 14:07
# @Last Modified by:   Muqy
# @Last Modified time: 2025-03-06 11:01


import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt

# 导入AutoGluon相关模块
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ==============================
# 全局配置
# ==============================
# 设置随机种子
SEED = 0
np.random.seed(SEED)
# 目标列名
LABEL = "income"
# 文件路径
TRAIN_DATA_PATH = "data/train_data.csv"
TEST_DATA_PATH = "data/test_data.csv"
# 样本提交文件路径
SAMPLE_SUBMISSION_PATH = "sample_submission.csv"
# 结果输出目录
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# AutoGluon模型保存目录
# （AutoGluon的intermediate path如果设置C盘可能会有权限错误，建议设置在其他盘）
MODELS_DIR = "H:/autogluon_models_adult_income"
os.makedirs(MODELS_DIR, exist_ok=True)

# ==============================
# 函数定义
# ==============================


def load_data(train_path, test_path, sample_submission_path=None):
    """
    加载训练集、测试集和样本提交文件数据，并返回DataFrame格式。

    参数：
    ----------
    train_path : str
        训练集文件的路径
    test_path : str
        测试集文件的路径
    sample_submission_path : str, 可选
        样本提交文件的路径

    返回：
    ----------
    train_df : pd.DataFrame
        训练集数据
    test_df : pd.DataFrame
        测试集数据
    sample_submission_df : pd.DataFrame 或 None
        样本提交文件数据
    """
    print("=== [1] 数据加载 ===")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_submission_df = None

    if sample_submission_path and os.path.exists(
        sample_submission_path
    ):
        sample_submission_df = pd.read_csv(sample_submission_path)
        print(
            f"训练集维度：{train_df.shape}，测试集维度：{test_df.shape}，样本提交文件维度：{sample_submission_df.shape}"
        )
    else:
        print(
            f"训练集维度：{train_df.shape}，测试集维度：{test_df.shape}"
        )

    print("\n训练集前5行示例：")
    print(train_df.head())

    if sample_submission_df is not None:
        print("\n样本提交文件前5行示例：")
        print(sample_submission_df.head())

    return train_df, test_df, sample_submission_df


def basic_preprocessing(train_df, test_df, label_col):
    """
    对训练集和测试集进行基础预处理，不涉及高级特征工程。
    主要包含：缺失值处理、布尔列转换等。

    参数：
    ----------
    train_df : pd.DataFrame
        训练集数据
    test_df : pd.DataFrame
        测试集数据
    label_col : str
        目标列的列名

    返回：
    ----------
    train_df : pd.DataFrame
        预处理后的训练集
    test_df : pd.DataFrame
        预处理后的测试集
    """
    print("\n=== [2] 数据预处理 ===")

    # ========== 2.1 缺失值处理 ==========
    print("\n检查训练集缺失值：")
    missing_train = train_df.isnull().sum()
    print(missing_train[missing_train > 0])

    print("\n检查测试集缺失值：")
    missing_test = test_df.isnull().sum()
    print(missing_test[missing_test > 0])

    # 数值列：缺失值用中位数填充
    numeric_cols = train_df.select_dtypes(
        include=["int64", "float64"]
    ).columns
    for col in numeric_cols:
        median_val = train_df[col].median()  # 以训练集的中位数为准
        if train_df[col].isnull().sum() > 0:
            train_df[col] = train_df[col].fillna(median_val)
        if col in test_df.columns and test_df[col].isnull().sum() > 0:
            test_df[col] = test_df[col].fillna(median_val)

    # 分类列：缺失值用众数填充
    obj_cols = train_df.select_dtypes(
        include=["object", "category"]
    ).columns
    for col in obj_cols:
        mode_val = train_df[col].mode(dropna=True)
        if not mode_val.empty:  # 众数可能有多个，默认取第一个
            mode_val = mode_val[0]
            if train_df[col].isnull().sum() > 0:
                train_df[col] = train_df[col].fillna(mode_val)
            if (
                col in test_df.columns
                and test_df[col].isnull().sum() > 0
            ):
                test_df[col] = test_df[col].fillna(mode_val)

    # ========== 2.2 布尔列转换（可选） ==========
    bool_map = {
        "Yes": True,
        "No": False,
        "yes": True,
        "no": False,
        "TRUE": True,
        "FALSE": False,
        "true": True,
        "false": False,
    }

    # 如果某些列明显是Yes/No，但未必叫这些名字，可手动指定
    # 这里只是示例扫描所有object列，尝试映射
    for col in obj_cols:
        unique_vals = train_df[col].dropna().unique()
        # 如果大部分值均在bool_map之内，尝试做布尔转换
        if all(str(val).lower() in bool_map for val in unique_vals):
            print(f"自动将列 {col} 转为布尔型...")
            train_df[col] = train_df[col].map(bool_map)
            if col in test_df.columns:
                test_df[col] = test_df[col].map(bool_map)

    print("基础预处理完成！")
    return train_df, test_df


def train_model(
    train_df, label_col, time_limit=300, eval_metric="roc_auc"
):
    """
    使用AutoGluon对训练集进行模型训练。

    参数：
    ----------
    train_df : pd.DataFrame
        训练集数据（已完成预处理）
    label_col : str
        目标列名
    time_limit : int
        训练时长限制（秒），可根据数据量调大或调小
    eval_metric : str
        评估指标，可选'accuracy', 'roc_auc', 'f1'等

    返回：
    ----------
    predictor : TabularPredictor
        训练完成的AutoGluon模型
    """
    print("\n=== [3] 模型训练 ===")

    # 转成AutoGluon专用的TabularDataset
    ag_train = TabularDataset(train_df)

    # 拆分出验证集（简单方式：train_test_split）
    train_data, val_data = train_test_split(
        ag_train,
        test_size=0.2,
        random_state=SEED,
        stratify=ag_train[label_col],  # 分层抽样，保证目标分布一致
    )
    print(
        f"训练集大小: {train_data.shape}, 验证集大小: {val_data.shape}"
    )

    # 初始化Predictor
    predictor = TabularPredictor(
        label=label_col,
        eval_metric=eval_metric,
        path=MODELS_DIR,
        verbosity=2,
    ).fit(
        train_data=train_data,
        time_limit=time_limit,
        presets="best_quality",
    )

    print("\n模型训练完成！")
    print("=== 验证集指标 ===")
    predictor.evaluate(val_data)

    print("\n=== 模型排行榜 ===")
    lb = predictor.leaderboard(val_data, silent=True)
    print(lb)

    # 显示部分特征重要性
    try:
        fi = predictor.feature_importance(val_data)
        print("\n=== 特征重要性（前10） ===")
        print(fi.head(10))

        # 可视化特征重要性
        plt.figure(figsize=(10, 8))
        fi.head(10).plot(kind="barh")
        plt.title("Feature Importance (Top 10)")
        plt.tight_layout()
        fi_fig_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
        plt.savefig(fi_fig_path)
        plt.close()
        print(f"特征重要性图已保存至: {fi_fig_path}")
    except Exception as e:
        print(f"特征重要性计算失败，可能因为模型不支持：{e}")

    return predictor


def predict(
    predictor,
    test_df,
    label_col,
    sample_submission_df=None,
    prediction_type="both",
):
    """
    使用训练好的模型对测试集进行预测，并将结果保存到 CSV。

    参数：
    ----------
    predictor : TabularPredictor
        训练完成的AutoGluon模型
    test_df : pd.DataFrame
        测试集数据（已完成预处理）
    label_col : str
        目标列名
    sample_submission_df : pd.DataFrame, 可选
        样本提交文件，若提供则基于此模板填充预测结果
    prediction_type : str, 可选
        预测类型，可选 'class'（仅预测类别）, 'prob'（仅预测概率）, 'both'（预测类别和概率）
    """
    print("\n=== [4] 测试集预测 ===")

    # 确定ID列
    id_col = None
    if sample_submission_df is not None:
        # 从样本提交文件获取ID列名
        possible_id_cols = [
            c
            for c in sample_submission_df.columns
            if c.lower() in ["id", "index"]
        ]
        if possible_id_cols:
            id_col = possible_id_cols[0]

    if id_col is None:
        # 从测试集获取ID列名
        possible_id_cols = [
            c for c in test_df.columns if c.lower() in ["id", "index"]
        ]
        if possible_id_cols:
            id_col = possible_id_cols[0]
        else:
            # 如果没有ID列，使用行索引
            id_col = "id"
            test_df[id_col] = test_df.index

    # 准备预测
    ag_test = TabularDataset(test_df)

    # 根据预测类型准备结果
    if prediction_type in ["class", "both"]:
        # 获取类别预测
        y_pred = predictor.predict(ag_test)

    if prediction_type in ["prob", "both"]:
        # 获取概率预测
        try:
            y_pred_proba = predictor.predict_proba(ag_test)

            # 自动确定正类对应的列名/索引
            if 1 in y_pred_proba.columns:
                prob_col = 1
            elif True in y_pred_proba.columns:
                prob_col = True
            else:
                prob_col = y_pred_proba.columns[-1]
        except Exception as e:
            print(f"无法获取概率预测: {e}")
            if prediction_type == "prob":
                prediction_type = "class"
                print("自动切换为仅类别预测")
            y_pred_proba = None

    # 构建提交结果
    if sample_submission_df is not None:
        # 基于样本提交文件模板
        submission = sample_submission_df.copy()

        # 根据预测类型更新结果
        if prediction_type == "class":
            submission[label_col] = y_pred.values
        elif prediction_type == "prob" and y_pred_proba is not None:
            submission[label_col] = y_pred_proba[prob_col].values
        elif prediction_type == "both" and y_pred_proba is not None:
            submission[label_col] = y_pred.values
            prob_col_name = f"{label_col}_probability"
            submission[prob_col_name] = y_pred_proba[prob_col].values
    else:
        # 创建新的提交文件
        if prediction_type == "class":
            submission = pd.DataFrame(
                {
                    id_col: test_df[id_col].values,
                    label_col: y_pred.values,
                }
            )
        elif prediction_type == "prob" and y_pred_proba is not None:
            submission = pd.DataFrame(
                {
                    id_col: test_df[id_col].values,
                    label_col: y_pred_proba[prob_col].values,
                }
            )
        elif prediction_type == "both" and y_pred_proba is not None:
            submission = pd.DataFrame(
                {
                    id_col: test_df[id_col].values,
                    label_col: y_pred.values,
                    f"{label_col}_probability": y_pred_proba[
                        prob_col
                    ].values,
                }
            )

    return submission


def save_submission(submission, prediction_type):
    # 保存结果
    prediction_type_str = prediction_type.replace("both", "full")
    submission_path = os.path.join(
        OUTPUT_DIR, f"submission_{prediction_type_str}.csv"
    )
    submission.to_csv(submission_path, index=False)
    print(f"测试集预测结果已保存到: {submission_path}")
    print("\n=== 部分预测结果预览 ===")
    print(submission.head(10))

    return submission


# ==============================
# 主程序，不用入口，直接执行，方便调试
# =============================

if __name__ == "__main__":
    # 1. 数据加载
    train_df, test_df, sample_submission_df = load_data(
        TRAIN_DATA_PATH, TEST_DATA_PATH, SAMPLE_SUBMISSION_PATH
    )

    # 2. 基础预处理
    train_df, test_df = basic_preprocessing(train_df, test_df, LABEL)

    # 3. 模型训练
    predictor = train_model(
        train_df,
        LABEL,
        time_limit=2000,  # 训练时间(秒)，根据数据规模与需求可调整
        eval_metric="roc_auc",
    )

    # 4 仅预测类别
    submission = predict(
        predictor,
        test_df,
        LABEL,
        sample_submission_df,
        prediction_type="class",
    )

    # 保存预测结果
    submission = save_submission(submission, "class")
