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
import yaml # Added for YAML configuration

# 导入AutoGluon相关模块
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common.features.feature_metadata import FeatureMetadata # Added for explicit feature types
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ==============================
# 配置加载
# ==============================
CONFIG_PATH = "config.yaml"

def load_config(config_path):
    """加载 YAML 配置文件"""
    print(f"加载配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    return config_data

config = load_config(CONFIG_PATH)

# 设置随机种子
SEED = config['SEED']
np.random.seed(SEED)
# 目标列名
LABEL = config['LABEL']
# 文件路径
TRAIN_DATA_PATH = config['TRAIN_DATA_PATH']
TEST_DATA_PATH = config['TEST_DATA_PATH']
# 样本提交文件路径
SAMPLE_SUBMISSION_PATH = config.get('SAMPLE_SUBMISSION_PATH') # Use .get() for optional keys
# 结果输出目录
OUTPUT_DIR = config['OUTPUT_DIR']
os.makedirs(OUTPUT_DIR, exist_ok=True)
# AutoGluon模型保存目录
MODELS_DIR = config['MODELS_DIR'] # Path will be relative as per config
os.makedirs(MODELS_DIR, exist_ok=True)

# AutoGluon settings from config
TIME_LIMIT = config['TIME_LIMIT']
EVAL_METRIC = config['EVAL_METRIC']
PRESETS = config['PRESETS'] 

# Prediction settings from config
PREDICTION_TYPE = config['PREDICTION_TYPE']

# ==============================
# Helper Function for Problem Type
# ==============================
def get_problem_type(eval_metric):
    """Determines problem type based on evaluation metric."""
    classification_metrics = ['roc_auc', 'accuracy', 'f1', 'precision', 'recall', 'log_loss', 'pac_score']
    regression_metrics = ['rmse', 'mse', 'mae', 'r2', 'root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error']
    
    if eval_metric.lower() in classification_metrics:
        return 'classification'
    elif eval_metric.lower() in regression_metrics:
        return 'regression'
    else:
        return 'other'

PROBLEM_TYPE = get_problem_type(EVAL_METRIC) # Determine problem type globally for use in multiple functions

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

    # Ensure sample_submission_path is not None and exists before trying to read
    if sample_submission_path and os.path.exists(sample_submission_path):
        sample_submission_df = pd.read_csv(sample_submission_path)
        print(
            f"训练集维度：{train_df.shape}，测试集维度：{test_df.shape}，样本提交文件维度：{sample_submission_df.shape}"
        )
    else:
        if sample_submission_path: # If path was provided but file doesn't exist
            print(f"警告: 样本提交文件 {sample_submission_path} 未找到。")
        print(
            f"训练集维度：{train_df.shape}，测试集维度：{test_df.shape}"
        )
        

    print("\n训练集前5行示例：")
    print(train_df.head())

    if sample_submission_df is not None:
        print("\n样本提交文件前5行示例：")
        print(sample_submission_df.head())

    return train_df, test_df, sample_submission_df


def basic_preprocessing(
    train_df, 
    test_df, 
    label_col,
    numerical_imputation_strategy="median",
    categorical_imputation_strategy="mode",
    categorical_fill_constant="missing"
):
    """
    对训练集和测试集进行基础预处理。
    主要包含：缺失值处理（支持不同策略）、布尔列转换等。

    参数：
    ----------
    train_df : pd.DataFrame
        训练集数据
    test_df : pd.DataFrame
        测试集数据
    label_col : str
        目标列的列名
    numerical_imputation_strategy : str, default 'median'
        数值列缺失值填充策略。可选 'median', 'mean'.
    categorical_imputation_strategy : str, default 'mode'
        类别列缺失值填充策略。可选 'mode', 'constant'.
    categorical_fill_constant : str, default 'missing'
        当 categorical_imputation_strategy 为 'constant' 时，用于填充的值。

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

    # 数值列：缺失值处理
    numeric_cols = train_df.select_dtypes(include=["int64", "float64"]).columns
    print(f"\n对数值列采用 '{numerical_imputation_strategy}' 策略填充缺失值...")
    for col in numeric_cols:
        if train_df[col].isnull().sum() > 0: # Check if column has missing values
            fill_value = None
            if numerical_imputation_strategy == "median":
                fill_value = train_df[col].median()
            elif numerical_imputation_strategy == "mean":
                fill_value = train_df[col].mean()
            else: # Default to median if strategy is unknown or not applicable
                fill_value = train_df[col].median()
                # Print warning only once for the first affected column with unknown strategy
                if numeric_cols.tolist().index(col) == 0 and numerical_imputation_strategy not in ["median", "mean"]:
                     print(f"警告: 未知的数值填充策略 '{numerical_imputation_strategy}' for column '{col}'. 将使用中位数填充。")
            
            train_df[col] = train_df[col].fillna(fill_value)
            if col in test_df.columns and test_df[col].isnull().sum() > 0:
                test_df[col] = test_df[col].fillna(fill_value) # Use same fill_value from train

    # 分类列：缺失值处理
    obj_cols = train_df.select_dtypes(include=["object", "category"]).columns
    print(f"\n对类别列采用 '{categorical_imputation_strategy}' 策略填充缺失值...")
    for col in obj_cols:
        if train_df[col].isnull().sum() > 0: # Check if column has missing values
            fill_value = None
            if categorical_imputation_strategy == "mode":
                mode_val = train_df[col].mode(dropna=True)
                if not mode_val.empty:
                    fill_value = mode_val[0]
                else: # If mode is empty (e.g., all NaN), use constant
                    fill_value = categorical_fill_constant
                    # Print warning only once for the first affected column
                    if obj_cols.tolist().index(col) == 0:
                        print(f"警告: 列 {col} 众数为空，将使用常量 '{fill_value}' 填充。")
            elif categorical_imputation_strategy == "constant":
                fill_value = categorical_fill_constant
            else: # Default to mode if strategy is unknown
                mode_val = train_df[col].mode(dropna=True)
                if not mode_val.empty:
                    fill_value = mode_val[0]
                else:
                    fill_value = categorical_fill_constant
                # Print warning only once for the first affected column with unknown strategy
                if obj_cols.tolist().index(col) == 0 and categorical_imputation_strategy not in ["mode", "constant"]:
                    print(f"警告: 未知的类别填充策略 '{categorical_imputation_strategy}' for column '{col}'. 将使用众数填充或常量（若众数为空）。")
            
            train_df[col] = train_df[col].fillna(fill_value)
            if col in test_df.columns and test_df[col].isnull().sum() > 0:
                 test_df[col] = test_df[col].fillna(fill_value) # Use same fill_value from train

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


def get_feature_metadata(df, label_col_name=None):
    """
    (示例函数) 根据列名后缀或手动指定来创建 FeatureMetadata 对象。
    AutoGluon 通常会自动推断特征类型，此函数仅为演示目的。
    用户可以扩展此函数，基于自己的数据特点和列名约定来定义更复杂的类型映射。
    """
    print("\n=== [可选] 特征元数据定义示例 ===")
    print("注意: AutoGluon 通常会自动推断特征类型。此部分仅为演示目的。")
    
    # 创建一个 FeatureMetadata 对象，可以从 DataFrame 推断初始类型
    # Exclude label column from metadata if provided
    df_for_metadata = df.copy()
    if label_col_name and label_col_name in df_for_metadata.columns:
        df_for_metadata = df_for_metadata.drop(columns=[label_col_name])
        
    feature_metadata = FeatureMetadata.from_df(df_for_metadata)
    
    # 示例：手动为特定列指定类型 (更直接的方式)
    # 假设我们有一些列，我们想确保它们的类型
    # 例如: 'column_ending_with_cat' should be 'category'
    # 'column_to_be_numeric' should be 'float'
    # 'id_column_to_ignore' could be 'text' if not dropped, or handled by AutoGluon
    
    # special_types_to_add = {}
    # for col in df_for_metadata.columns:
    #     if '_cat' in col.lower(): # Example: any column with '_cat' in its name
    #         special_types_to_add[col] = ['category']
    #     elif '_num' in col.lower(): # Example: any column with '_num' in its name
    #         special_types_to_add[col] = ['float']
    #     # Add more rules as needed based on your column naming conventions

    # if 'specific_column_name_a' in df_for_metadata.columns:
    #    special_types_to_add['specific_column_name_a'] = ['text'] 
    # if 'specific_column_name_b' in df_for_metadata.columns:
    #    special_types_to_add['specific_column_name_b'] = ['int']

    # if special_types_to_add:
    #    print(f"应用手动定义的特殊类型: {special_types_to_add}")
    #    feature_metadata = feature_metadata.add_special_types(special_types_to_add)
    # else:
    #    print("没有基于示例规则（如后缀 '_cat', '_num'）找到要手动类型指定的列。")

    # 返回 None 或空的 FeatureMetadata 如果不想覆盖 AutoGluon 的自动推断
    # 在这个演示中，我们不激活它，所以返回 None
    print("在此示例配置中，将返回 None，让 AutoGluon 自动推断特征类型。")
    print("要激活手动特征类型，请取消注释此函数中的相关行，配置规则，并确保它返回一个配置好的 FeatureMetadata 对象。")
    return None # Default behavior: do not override AutoGluon's type inference.
    
    # print(f"最终定义的特征元数据 (部分): {feature_metadata.get_type_map_raw()}")
    # return feature_metadata # Uncomment to return the configured metadata


def train_model(
    train_df, label_col, time_limit=300, eval_metric="roc_auc", presets="best_quality", feature_metadata=None
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
    presets : str
        AutoGluon的预设，例如 'best_quality', 'medium_quality'

    返回：
    ----------
    predictor : TabularPredictor
        训练完成的AutoGluon模型
    """
    print("\n=== [3] 模型训练 ===")

    # 转成AutoGluon专用的TabularDataset
    print(f"\n训练模型时使用的特征元数据: {feature_metadata}") # Log what metadata is being used
    if feature_metadata:
        # Ensure label column is not in feature_metadata's type map if it was accidentally included
        # This is more of a safeguard; FeatureMetadata.from_df usually handles this if label is in df.
        # However, if metadata is built completely manually, ensure it only contains features.
        current_type_map = feature_metadata.get_type_map_raw()
        if label_col in current_type_map:
            print(f"警告: 目标列 '{label_col}' 存在于特征元数据中。AutoGluon 通常期望元数据仅包含特征。")
            # It's generally safer for TabularDataset to receive metadata only for features.
            # However, AutoGluon might handle it. For robustness, one might consider removing it here.
            
        print("使用提供的特征元数据初始化 TabularDataset。")
        # It's important that train_df for TabularDataset contains the label column.
        # FeatureMetadata should describe the types of the *features*.
        ag_train = TabularDataset(train_df, feature_metadata=feature_metadata)
    else:
        print("未提供特征元数据，TabularDataset 将自动推断类型。")
        ag_train = TabularDataset(train_df) # Default: AutoGluon infers types


    # 拆分出验证集（简单方式：train_test_split）
    # SEED is used here from global scope
    # Determine stratification based on problem type
    print(f"Task type based on EVAL_METRIC ('{EVAL_METRIC}'): {PROBLEM_TYPE.capitalize()}")
    stratify_column = None
    if PROBLEM_TYPE == 'classification':
        if label_col in ag_train.columns:
            stratify_column = ag_train[label_col]
            print(f"Using stratification for column: {label_col}")
        else:
            print(f"Warning: Label column '{label_col}' not found in training data for stratification.")
    else:
        print("Stratification is not used for regression tasks.")

    train_data, val_data = train_test_split(
        ag_train,
        test_size=0.2,
        random_state=SEED, 
        stratify=stratify_column,
    )
    print(
        f"训练集大小: {train_data.shape}, 验证集大小: {val_data.shape}"
    )

    # 初始化Predictor
    # AutoGluon's TabularPredictor will also infer problem_type, 
    # but we use our PROBLEM_TYPE for explicit control elsewhere.
    # MODELS_DIR is used here from global scope
    predictor = TabularPredictor(
        label=label_col,
        eval_metric=eval_metric,
        path=MODELS_DIR, 
        verbosity=2,
    ).fit(
        train_data=train_data,
        time_limit=time_limit,
        presets=presets, # Use presets from config
    )

    print("\n模型训练完成！")
    print("=== 验证集指标 ===")
    predictor.evaluate(val_data)

    print("\n=== 模型排行榜 ===")
    lb = predictor.leaderboard(val_data, silent=True)
    print(lb)
    # Save leaderboard
    leaderboard_path = os.path.join(OUTPUT_DIR, "leaderboard.csv")
    lb.to_csv(leaderboard_path)
    print(f"Leaderboard saved to {leaderboard_path}")

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
        # OUTPUT_DIR is used here from global scope
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
    prediction_type="both", # This comes from config, will be interpreted based on PROBLEM_TYPE
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
    prediction_type : str
        预测类型 (来自配置), 对于分类: 'class', 'prob', 'both'. 对于回归, outputs numerical predictions.
    """
    print("\n=== [4] 测试集预测 ===")
    # PROBLEM_TYPE is determined globally based on EVAL_METRIC from config
    print(f"Preparing predictions for {PROBLEM_TYPE} task.")

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
    y_pred = None
    y_pred_proba = None
    prob_col_name = f"{label_col}_probability" # Define consistent probability column name

    if PROBLEM_TYPE == 'regression':
        print("For regression tasks, predicting numerical values.")
        y_pred = predictor.predict(ag_test)
        # PREDICTION_TYPE from config is effectively ignored for regression in terms of 'class' or 'prob'
        # Output will always be the numerical prediction.
    
    elif PROBLEM_TYPE == 'classification':
        if prediction_type in ["class", "both"]:
            y_pred = predictor.predict(ag_test)
        
        if prediction_type in ["prob", "both"]:
            try:
                y_pred_proba_all = predictor.predict_proba(ag_test)
                # Determine positive class probability column
                # Assuming binary classification for simplicity here, 
                # AutoGluon often names columns 0 and 1 for probabilities.
                # If predictor.class_labels has ['A', 'B'], then predict_proba might have columns 'A', 'B'.
                # We need to robustly find the probability of the "positive" class.
                # For now, let's assume positive class is typically the last column or '1' or True.
                if 1 in y_pred_proba_all.columns: # Common for binary: 0, 1
                    y_pred_proba = y_pred_proba_all[1]
                elif True in y_pred_proba_all.columns: # Common for boolean target: False, True
                     y_pred_proba = y_pred_proba_all[True]
                elif len(predictor.class_labels) == 2: # Binary classification, take prob of the second label
                    # This assumes predictor.class_labels is ordered [negative_class, positive_class]
                    # or that the user is interested in the probability of the class at index 1.
                    # For datasets like Adult Income ('>50K', '<=50K'), this might need careful checking
                    # of predictor.class_labels order.
                    positive_class_label = predictor.class_labels[-1] # A common convention
                    if positive_class_label in y_pred_proba_all.columns:
                         y_pred_proba = y_pred_proba_all[positive_class_label]
                    else: # Fallback if specific positive class label not found, take last column
                        print(f"Warning: Could not determine positive class probability column explicitly. Using last column of predict_proba output: {y_pred_proba_all.columns[-1]}")
                        y_pred_proba = y_pred_proba_all.iloc[:, -1]
                elif not y_pred_proba_all.empty: # Fallback for multi-class or unclear binary
                     print(f"Warning: Using last column of predict_proba output as probability: {y_pred_proba_all.columns[-1]}")
                     y_pred_proba = y_pred_proba_all.iloc[:, -1] # Take the last column as a guess
                else:
                    print("Warning: predict_proba output is empty or structure not recognized for probability extraction.")
                    y_pred_proba = None

            except Exception as e:
                print(f"无法获取概率预测: {e}")
                if prediction_type == "prob":
                    print("警告: 概率预测失败。")
                y_pred_proba = None
    else: # PROBLEM_TYPE == 'other'
        print(f"预测逻辑未针对 '{PROBLEM_TYPE}' 任务类型进行特定调整。将尝试默认预测。")
        y_pred = predictor.predict(ag_test)


    # 构建提交结果
    submission = pd.DataFrame()
    if id_col not in test_df:
         test_df[id_col] = test_df.index # Ensure id_col exists

    if sample_submission_df is not None:
        submission = sample_submission_df.copy()
        # Ensure the ID column from sample_submission is used
        if id_col not in submission.columns and id_col in test_df.columns:
             # If sample_submission has a different ID name, this needs manual adjustment.
             # For now, assume test_df[id_col] can be directly used if submission[id_col] is missing.
             submission[id_col] = test_df[id_col].values


        if PROBLEM_TYPE == 'regression':
            if y_pred is not None:
                submission[label_col] = y_pred.values
            else:
                print("警告: 回归预测未生成。")
        
        elif PROBLEM_TYPE == 'classification':
            if prediction_type == "class":
                if y_pred is not None:
                    submission[label_col] = y_pred.values
                else:
                    print("警告: 类别预测未生成。")
            elif prediction_type == "prob":
                if y_pred_proba is not None:
                    submission[label_col] = y_pred_proba.values # Main target col gets probability
                else:
                    print("警告: 概率预测未生成。")
            elif prediction_type == "both":
                if y_pred is not None:
                    submission[label_col] = y_pred.values
                else:
                    print("警告: 类别预测未生成。")
                if y_pred_proba is not None:
                    submission[prob_col_name] = y_pred_proba.values
                else:
                    print(f"警告: 概率预测 ({prob_col_name}) 未生成。")
    
    else: # Create new submission file
        submission_data = {id_col: test_df[id_col].values}
        if PROBLEM_TYPE == 'regression':
            if y_pred is not None:
                submission_data[label_col] = y_pred.values
            else:
                print("警告: 回归预测未生成。")
        
        elif PROBLEM_TYPE == 'classification':
            if prediction_type == "class":
                if y_pred is not None:
                    submission_data[label_col] = y_pred.values
            elif prediction_type == "prob":
                if y_pred_proba is not None:
                    # If sample_submission_df is None and type is 'prob', 
                    # the main column in submission_data should be the probability, named {label_col}_probability.
                    # If sample_submission_df is NOT None, it's assumed label_col in that df gets the prob.
                    submission_data[prob_col_name] = y_pred_proba.values # Standardized name
                else:
                    print(f"警告: 概率预测 ({prob_col_name}) 未生成。")
            elif prediction_type == "both":
                if y_pred is not None:
                    submission_data[label_col] = y_pred.values
                if y_pred_proba is not None:
                    submission_data[prob_col_name] = y_pred_proba.values # Standardized name
                else:
                    print(f"警告: 概率预测 ({prob_col_name}) 未生成 (当 type='both')。")
        
        submission = pd.DataFrame(submission_data)

    return submission


def save_submission(submission, base_prediction_type):
    # 保存结果
    # OUTPUT_DIR and LABEL are used here from global scope
    if submission.empty:
        print("提交文件为空，不保存。")
        return None

    # Adjust filename based on problem type and actual content
    output_content_description = base_prediction_type # Default
    prob_col_name_check = f"{LABEL}_probability" # Get LABEL from global config for check

    if PROBLEM_TYPE == 'regression':
        output_content_description = "regression_predictions"
    elif PROBLEM_TYPE == 'classification':
        is_class_present = LABEL in submission.columns
        is_prob_present = prob_col_name_check in submission.columns

        if base_prediction_type == "both":
            if is_class_present and is_prob_present:
                output_content_description = "full_classification" # class + prob
            elif is_class_present:
                output_content_description = "class_only_classification"
            elif is_prob_present:
                 output_content_description = "prob_only_classification" # Should ideally not happen if 'both' selected and class failed
            else:
                output_content_description = "empty_classification_output"
        elif base_prediction_type == "prob":
            output_content_description = "prob_only_classification"
        elif base_prediction_type == "class":
            output_content_description = "class_only_classification"
    else: # PROBLEM_TYPE == 'other'
        output_content_description = "unknown_type_predictions"


    submission_path = os.path.join(
        OUTPUT_DIR, f"submission_{output_content_description}.csv"
    )
    submission.to_csv(submission_path, index=False)
    print(f"测试集预测结果已保存到: {submission_path}")
    print("\n=== 部分预测结果预览 ===")
    print(submission.head(10))

    return submission_path # Return path for consistency


# ==============================
# 主程序，不用入口，直接执行，方便调试
# =============================

if __name__ == "__main__":
    # 1. 数据加载
    # TRAIN_DATA_PATH, TEST_DATA_PATH, SAMPLE_SUBMISSION_PATH, LABEL are from global config
    train_df, test_df, sample_submission_df = load_data(
        TRAIN_DATA_PATH, TEST_DATA_PATH, SAMPLE_SUBMISSION_PATH
    )

    # 2. 基础预处理
    # 示例：使用不同的填充策略 (默认情况下这些行是注释掉的)
    # print("使用自定义预处理策略示例：")
    # train_df, test_df = basic_preprocessing(
    #     train_df.copy(), test_df.copy(), LABEL, # Use .copy() to be safe
    #     numerical_imputation_strategy="mean",
    #     categorical_imputation_strategy="constant",
    #     categorical_fill_constant="UNKNOWN_CAT_VALUE" 
    # )
    print("使用默认预处理策略。")
    train_df, test_df = basic_preprocessing(train_df.copy(), test_df.copy(), LABEL)

    # 可选: 获取特征元数据 (默认不激活，因此 feature_meta 将为 None)
    # 要激活，请取消注释下一行，并修改 get_feature_metadata 以返回配置好的对象。
    # feature_meta = get_feature_metadata(train_df, label_col_name=LABEL)


    # 3. 模型训练
    # TIME_LIMIT, EVAL_METRIC, PRESETS are from global config
    predictor = train_model(
        train_df,
        LABEL,
        time_limit=TIME_LIMIT,
        eval_metric=EVAL_METRIC,
        presets=PRESETS,
        feature_metadata=None # By default, pass None. Change to feature_meta to activate.
        # feature_metadata=feature_meta # Uncomment to pass the generated metadata
    )

    # 4 预测
    # PREDICTION_TYPE is from global config (config['PREDICTION_TYPE'])
    # PROBLEM_TYPE is determined globally
    submission_df = predict(
        predictor,
        test_df.copy(), # Pass a copy to avoid modification issues if id_col is added
        LABEL, # Global LABEL from config
        sample_submission_df,
        prediction_type=PREDICTION_TYPE, # Global PREDICTION_TYPE from config
    )

    # 保存预测结果
    if submission_df is not None and not submission_df.empty:
        save_submission(submission_df, PREDICTION_TYPE) # Pass original PREDICTION_TYPE for context
    else:
        print("没有生成提交文件或提交文件为空。")
