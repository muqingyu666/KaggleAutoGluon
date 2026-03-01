# -*- coding: utf-8 -*-
"""模型注册、训练与评估模块。

通过 REGISTRY 字典管理所有可用模型。添加新模型只需在 REGISTRY 中增加一行。
支持交叉验证评估、模型对比、测试集预测。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, log_loss,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

# ---------- 评估指标 ----------
METRICS = {
    # 分类
    "roc_auc": roc_auc_score,
    "accuracy": accuracy_score,
    "f1": f1_score,
    "log_loss": log_loss,
    # 回归
    "rmse": lambda y, p: np.sqrt(mean_squared_error(y, p)),
    "mae": mean_absolute_error,
    "r2": r2_score,
}

# 这些指标分数越小越好
_LOWER_IS_BETTER = {"rmse", "mae", "log_loss"}

# ---------- 模型注册表 ----------
# 格式: name -> { problem_type: (ModelClass, default_params) }
# 添加新模型只需在这里加一行
REGISTRY = {
    "random_forest": {
        "classification": (RandomForestClassifier, {"n_estimators": 200, "n_jobs": -1, "random_state": 42}),
        "regression":     (RandomForestRegressor,  {"n_estimators": 200, "n_jobs": -1, "random_state": 42}),
    },
    "logistic_regression": {
        "classification": (LogisticRegression, {"max_iter": 1000, "random_state": 42}),
    },
    "linear_regression": {
        "regression": (LinearRegression, {}),
    },
}

# 注册 XGBoost（可选依赖）
try:
    from xgboost import XGBClassifier, XGBRegressor
    REGISTRY["xgboost"] = {
        "classification": (XGBClassifier, {
            "n_estimators": 200, "learning_rate": 0.1, "max_depth": 6,
            "use_label_encoder": False, "eval_metric": "logloss",
            "random_state": 42, "n_jobs": -1,
        }),
        "regression": (XGBRegressor, {
            "n_estimators": 200, "learning_rate": 0.1, "max_depth": 6,
            "random_state": 42, "n_jobs": -1,
        }),
    }
except ImportError:
    pass

# 注册 LightGBM（可选依赖）
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    REGISTRY["lightgbm"] = {
        "classification": (LGBMClassifier, {
            "n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31,
            "random_state": 42, "n_jobs": -1, "verbose": -1,
        }),
        "regression": (LGBMRegressor, {
            "n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31,
            "random_state": 42, "n_jobs": -1, "verbose": -1,
        }),
    }
except ImportError:
    pass


def get_model(name, problem_type):
    """根据名称和问题类型获取模型实例。

    Args:
        name: 模型名称（如 "random_forest"）
        problem_type: "classification" 或 "regression"

    Returns:
        模型实例，若该模型不支持当前问题类型则返回 None
    """
    entry = REGISTRY.get(name)
    if entry is None:
        print(f"[警告] 未知模型: {name}，跳过")
        return None
    if problem_type not in entry:
        print(f"[跳过] {name} 不支持 {problem_type}")
        return None
    cls, params = entry[problem_type]
    return cls(**params)


def cross_validate(model, X, y, cfg):
    """交叉验证评估模型。

    分类任务用 StratifiedKFold，回归任务用 KFold。

    Args:
        model: sklearn 兼容的模型实例
        X: 特征 DataFrame
        y: 目标 Series
        cfg: 配置字典

    Returns:
        fold 分数列表
    """
    problem_type = cfg["problem_type"]
    metric_name = cfg["eval_metric"]
    metric_fn = METRICS[metric_name]
    n_splits = cfg["cv"]["n_splits"]
    seed = cfg["cv"]["seed"]

    if problem_type == "classification":
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores = []
    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)

        # 需要概率的指标用 predict_proba
        if metric_name in ("roc_auc", "log_loss") and hasattr(model, "predict_proba"):
            y_pred = model.predict_proba(X_val)[:, 1]
        else:
            y_pred = model.predict(X_val)

        scores.append(metric_fn(y_val, y_pred))

    return scores


def train_and_evaluate(X_train, y_train, cfg):
    """训练并评估配置中指定的所有模型。

    打印模型对比表格，返回结果 DataFrame 和最佳模型（在全量数据上 refit）。

    Args:
        X_train: 特征 DataFrame
        y_train: 目标 Series
        cfg: 配置字典

    Returns:
        results: 包含 model, mean_score, std_score 的 DataFrame
        best_model: 在全量数据上 refit 的最佳模型实例
    """
    model_names = cfg["models"]
    problem_type = cfg["problem_type"]
    metric_name = cfg["eval_metric"]
    lower_better = metric_name in _LOWER_IS_BETTER

    results = []
    all_scores = {}

    for name in model_names:
        # AutoGluon 单独处理
        if name == "autogluon":
            continue

        model = get_model(name, problem_type)
        if model is None:
            continue

        scores = cross_validate(model, X_train, y_train, cfg)
        mean_s, std_s = np.mean(scores), np.std(scores)
        results.append({"model": name, "mean_score": mean_s, "std_score": std_s})
        all_scores[name] = mean_s
        print(f"  {name:<25s}  {metric_name}={mean_s:.5f} (+/- {std_s:.5f})")

    if not results:
        raise ValueError("没有有效的模型结果，请检查 config.yaml 中的 models 列表")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mean_score", ascending=lower_better).reset_index(drop=True)

    # 选出最佳模型并在全量数据上 refit
    best_name = results_df.iloc[0]["model"]
    print(f"\n最佳模型: {best_name}")

    best_model = get_model(best_name, problem_type)
    best_model.fit(X_train, y_train)

    return results_df, best_model


def predict_test(model, X_test, cfg):
    """用训练好的模型对测试集做预测。

    分类任务同时返回类别和概率（如果模型支持 predict_proba）。

    Args:
        model: 已训练的模型
        X_test: 测试集特征
        cfg: 配置字典

    Returns:
        predictions: 预测结果（分类为概率，回归为数值）
    """
    if cfg["problem_type"] == "classification" and hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return model.predict(X_test)


def run_autogluon(train_df, test_df, cfg):
    """运行 AutoGluon（可选）。需要单独安装: pip install autogluon

    Args:
        train_df: 原始训练数据（未经 preprocess 的 DataFrame）
        test_df: 原始测试数据
        cfg: 配置字典

    Returns:
        predictions: 预测结果
    """
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print("[错误] AutoGluon 未安装。请运行: pip install autogluon")
        return None

    ag_cfg = cfg.get("autogluon", {})
    time_limit = ag_cfg.get("time_limit", 600)
    presets = ag_cfg.get("presets", "medium_quality")
    target = cfg["target"]

    print(f"\n运行 AutoGluon (time_limit={time_limit}s, presets={presets})...")
    predictor = TabularPredictor(label=target, eval_metric=cfg["eval_metric"])
    predictor.fit(train_df, time_limit=time_limit, presets=presets)

    leaderboard = predictor.leaderboard(silent=True)
    print(leaderboard.to_string())

    id_col = cfg.get("id_col")
    test_features = test_df.drop(columns=[c for c in [id_col] if c and c in test_df.columns])
    predictions = predictor.predict(test_features)
    return predictions
