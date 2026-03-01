# KaggleML

通用机器学习竞赛快速测试框架。一条命令对比多个模型效果，开箱即用。

## 支持的模型

| 模型 | 分类 | 回归 | 依赖 |
|------|:----:|:----:|------|
| Random Forest | Y | Y | scikit-learn |
| XGBoost | Y | Y | xgboost |
| LightGBM | Y | Y | lightgbm |
| Logistic Regression | Y | - | scikit-learn |
| Linear Regression | - | Y | scikit-learn |
| AutoGluon | Y | Y | autogluon (可选) |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 修改 config.yaml（数据路径、目标列、问题类型等）

# 3. 运行
python run.py
```

输出示例：
```
问题类型: classification  评估指标: roc_auc
模型列表: ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression']

训练集: (10, 6)  测试集: (5, 5)
特征数: 4 列

模型交叉验证结果:
------------------------------------------------------------
  random_forest              roc_auc=0.85000 (+/- 0.12345)
  xgboost                    roc_auc=0.87000 (+/- 0.10234)
  lightgbm                   roc_auc=0.86000 (+/- 0.11456)
  logistic_regression        roc_auc=0.82000 (+/- 0.13567)

============================================================
模型排行榜:
       model  mean_score  std_score
     xgboost     0.87000    0.10234
    lightgbm     0.86000    0.11456
random_forest    0.85000    0.12345
logistic_regression  0.82000  0.13567
============================================================

最佳模型: xgboost
提交文件已保存: results/submission.csv
```

## 项目结构

```
├── run.py              # 入口脚本：加载配置 → 预处理 → 训练对比 → 预测 → 保存
├── models.py           # 模型注册表、交叉验证、训练评估
├── preprocess.py       # 数据加载、特征预处理
├── eda.py              # EDA 工具（数据概览、分布图、相关性、特征重要性）
├── config.yaml         # 实验配置
├── requirements.txt    # 依赖
├── data/               # 数据目录
│   ├── train_data.csv
│   └── test_data.csv
└── sample_submission.csv
```

## 配置说明

编辑 `config.yaml` 即可适配不同比赛：

```yaml
train_path: "data/train_data.csv"    # 训练集路径
test_path: "data/test_data.csv"      # 测试集路径
target: "income"                      # 目标列名
id_col: "id"                          # ID列名，无则设为 null
problem_type: "classification"        # classification 或 regression
eval_metric: "roc_auc"                # 评估指标

models:                               # 要对比的模型
  - random_forest
  - xgboost
  - lightgbm
  - logistic_regression

cv:
  n_splits: 5                         # 交叉验证折数
  seed: 42                            # 随机种子
```

## 如何添加新模型

在 `models.py` 的 `REGISTRY` 字典中加一行：

```python
# 例如添加 CatBoost
from catboost import CatBoostClassifier, CatBoostRegressor

REGISTRY["catboost"] = {
    "classification": (CatBoostClassifier, {"iterations": 200, "verbose": 0}),
    "regression":     (CatBoostRegressor,  {"iterations": 200, "verbose": 0}),
}
```

然后在 `config.yaml` 的 `models` 列表中加上 `catboost` 即可。

## 如何使用 EDA

在 `run.py` 中取消 EDA 相关行的注释：

```python
# 2. EDA（取消注释以启用）
summarize(train_df, cfg["target"])
plot_target(train_df, cfg["target"], cfg.get("output_dir", "results"))
plot_correlations(train_df, cfg.get("output_dir", "results"))
```

也可以自行在 `eda.py` 中添加自定义分析函数。

## 使用 AutoGluon（可选）

AutoGluon 依赖较多，默认不包含在 `requirements.txt` 中。如需使用：

```bash
pip install autogluon

# 在 config.yaml 中启用：
# models:
#   - autogluon
# autogluon:
#   time_limit: 600
#   presets: "medium_quality"
```

## 自定义预处理

编辑 `preprocess.py` 中的 `preprocess()` 函数。当前默认处理：
- 类别特征 → LabelEncoder
- 数值缺失值 → 中位数填充

可根据比赛需求修改为 OneHotEncoder、TargetEncoder、特征工程等。
