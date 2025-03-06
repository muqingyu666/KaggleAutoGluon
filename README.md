# AutoML 竞赛通用方案（AutoGluon Baseline）

一个适用于Kaggle/天池等数据科学竞赛的通用预测解决方案，基于AutoGluon实现自动化特征处理与模型堆叠，帮助快速构建基线模型。

## 项目概述

本方案为结构化数据二分类问题（如收入预测）提供端到端流程，主要优势：

🚀 **自动化特征处理**：智能处理缺失值，自动识别布尔型特征  
📚 **模型集成学习**：通过AutoGluon自动集成多层异构模型（XGBoost/LightGBM/CatBoost等）  
⚡ **快速验证迭代**：1小时内生成高精度基线，辅助后续特征工程优化  
📁 **开箱即用**：适配常见CSV数据格式，支持概率输出与类别预测

## 功能亮点

- 自动化数据预处理（缺失值填充/类型推断）
- 灵活选择评估指标（AUC/Accuracy/F1等）
- 自动保存模型及特征重要性分析图
- 兼容样本提交模板，生成标准格式结果

## 快速开始

1. **安装依赖**：
```bash
pip install autogluon pandas matplotlib
```

2. **准备数据**：
- 训练数据：`data/train_data.csv`
- 测试数据：`data/test_data.csv`
- 样本提交文件：`sample_submission.csv`

3. **运行主程序**：
```python
python main.py
```

4. **查看结果**：
- 模型文件：`autogluon_models_adult_income/`
- 预测结果：`results/submission_*.csv`
- 特征重要性图：`results/feature_importance.png`

## 适配你的数据

修改以下配置项即可适配新任务：
```python
# 目标变量名
LABEL = "your_target_column"
# 数据路径
TRAIN_DATA_PATH = "your_train.csv"
TEST_DATA_PATH = "your_test.csv"
# 调整训练时间（秒）
time_limit = 3600
```

## 注意事项

- 推荐GPU环境运行以加速神经网络模型训练
- 可通过`presets`参数切换模式（"best_quality"/"optimize_for_deployment"）

```
