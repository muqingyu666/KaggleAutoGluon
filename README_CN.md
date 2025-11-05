# Kaggle AutoGluon - 通用AutoML竞赛模板

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AutoGluon](https://img.shields.io/badge/AutoGluon-1.0+-green.svg)](https://auto.gluon.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **一键启动AutoML流程，快速生成Kaggle竞赛基线模型**

这是一个开箱即用的AutoML竞赛模板，基于AutoGluon构建，支持分类和回归任务。只需准备数据，修改配置文件，即可自动完成数据预处理、模型训练、预测和提交文件生成的全流程。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🚀 **开箱即用** | 提供完整示例数据，克隆即可运行 |
| 🧠 **智能识别** | 根据评估指标自动判断分类/回归任务 |
| 🔧 **灵活配置** | 所有参数通过YAML文件集中管理 |
| 📊 **自动预处理** | 智能处理缺失值、类型转换、特征推断 |
| 🏆 **集成学习** | 自动训练和集成多种模型（XGBoost、LightGBM、CatBoost等）|
| 📈 **可视化分析** | 自动生成特征重要性图和模型排行榜 |
| 💾 **即时提交** | 直接生成符合Kaggle格式的提交文件 |

---

## 📦 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd KaggleAutoGluon

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

将你的Kaggle竞赛数据放入 `data/` 目录：

```
data/
├── train_data.csv          # 训练数据（必需）
└── test_data.csv           # 测试数据（必需）
sample_submission.csv       # 提交格式示例（可选）
```

> 💡 **提示**: 项目已包含示例数据，可直接运行测试

### 3. 修改配置

编辑 `config.yaml`，只需修改以下关键参数：

```yaml
# 目标列名称
LABEL: "income"              # 改为你的目标列名

# 数据文件路径
TRAIN_DATA_PATH: "data/train_data.csv"
TEST_DATA_PATH: "data/test_data.csv"

# 评估指标（决定任务类型）
EVAL_METRIC: "roc_auc"       # 分类: roc_auc, accuracy, f1
                             # 回归: rmse, mae, r2

# 训练时间（秒）
TIME_LIMIT: 3600             # 根据数据量调整
```

### 4. 运行训练

```bash
python autogluon_pipeline.py
```

### 5. 查看结果

训练完成后，在 `results/` 目录下查看：

```
results/
├── submission_*.csv           # Kaggle提交文件
├── feature_importance.png     # 特征重要性可视化
└── leaderboard.csv           # 模型性能排行榜
```

---

## 📋 配置说明

### 核心配置项

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `LABEL` | 目标列名称 | `"income"`, `"price"`, `"target"` |
| `EVAL_METRIC` | 评估指标（决定任务类型） | 分类: `"roc_auc"`, `"accuracy"`<br>回归: `"rmse"`, `"r2"` |
| `TIME_LIMIT` | 训练时间限制（秒） | 快速测试: `300`<br>正式训练: `3600`<br>高质量: `7200+` |
| `PRESETS` | 模型质量档位 | `"best_quality"` - 最高质量<br>`"medium_quality"` - 平衡质量和速度 |
| `PREDICTION_TYPE` | 预测输出类型（分类） | `"class"` - 仅类别<br>`"prob"` - 仅概率<br>`"both"` - 类别+概率 |

### 完整配置参数

所有配置参数详见 [`config.yaml`](config.yaml)，包含详细中文注释。

---

## 🎯 使用场景

### 场景1: Kaggle二分类竞赛

```yaml
LABEL: "Survived"
EVAL_METRIC: "roc_auc"
PREDICTION_TYPE: "prob"
TIME_LIMIT: 3600
```

### 场景2: 房价预测（回归）

```yaml
LABEL: "SalePrice"
EVAL_METRIC: "rmse"
TIME_LIMIT: 5400
PRESETS: "best_quality"
```

### 场景3: 多分类任务

```yaml
LABEL: "species"
EVAL_METRIC: "accuracy"
PREDICTION_TYPE: "both"
```

---

## 🔬 高级功能

### 自定义缺失值填充策略

修改 `autogluon_pipeline.py` 中的预处理调用：

```python
train_df, test_df = basic_preprocessing(
    train_df, test_df, LABEL,
    numerical_imputation_strategy="mean",      # 数值列使用均值
    categorical_imputation_strategy="constant", # 类别列使用常量
    categorical_fill_constant="UNKNOWN"        # 填充值
)
```

### 手动定义特征类型

在 `get_feature_metadata()` 函数中取消注释并配置：

```python
special_types_to_add = {
    'age': ['float'],
    'category_col': ['category'],
    'text_col': ['text']
}
```

---

## 📊 输出文件说明

| 文件 | 说明 |
|------|------|
| `submission_*.csv` | Kaggle提交文件，可直接上传 |
| `feature_importance.png` | Top 10特征重要性柱状图 |
| `leaderboard.csv` | 所有训练模型的性能对比 |
| `autogluon_models/` | 训练好的模型文件，可用于推理 |

---

## 💡 最佳实践

### 1. 时间预算分配

- **探索阶段**: `TIME_LIMIT=300`, `PRESETS="medium_quality"` - 快速验证
- **优化阶段**: `TIME_LIMIT=3600`, `PRESETS="high_quality"` - 平衡性能
- **最终提交**: `TIME_LIMIT=7200+`, `PRESETS="best_quality"` - 追求极致

### 2. 评估指标选择

根据Kaggle竞赛页面的评估方式选择对应指标：

| Kaggle评估方式 | 对应EVAL_METRIC |
|---------------|----------------|
| AUC | `roc_auc` |
| Accuracy | `accuracy` |
| F1 Score | `f1` |
| RMSE | `rmse` |
| MAE | `mae` |
| R² | `r2` |

### 3. GPU加速

如果有GPU，AutoGluon会自动使用，显著提升神经网络训练速度。

---

## 🛠️ 故障排除

### 问题1: 内存不足

**解决方案**:
- 减少 `TIME_LIMIT`
- 使用 `PRESETS="medium_quality"` 或 `"good_quality"`
- 采样训练数据（在代码中添加 `train_df = train_df.sample(frac=0.5)`）

### 问题2: 训练时间过长

**解决方案**:
- 降低 `TIME_LIMIT`
- 使用更快的预设: `PRESETS="optimize_for_deployment"`

### 问题3: 预测文件格式不匹配

**解决方案**:
- 确保提供 `SAMPLE_SUBMISSION_PATH` 指向正确的示例提交文件
- 检查 `LABEL` 列名是否与Kaggle要求一致

---

## 📝 项目结构

```
KaggleAutoGluon/
├── autogluon_pipeline.py      # 核心AutoML流程脚本
├── config.yaml                # 配置文件（重点修改）
├── requirements.txt           # Python依赖
├── README.md                  # 英文文档
├── README_CN.md               # 中文文档（本文件）
├── .gitignore                 # Git忽略规则
├── data/                      # 数据目录
│   ├── train_data.csv         # 训练数据示例
│   └── test_data.csv          # 测试数据示例
├── sample_submission.csv      # 提交格式示例
├── results/                   # 输出目录（自动生成）
└── autogluon_models/          # 模型目录（自动生成）
```

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [AutoGluon](https://auto.gluon.ai/) - 强大的AutoML框架
- [Kaggle](https://www.kaggle.com/) - 数据科学竞赛平台

---

## 📮 联系方式

- 作者: Muqy
- 项目链接: [GitHub Repository](https://github.com/muqingyu666/KaggleAutoGluon)

---

**⭐ 如果这个项目对你有帮助，请给个Star支持一下！**
