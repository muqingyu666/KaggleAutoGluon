# AutoML Pipeline with AutoGluon (Classification & Regression)

This repository provides a configurable and easy-to-use AutoML pipeline based on AutoGluon. It's designed to help users quickly build high-quality baseline models for structured data competitions (e.g., Kaggle, Tianchi) and general machine learning tasks, supporting both classification and regression.

## Key Advantages

🚀 **Rapid Prototyping**: Generate strong baseline models in a short amount of time.
📚 **State-of-the-Art Models**: Leverages AutoGluon to train and ensemble various models (XGBoost, LightGBM, CatBoost, Neural Networks, etc.).
📊 **Versatile**: Automatically adapts to classification or regression tasks based on the chosen evaluation metric.
🔧 **Highly Configurable**: Most parameters are managed through a central `config.yaml` file.
📁 **User-Friendly**: Includes example data and is designed for easy adaptation to new datasets.
💡 **Advanced Customization**: Offers options for custom imputation strategies and explicit feature type definitions for advanced users.

## Example Data

This repository includes example data to run the AutoML pipeline out-of-the-box:
- **`data/train_data.csv`**: Sample data for model training.
- **`data/test_data.csv`**: Sample data for generating predictions.
- **`sample_submission.csv`**: An example of the submission file format.

The default `config.yaml` is pre-configured to use this dataset for a binary classification task (target column: `income`).

## Core Features

- **Intelligent Task Detection**: Automatically determines if the task is classification or regression based on the `EVAL_METRIC` in `config.yaml`.
  - **Classification Tasks**: Employs stratified K-fold for robust validation data splitting.
  - **Regression Tasks**: Uses standard K-fold cross-validation.
- **Automated Preprocessing**: Includes intelligent defaults for:
  - Missing value imputation (with options for mean/median for numerical, mode/constant for categorical features - configurable in the script).
  - Boolean and categorical feature type inference.
- **Flexible Evaluation**: Supports a wide range of standard evaluation metrics (e.g., AUC, F1, Accuracy, RMSE, R2).
- **Comprehensive Output**:
  - Saves trained AutoGluon models.
  - Generates a `leaderboard.csv` detailing the performance of all trained models.
  - Produces a feature importance plot (`feature_importance.png`).
  - Creates submission files in the specified format.
- **Advanced Control (Optional)**:
  - Allows users to define custom imputation strategies directly in the script.
  - Provides a template (`get_feature_metadata` function in script) for explicitly defining feature types, overriding AutoGluon's default inference if needed.

## Quick Start Guide

1.  **Environment Setup (Recommended)**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Your Data (Optional - For Custom Datasets)**:
    *   Place your `train_data.csv`, `test_data.csv`, and (optionally) `sample_submission.csv` in the `data/` directory or any other location.
    *   Update `TRAIN_DATA_PATH`, `TEST_DATA_PATH`, and `SAMPLE_SUBMISSION_PATH` in `config.yaml` to point to your files.
    *   If using the provided example data, no path changes are needed in `config.yaml`.

4.  **Configure the Pipeline (`config.yaml`)**:
    *   Open `config.yaml`.
    *   Set `LABEL`: Your target column name (e.g., `"income"` for the example data).
    *   Set `EVAL_METRIC`: This is crucial. Choose a metric appropriate for your task.
        *   **Classification Examples**: `'roc_auc'`, `'accuracy'`, `'f1'`, `'log_loss'`
        *   **Regression Examples**: `'rmse'`, `'mse'`, `'r2'`, `'mae'`
    *   Adjust `TIME_LIMIT`, `PRESETS`, and other parameters as needed. See the detailed "Configuration (`config.yaml`)" section below.

5.  **Run the AutoML Pipeline**:
    ```bash
    python autogluon_pipeline.py
    ```

6.  **Review Results**:
    *   All outputs are saved in the directory specified by `OUTPUT_DIR` in `config.yaml` (default: `results/`).
    *   **Trained Models**: Located in the directory specified by `MODELS_DIR` (default: `autogluon_models/`). These are AutoGluon predictor objects.
    *   **Submission File**: `submission_*.csv` (filename varies based on task and prediction type).
    *   **Model Leaderboard**: `leaderboard.csv` (shows performance of all models trained by AutoGluon).
    *   **Feature Importance**: `feature_importance.png`.

## Configuration (`config.yaml`)

All primary configurations are managed via the `config.yaml` file:

```yaml
# Configuration for AutoML pipeline

# Seed for reproducibility
SEED: 0

# Target variable name
LABEL: "income" # Name of the target column in your dataset.

# File paths
TRAIN_DATA_PATH: "data/train_data.csv"    # Path to the training data file.
TEST_DATA_PATH: "data/test_data.csv"     # Path to the test data file.
SAMPLE_SUBMISSION_PATH: "sample_submission.csv" # Optional: Path to a sample submission file. Can be empty or null if not used.
OUTPUT_DIR: "results"                     # Directory for all generated outputs (submission files, plots, leaderboard.csv).
MODELS_DIR: "autogluon_models"            # Directory to save trained AutoGluon models. Relative to the project root.

# AutoGluon settings
TIME_LIMIT: 3600       # Total training time limit for AutoGluon in seconds.
EVAL_METRIC: "roc_auc" # Evaluation metric. This choice also dictates the problem type:
                       # - Classification metrics (e.g., 'roc_auc', 'accuracy', 'f1', 'log_loss') trigger classification mode.
                       # - Regression metrics (e.g., 'rmse', 'mse', 'mae', 'r2') trigger regression mode.
PRESETS: "best_quality" # AutoGluon presets. Options include: 'best_quality', 'high_quality', 'medium_quality', 'good_quality', 'optimize_for_deployment'.

# Prediction settings
# PREDICTION_TYPE: Defines the content of the prediction output.
#   For Classification Tasks:
#     'class': Predicts only the class label (output to the column named by LABEL).
#     'prob': Predicts only the probability of the positive class.
#              - If SAMPLE_SUBMISSION_PATH is not used, the output column is named '{LABEL}_probability'.
#              - If SAMPLE_SUBMISSION_PATH is used, probabilities fill the column named by LABEL in that file.
#     'both': Predicts both the class label (in LABEL column) and the positive class probability (in '{LABEL}_probability' column).
#   For Regression Tasks:
#     This setting is largely ignored. The pipeline always predicts numerical values, output to the column named by LABEL.
PREDICTION_TYPE: "both"
```

## Advanced Script Customizations

For users needing finer control, the `autogluon_pipeline.py` script offers points for customization:

### 1. Custom Imputation Strategies
The `basic_preprocessing` function in the script allows you to specify imputation strategies beyond the defaults:
- `numerical_imputation_strategy`: Set to `'mean'` to use mean imputation for numerical features (default is `'median'`).
- `categorical_imputation_strategy`: Set to `'constant'` to use a custom constant for categorical features (default is `'mode'`).
- `categorical_fill_constant`: Specify the constant value when using `'constant'` for categorical imputation (default is `"missing"`).

To use these, modify the call to `basic_preprocessing` in the `if __name__ == "__main__":` block of `autogluon_pipeline.py` as shown in the commented-out example within the script.

### 2. Explicit Feature Type Definition
While AutoGluon excels at automatic feature type inference, you can provide explicit types if needed. The `get_feature_metadata` function in the script provides a template for this. By default, this function is not actively used, and AutoGluon infers types. To enable it:
1.  Modify the logic within `get_feature_metadata` to correctly assign types (e.g., `'category'`, `'float'`, `'text'`) to your specific columns.
2.  Uncomment the call to `get_feature_metadata` in the `if __name__ == "__main__":` block.
3.  Ensure the `feature_metadata` object returned by your modified function is passed to the `train_model` call.
Refer to the comments within `autogluon_pipeline.py` for detailed guidance.

## Important Notes
- **GPU Usage**: AutoGluon will automatically use a GPU if one is detected and available with appropriate drivers. This can significantly speed up training, especially for neural network models.
- **Presets**: The `PRESETS` parameter in `config.yaml` greatly affects training time and model quality. `'best_quality'` is thorough but time-consuming. For faster iterations, consider `'medium_quality'` or `'high_quality'`.

## Deprecated Configuration
Directly modifying global variables within the Python script for configuration is deprecated. Please use `config.yaml` for all standard configurations.
```
