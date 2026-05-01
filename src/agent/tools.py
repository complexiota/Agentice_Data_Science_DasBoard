import pandas as pd
from langchain_core.tools import tool
from src.core.data_cleaner import clean_nulls
from src.core.model_trainer import train_model, detect_problem_type
from src.core import data_store


@tool
def analyze_data() -> str:
    """Analyzes the currently uploaded dataset and returns column names, types, null counts, and basic stats."""
    df = data_store.get_data()
    if df is None:
        return "No data is currently uploaded. Please upload a dataset first."

    null_counts = df.isnull().sum()
    types = df.dtypes

    analysis = [f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns\n"]
    for col in df.columns:
        analysis.append(f"- `{col}` (Type: {types[col]}): {null_counts[col]} nulls")

    total_nulls = null_counts.sum()
    analysis.append(f"\n**Total null values:** {total_nulls}")

    return "\n".join(analysis)


@tool
def clean_data_tool(method: str, columns: list) -> str:
    """
    Cleans null values in the dataset using the specified method on the given columns.

    Args:
        method: One of:
            'Drop rows with nulls', 'Drop columns with nulls',
            'Fill with Mean (numeric only)', 'Fill with Median (numeric only)',
            'Fill with Mode (works for all types)', 'Fill with Constant Value',
            'Forward Fill (ffill)', 'Backward Fill (bfill)',
            'KNN Imputation (numeric only)', 'Interpolation (numeric only)'
        columns: List of column names to apply the method to. Pass all column names to apply to all.
    """
    df = data_store.get_data()
    if df is None:
        return "Error: No data available. Please upload a dataset first."

    # If user said "all columns", use all columns with nulls
    if not columns or columns == ["all"]:
        columns = df.columns[df.isnull().any()].tolist()

    if not columns:
        return "No columns with null values were found. Data is already clean!"

    try:
        cleaned_df = clean_nulls(
            df_in=df,
            method=method,
            selected_cols=columns
        )
        data_store.set_cleaned_data(cleaned_df)
        remaining_nulls = cleaned_df.isnull().sum().sum()
        rows_before = len(df)
        rows_after = len(cleaned_df)
        return (
            f"✅ Successfully cleaned data using **'{method}'** on columns: {columns}.\n"
            f"- Rows before: {rows_before} → Rows after: {rows_after}\n"
            f"- Remaining null values: {remaining_nulls}\n"
            f"The cleaned dataset is now ready for further analysis or model training."
        )
    except Exception as e:
        return f"Failed to clean data: {e}"


@tool
def train_model_tool(target_column: str, algorithm: str) -> str:
    """
    Trains a machine learning model. Auto-detects Classification vs Regression.

    Args:
        target_column: The column to predict.
        algorithm: One of:
            Classification: 'K-Nearest Neighbors (KNN)', 'Decision Tree',
                            'Support Vector Machine (SVM)', 'Random Forest Classifier'
            Regression: 'Linear Regression', 'Lasso Regression',
                        'Gradient Boosting Regression (GBR)', 'Decision Tree Regression',
                        'Artificial Neural Networks (ANN)'
    """
    # Prefer cleaned data, fallback to raw
    df = data_store.get_cleaned_data()
    if df is None:
        return "Error: No data available. Please upload a dataset first."

    if target_column not in df.columns:
        available = df.columns.tolist()
        return f"Error: Column '{target_column}' not found. Available columns: {available}"

    # Auto-detect problem type using robust heuristic
    # (model_trainer will also override this based on algorithm family)
    target_series = df[target_column]
    problem_type = detect_problem_type(target_series)

    try:
        results = train_model(
            data=df,
            target_column=target_column,
            selected_algorithm=algorithm,
            problem_type=problem_type,
            test_size_pct=20,
            cv_folds=5,
            hyperparams={}
        )

        data_store.set_model_results(results)

        cv_scores = results["cv_scores"]
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        summary_lines = [
            f"✅ **Model Trained Successfully!**",
            f"- **Algorithm:** {algorithm}",
            f"- **Problem Type:** {problem_type}",
            f"- **Target Column:** `{target_column}`",
            f"- **CV Score ({len(cv_scores)}-fold):** {cv_mean:.4f} ± {cv_std:.4f}",
        ]

        if problem_type == "Classification":
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(results["y_test"], results["y_pred"])
            summary_lines.append(f"- **Test Accuracy:** {acc:.4f}")
        else:
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(results["y_test"], results["y_pred"])
            rmse = mean_squared_error(results["y_test"], results["y_pred"]) ** 0.5
            summary_lines.append(f"- **Test R²:** {r2:.4f}")
            summary_lines.append(f"- **Test RMSE:** {rmse:.4f}")

        return "\n".join(summary_lines)

    except Exception as e:
        return f"Failed to train model: {e}"


@tool
def get_data_summary() -> str:
    """Returns a quick summary of the dataset including shape, column types, and descriptive statistics."""
    df = data_store.get_cleaned_data()
    if df is None:
        return "No data available. Please upload a dataset first."

    numeric_df = df.select_dtypes(include=["number"])
    summary = [
        f"**Dataset Summary**",
        f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        f"- Numeric columns: {list(numeric_df.columns)}",
        f"- Categorical columns: {list(df.select_dtypes(include='object').columns)}",
        f"- Total null values: {df.isnull().sum().sum()}",
        "",
        "**Descriptive Statistics (numeric only):**",
        numeric_df.describe().round(4).to_markdown() if not numeric_df.empty else "No numeric columns."
    ]
    return "\n".join(summary)
