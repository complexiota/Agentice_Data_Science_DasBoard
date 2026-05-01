import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import (
    train_test_split, learning_curve, cross_val_score,
    GridSearchCV, RandomizedSearchCV,
)
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestClassifier,
    RandomForestRegressor, GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    accuracy_score, r2_score,
    mean_absolute_error, mean_squared_error,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
)
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import re

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Data Analysis", layout="wide")
st.title("🏛️ Data DashBoard ")
st.sidebar.title("📋 Menu")

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
FILE_EXTENSIONS = {
    "CSV":   ["csv"],
    "XML":   ["xml"],
    "JSON":  ["json"],
    "Excel": ["xlsx", "xls"],
    "Text":  ["txt"],
}

FEATURES = [
    "Data Overview",
    "AI Agent Query",
    "Data Profiling Report",
    "EDA – Null Value Handling",
    "Feature Engineering",
    "Basic Statistics",
    "Visualizations",
    "Correlation Plot",
    "Outlier Detection",
    "Class Imbalance Handling",
    "Time Series Analysis",
    "ML Algorithm",
    "Multi-Model Comparison",
    "Hyperparameter Tuning",
    "Model Export & Predict",
    "Generate Report",
    "🧠 Domain Knowledge Constraints",
    "📓 Export Pipeline Notebook",  
]


# ─────────────────────────────────────────────
#  FILE PROCESSING
# ─────────────────────────────────────────────
def process_file(uploaded_file, file_mode: str):
    try:
        if file_mode == "CSV":
            return pd.read_csv(uploaded_file)
        elif file_mode == "XML":
            tree = ET.parse(uploaded_file)
            root = tree.getroot()
            xml_data = [
                {child.tag: child.text if child.text else None for child in element}
                for element in root
            ]
            return pd.DataFrame(xml_data)
        elif file_mode == "JSON":
            return pd.json_normalize(json.load(uploaded_file))
        elif file_mode == "Excel":
            return pd.read_excel(uploaded_file)
        elif file_mode == "Text":
            try:
                lines = [line.decode().strip() for line in uploaded_file.readlines()]
            except UnicodeDecodeError:
                lines = [line.strip() for line in uploaded_file.readlines()]
            return pd.DataFrame(lines, columns=["Text"])
    except Exception as e:
        st.error(f"Error processing file: {e}")
    return None


# ─────────────────────────────────────────────
#  FEATURE — AI AGENT QUERY (GROQ)
# ─────────────────────────────────────────────
def show_ai_agent_query(df: pd.DataFrame, api_key: str):
    st.subheader("🤖 AI Agent Query")
    st.caption("Ask natural language questions about your data, and the agent will generate and execute code to answer.")

    # UI for query
    query = st.text_area("What would you like to know or do with your data?",
                         placeholder="e.g., 'Show me a scatter plot of column A vs B', 'What is the average of column C grouped by D?', 'Detect outliers in column E'",
                         height=100)

    if st.button("🚀 Run AI Agent"):
        if not query:
            st.warning("Please enter a query.")
            return

        client = Groq(api_key=api_key)

        # Context for the LLM
        df_info = {
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
            "head": df.head(3).to_dict()
        }

        prompt = f"""
        You are an advanced agentic data science assistant integrated into a Streamlit dashboard.
        The user has a pandas DataFrame named 'df'.
        Here is the information about the DataFrame:
        - Columns: {df_info['columns']}
        - Data Types: {df_info['dtypes']}
        - Sample Data (first 3 rows): {df_info['head']}

        Dashboard Features available for navigation:
        {FEATURES}

        User Request: "{query}"

        Instructions:
        1. You can fulfill the request by writing Python code that will be executed via `exec()`.
        2. Always use 'df' as the variable name for the DataFrame.
        3. To show visualizations or results directly here, use `st.pyplot(fig)`, `st.plotly_chart(fig)`, or `st.write()`.
        4. **CRITICAL**: You can "select" features in the dashboard or navigate the user by modifying `st.session_state`.
           - To navigate to a feature, set `st.session_state["main_navigation_radio"] = "Feature Name"` and then call `st.rerun()`.
           - Example: To navigate to ML Algorithm, write `st.session_state["main_navigation_radio"] = "ML Algorithm"; st.rerun()`.
        5. To preset options in other modules, set their session state keys before navigating:
           - ML Algorithm Target Column: `st.session_state["ml_target_column"] = "ColumnName"`
           - ML Algorithm Choice: `st.session_state["ml_algo"] = "Algorithm Name"`
           - Null Handling Method: `st.session_state["null_method"] = "Method Name"`
        6. To automatically trigger actions upon navigation, set these flags:
           - `st.session_state["ml_auto_train"] = True` (Auto-clicks Train & Evaluate)
           - `st.session_state["null_auto_apply"] = True` (Auto-clicks Apply Cleaned Data)
        7. Provide ONLY the Python code block, wrapped in ```python ... ```.
        8. Do not include any explanations or extra text outside the code block.
        """

        with st.spinner("Agent is thinking..."):
            try:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful data science assistant that only outputs Python code."},
                        {"role": "user", "content": prompt}
                    ],
                    model=os.getenv("GROQ_MODEL", "llama3-70b-8192"), # or another groq model
                    temperature=0.1,
                )

                code_response = response.choices[0].message.content
                # Extract code from markdown block
                code_match = re.search(r"```python\s+(.*?)\s+```", code_response, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                else:
                    code = code_response # fallback

                st.write("### 🛠️ Generated Code")
                st.code(code, language="python")

                st.write("### 📈 Result")
                # Execute the code
                # We provide df, st, pd, np, plt, sns, px, go in the globals
                exec_globals = {
                    "df": df,
                    "pd": pd,
                    "np": np,
                    "plt": plt,
                    "sns": sns,
                    "px": px,
                    "go": go,
                    "st": st
                }
                exec(code, exec_globals)

            except Exception as e:
                st.error(f"Error running agent: {e}")
                st.write("Raw LLM response:")
                st.write(code_response)


# ─────────────────────────────────────────────
#  FEATURE 1 — DATA OVERVIEW
# ─────────────────────────────────────────────
def show_data_overview(data: pd.DataFrame):
    st.subheader("📊 Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", data.shape[0])
    col2.metric("Columns", data.shape[1])
    col3.metric("Total Cells", data.shape[0] * data.shape[1])
    st.write("**Column Names:**", data.columns.tolist())

    st.write("### Data Types Summary")
    dtype_counts = data.dtypes.value_counts().reset_index()
    dtype_counts.columns = ["Data Type", "Count"]
    st.dataframe(dtype_counts, use_container_width=True)

    st.write("### Columns by Data Type")
    dtype_dict = {}
    for col, dtype in zip(data.columns, data.dtypes):
        dtype_dict.setdefault(str(dtype), []).append(col)
    for dtype_name, columns in dtype_dict.items():
        with st.expander(f"**{dtype_name}** — {len(columns)} column(s)"):
            st.dataframe(pd.DataFrame({"Column Name": columns}), use_container_width=True)


# ─────────────────────────────────────────────
#  FEATURE 2 — EDA: NULL VALUE HANDLING
# ─────────────────────────────────────────────
def show_eda_null_handling(data: pd.DataFrame):
    st.subheader("🔍 EDA – Null Value Handling")

    # 1. Null Summary
    st.write("### 1. Null Value Summary")
    null_counts  = data.isnull().sum()
    null_percent = (null_counts / len(data) * 100).round(2)
    null_df = pd.DataFrame({
        "Column":         data.columns,
        "Null Count":     null_counts.values,
        "Null %":         null_percent.values,
        "Data Type":      data.dtypes.astype(str).values,
        "Non-Null Count": data.notnull().sum().values,
    }).sort_values("Null Count", ascending=False).reset_index(drop=True)

    def highlight_nulls(row):
        return ["background-color: #ffe0e0" if row["Null Count"] > 0 else "" for _ in row]

    st.dataframe(null_df.style.apply(highlight_nulls, axis=1), use_container_width=True)

    total_nulls = null_counts.sum()
    if total_nulls == 0:
        st.success("✅ No null values found in the dataset!")
        return

    st.warning(f"⚠️ Found **{total_nulls}** null value(s) across **{(null_counts > 0).sum()}** column(s).")

    # 2. Null Visualisation
    st.write("### 2. Null Value Visualisation")
    cols_with_nulls = null_counts[null_counts > 0].index.tolist()
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Bar Chart", "Heatmap", "Percentage Chart"])

    with viz_tab1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(cols_with_nulls, null_counts[cols_with_nulls], color="#E74C3C", edgecolor="white")
        ax.set(xlabel="Columns", ylabel="Null Count", title="Null Value Counts per Column")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    with viz_tab2:
        sample_size = min(100, len(data))
        sample_df   = data[cols_with_nulls].head(sample_size)
        fig, ax     = plt.subplots(figsize=(max(8, len(cols_with_nulls) * 0.8), 6))
        sns.heatmap(sample_df.isnull(), cbar=False, cmap="viridis", ax=ax, yticklabels=False)
        ax.set_title(f"Null Heatmap (first {sample_size} rows)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    with viz_tab3:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(cols_with_nulls, null_percent[cols_with_nulls], color="#3498DB", edgecolor="white")
        ax.axvline(x=50, color="red", linestyle="--", label="50% threshold")
        ax.set(xlabel="Null %", title="Null Percentage per Column")
        ax.legend()
        st.pyplot(fig)

    # 3. Handle Null Values
    st.write("### 3. Handle Null Values")
    st.info("Choose a scope and method. The cleaned dataset will be used in all subsequent steps.")

    scope = st.radio("Apply to:", ["All columns with nulls", "Selected columns only"], horizontal=True, key="null_scope")
    if scope == "Selected columns only":
        selected_cols = st.multiselect("Pick columns:", cols_with_nulls, default=cols_with_nulls[:1])
    else:
        selected_cols = cols_with_nulls

    if not selected_cols:
        st.warning("No columns selected.")
        return

    numeric_selected     = [c for c in selected_cols if pd.api.types.is_numeric_dtype(data[c])]
    categorical_selected = [c for c in selected_cols if not pd.api.types.is_numeric_dtype(data[c])]

    method = st.selectbox("Select method:", [
        "Drop rows with nulls",
        "Drop columns with nulls",
        "Fill with Mean (numeric only)",
        "Fill with Median (numeric only)",
        "Fill with Mode (works for all types)",
        "Fill with Constant Value",
        "Forward Fill (ffill)",
        "Backward Fill (bfill)",
        "KNN Imputation (numeric only)",
        "Interpolation (numeric only)",
    ], key="null_method")

    num_const     = 0.0
    cat_const     = "Unknown"
    knn_neighbors = 5

    if method == "Fill with Constant Value":
        col_a, col_b = st.columns(2)
        with col_a:
            num_const = st.number_input("Constant for numeric columns:", value=0.0)
        with col_b:
            cat_const = st.text_input("Constant for categorical columns:", value="Unknown")
    elif method == "KNN Imputation (numeric only)":
        knn_neighbors = st.slider("Number of neighbours (k):", 1, 20, 5)

    def apply_method(df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        if method == "Drop rows with nulls":
            df = df.dropna(subset=selected_cols)
        elif method == "Drop columns with nulls":
            df = df.drop(columns=selected_cols)
        elif method == "Fill with Mean (numeric only)":
            for c in numeric_selected:
                df[c] = df[c].fillna(df[c].mean())
        elif method == "Fill with Median (numeric only)":
            for c in numeric_selected:
                df[c] = df[c].fillna(df[c].median())
        elif method == "Fill with Mode (works for all types)":
            for c in selected_cols:
                mode_val = df[c].mode()
                if not mode_val.empty:
                    df[c] = df[c].fillna(mode_val[0])
        elif method == "Fill with Constant Value":
            for c in numeric_selected:
                df[c] = df[c].fillna(num_const)
            for c in categorical_selected:
                df[c] = df[c].fillna(cat_const)
        elif method == "Forward Fill (ffill)":
            df[selected_cols] = df[selected_cols].ffill()
        elif method == "Backward Fill (bfill)":
            df[selected_cols] = df[selected_cols].bfill()
        elif method == "KNN Imputation (numeric only)":
            from sklearn.impute import KNNImputer
            if numeric_selected:
                imputer = KNNImputer(n_neighbors=knn_neighbors)
                df[numeric_selected] = imputer.fit_transform(df[numeric_selected])
        elif method == "Interpolation (numeric only)":
            for c in numeric_selected:
                df[c] = df[c].interpolate(method="linear", limit_direction="both")
        return df

    # Before / After preview
    st.write("#### Preview: Before vs After (first 10 rows)")
    preview_before = data[selected_cols].head(10)
    if method == "Drop columns with nulls":
        st.write("*Columns will be removed — no preview available.*")
    else:
        preview_after_df = apply_method(data)
        safe_cols = [c for c in selected_cols if c in preview_after_df.columns]
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Before**")
            st.dataframe(preview_before.style.highlight_null(color="#ffe0e0"), use_container_width=True)
        with c2:
            st.write("**After**")
            st.dataframe(preview_after_df[safe_cols].head(10), use_container_width=True)

    if st.button("✅ Apply & Save Cleaned Data", type="primary") or st.session_state.pop("null_auto_apply", False):
        cleaned = apply_method(data)
        st.session_state["cleaned_data"] = cleaned
        remaining = cleaned.isnull().sum().sum()
        rows_drop = len(data) - len(cleaned)
        cols_drop = len(data.columns) - len(cleaned.columns)
        st.success("✅ Cleaned data saved! All further analysis will now use this processed dataset.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Remaining Nulls", remaining)
        col2.metric("Rows Dropped",    rows_drop)
        col3.metric("Columns Dropped", cols_drop)
        st.dataframe(cleaned.head(20), use_container_width=True)
        st.rerun()

    # 4. Strategy Guide
    with st.expander("📖 Guide: Which method should I use?"):
        guide = {
            "Method": [
                "Drop rows", "Drop columns", "Fill – Mean", "Fill – Median",
                "Fill – Mode", "Fill – Constant", "Forward Fill",
                "Backward Fill", "KNN Imputation", "Interpolation",
            ],
            "Best for": [
                "Very few null rows (< 5%)", "Column has > 60% nulls",
                "Normally distributed numeric data", "Skewed numeric / outliers present",
                "Categorical or bimodal data", "Domain-specific known default (e.g. 0)",
                "Time series / ordered data", "Time series / ordered data",
                "Correlated numeric features", "Numeric time series with trends",
            ],
            "Drawback": [
                "Loses data", "Loses features", "Sensitive to outliers",
                "Loses variance info", "May over-represent one class",
                "May introduce bias", "Propagates last known value",
                "May propagate future values backward",
                "Computationally expensive on large data",
                "Assumes linear trend between points",
            ],
        }
        st.dataframe(pd.DataFrame(guide), use_container_width=True)


# ─────────────────────────────────────────────
#  FEATURE 3 — BASIC STATISTICS
# ─────────────────────────────────────────────
def show_basic_statistics(data: pd.DataFrame):
    st.subheader("📈 Basic Statistics")
    numeric_data = data.select_dtypes(include=["number"])
    if numeric_data.empty:
        st.warning("No numeric columns available.")
        return
    st.write("#### Descriptive Statistics")
    st.dataframe(numeric_data.describe(), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Skewness")
        st.dataframe(numeric_data.skew().rename("Skewness"), use_container_width=True)
    with col2:
        st.write("#### Kurtosis")
        st.dataframe(numeric_data.kurtosis().rename("Kurtosis"), use_container_width=True)


# ─────────────────────────────────────────────
#  FEATURE 4 — VISUALIZATIONS
# ─────────────────────────────────────────────
def show_visualizations(data: pd.DataFrame):
    st.subheader("📉 Visualizations")
    numeric_columns     = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()

    if not numeric_columns:
        st.warning("No numeric columns available.")
        return

    selected_columns = st.multiselect("Select columns to visualise:", numeric_columns)
    vis_types = st.multiselect("Select visualisation types:", [
        "Histogram", "Boxplot", "Scatterplot", "Lineplot",
        "Violinplot", "Barplot", "Pairplot", "Heatmap",
    ])

    if not selected_columns or not vis_types:
        st.info("Please select at least one column and one visualisation type.")
        return

    for i, vis_type in enumerate(vis_types):
        st.write(f"#### {vis_type}")
        fig, ax = plt.subplots(figsize=(8, 5))

        if vis_type == "Histogram":
            kde  = st.checkbox("Show KDE", value=True, key=f"kde_{i}")
            bins = st.slider("Bins", 5, 50, 20, key=f"bins_{i}")
            for col in selected_columns:
                sns.histplot(data[col], kde=kde, bins=bins, label=col, ax=ax)
            ax.legend(); ax.set_title("Histogram")

        elif vis_type == "Boxplot":
            sns.boxplot(data=data[selected_columns], ax=ax)
            ax.set_title("Boxplot")

        elif vis_type == "Scatterplot":
            if len(selected_columns) < 2:
                st.warning("Select at least 2 columns for scatterplot."); continue
            sns.scatterplot(x=data[selected_columns[0]], y=data[selected_columns[1]], ax=ax)
            ax.set_title("Scatterplot")

        elif vis_type == "Lineplot":
            if len(selected_columns) < 2:
                st.warning("Select at least 2 columns for lineplot."); continue
            sns.lineplot(x=data[selected_columns[0]], y=data[selected_columns[1]], ax=ax)
            ax.set_title("Lineplot")

        elif vis_type == "Violinplot":
            for col in selected_columns:
                sns.violinplot(y=data[col], ax=ax)
            ax.set_title("Violinplot")

        elif vis_type == "Barplot":
            if not categorical_columns:
                st.warning("No categorical columns available for Barplot."); continue
            cat_col = st.selectbox("Categorical column:", categorical_columns, key=f"cat_{i}")
            for col in selected_columns:
                sns.barplot(x=data[cat_col], y=data[col], ax=ax)
            ax.set_title("Barplot")

        elif vis_type == "Pairplot":
            pair_fig = sns.pairplot(data[selected_columns])
            st.pyplot(pair_fig); continue

        elif vis_type == "Heatmap":
            corr = data[selected_columns].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Heatmap")

        st.pyplot(fig)


# ─────────────────────────────────────────────
#  TURE 5 — CORRELATION PLOT
# ─────────────────────────────────────────────
def show_correlation_plot(data: pd.DataFrame):
    st.subheader("🔗 Correlation Plot")
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if not numeric_columns:
        st.warning("No numeric columns to generate correlation plot.")
        return
    corr = data[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
                annot_kws={"size": 8}, linewidths=0.5, ax=ax)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    st.pyplot(fig)


# ─────────────────────────────────────────────
#  FEATURE 6 — ML ALGORITHM
# ─────────────────────────────────────────────
def show_ml_algorithm(data: pd.DataFrame):
    from sklearn.model_selection import learning_curve, cross_val_score
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        mean_absolute_error, mean_squared_error,
        ConfusionMatrixDisplay,
    )
    from sklearn.preprocessing import LabelEncoder

    st.subheader("🤖 ML Algorithm Selection")

    CLASSIFICATION_ALGOS = [
        "K-Nearest Neighbors (KNN)",
        "Decision Tree",
        "Support Vector Machine (SVM)",
        "Random Forest Classifier",
    ]
    REGRESSION_ALGOS = [
        "Linear Regression",
        "Lasso Regression",
        "Gradient Boosting Regression (GBR)",
        "Decision Tree Regression",
        "Artificial Neural Networks (ANN)",
    ]

    def detect_problem_type(target: pd.Series) -> str:
        return (
            "Classification"
            if target.dtype == "object" or target.nunique() < 20
            else "Regression"
        )

    numeric_columns = data.select_dtypes(include=["number"]).columns
    if numeric_columns.empty:
        st.error("No valid numeric target columns available.")
        return

    # ── Config ────────────────────────────────
    st.write("#### ⚙️ Configuration")
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        target_column = st.selectbox("Target Column:", numeric_columns, key="ml_target_column")
    with cfg2:
        test_size_pct = st.slider("Test size %:", 10, 40, 20, 5, key="ml_test_pct")
    with cfg3:
        cv_folds = st.slider("Cross-validation folds:", 3, 10, 5, key="ml_cv")

    target       = data[target_column]
    problem_type = detect_problem_type(target)
    algorithms   = (
        CLASSIFICATION_ALGOS if problem_type == "Classification" else REGRESSION_ALGOS
    )

    st.info(f"**Detected Problem Type:** `{problem_type}`")

    selected_algorithm = st.selectbox("Select Algorithm:", algorithms, key="ml_algo")

    # Algorithm-specific hyperparameters
    st.write("#### 🔧 Hyperparameters")
    hp1, hp2 = st.columns(2)

    with hp1:
        if selected_algorithm == "K-Nearest Neighbors (KNN)":
            knn_k        = st.slider("n_neighbors:", 1, 30, 5, key="knn_k")
            knn_weights  = st.selectbox("weights:", ["uniform", "distance"], key="knn_w")
        elif selected_algorithm in ("Decision Tree", "Decision Tree Regression"):
            dt_depth     = st.slider("max_depth (0 = None):", 0, 20, 0, key="dt_d")
            dt_min_split = st.slider("min_samples_split:", 2, 20, 2, key="dt_ms")
        elif selected_algorithm == "Lasso Regression":
            lasso_alpha  = st.number_input("alpha:", min_value=0.0001, value=1.0,
                                           step=0.1, key="lasso_a")
        elif selected_algorithm in ("Random Forest Classifier",):
            rf_n         = st.slider("n_estimators:", 10, 300, 100, 10, key="rf_n")
            rf_depth     = st.slider("max_depth (0 = None):", 0, 20, 0, key="rf_d")
        elif selected_algorithm == "Gradient Boosting Regression (GBR)":
            gbr_n        = st.slider("n_estimators:", 50, 500, 100, 50, key="gbr_n")
            gbr_lr       = st.number_input("learning_rate:", 0.001, 1.0, 0.1,
                                           step=0.01, key="gbr_lr")
        elif selected_algorithm == "Artificial Neural Networks (ANN)":
            ann_layers   = st.text_input("Hidden layer sizes (e.g. 100,50):",
                                         value="100,50", key="ann_l")
            ann_lr       = st.number_input("learning_rate_init:", 0.0001, 0.1, 0.001,
                                           step=0.0001, key="ann_lr")
        elif selected_algorithm == "Support Vector Machine (SVM)":
            svm_c        = st.number_input("C:", 0.01, 100.0, 1.0, step=0.1, key="svm_c")
            svm_kernel   = st.selectbox("kernel:", ["rbf","linear","poly","sigmoid"],
                                        key="svm_k")

    if st.button("▶️ Train & Evaluate", key="ml_run_btn", type="primary") or st.session_state.pop("ml_auto_train", False):

        # ── Prepare data ──────────────────────
        X_raw = data.drop(target_column, axis=1).select_dtypes(include=["number"])
        y_raw = data[target_column]
        valid = y_raw.dropna().index
        X_raw, y_raw = X_raw.loc[valid], y_raw.loc[valid]

        imputer = SimpleImputer(strategy="mean")
        X_imp   = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

        # Encode target for classification if needed
        y_enc = y_raw.reset_index(drop=True)
        le    = None
        if problem_type == "Classification" and y_enc.dtype not in ["int64","int32"]:
            le    = LabelEncoder()
            y_enc = pd.Series(le.fit_transform(y_enc))

        X_imp = X_imp.reset_index(drop=True)
        scaler   = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_enc, test_size=test_size_pct / 100, random_state=42
        )

        # ── Build model ───────────────────────
        if selected_algorithm == "K-Nearest Neighbors (KNN)":
            model = KNeighborsClassifier(n_neighbors=knn_k, weights=knn_weights)
        elif selected_algorithm == "Decision Tree":
            model = DecisionTreeClassifier(
                max_depth=dt_depth if dt_depth > 0 else None,
                min_samples_split=dt_min_split, random_state=42,
            )
        elif selected_algorithm == "Support Vector Machine (SVM)":
            model = SVC(C=svm_c, kernel=svm_kernel, probability=True)
        elif selected_algorithm == "Random Forest Classifier":
            model = RandomForestClassifier(
                n_estimators=rf_n,
                max_depth=rf_depth if rf_depth > 0 else None,
                random_state=42,
            )
        elif selected_algorithm == "Linear Regression":
            model = LinearRegression()
        elif selected_algorithm == "Lasso Regression":
            model = Lasso(alpha=lasso_alpha)
        elif selected_algorithm == "Gradient Boosting Regression (GBR)":
            model = GradientBoostingRegressor(
                n_estimators=gbr_n, learning_rate=gbr_lr, random_state=42
            )
        elif selected_algorithm == "Decision Tree Regression":
            model = DecisionTreeRegressor(
                max_depth=dt_depth if dt_depth > 0 else None,
                min_samples_split=dt_min_split, random_state=42,
            )
        elif selected_algorithm == "Artificial Neural Networks (ANN)":
            layers = tuple(int(x.strip()) for x in ann_layers.split(",") if x.strip())
            model  = MLPRegressor(
                hidden_layer_sizes=layers, learning_rate_init=ann_lr,
                max_iter=500, random_state=42,
            )

        model.fit(X_train, y_train)
        y_pred       = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # ── Cross-validation ──────────────────
        cv_scoring = "accuracy" if problem_type == "Classification" else "r2"
        cv_scores  = cross_val_score(model, X_scaled, y_enc,
                                     cv=cv_folds, scoring=cv_scoring)

        # ── Store results ─────────────────────
        st.session_state["ml_res_model"]    = model
        st.session_state["ml_res_X_train"]  = X_train
        st.session_state["ml_res_X_test"]   = X_test
        st.session_state["ml_res_y_train"]  = y_train
        st.session_state["ml_res_y_test"]   = y_test
        st.session_state["ml_res_y_pred"]   = y_pred
        st.session_state["ml_res_y_pred_tr"]= y_pred_train
        st.session_state["ml_res_cv"]       = cv_scores
        st.session_state["ml_res_prob"]     = problem_type
        st.session_state["ml_res_algo"]     = selected_algorithm
        st.session_state["ml_res_X_cols"]   = X_imp.columns.tolist()
        st.session_state["ml_res_le"]       = le
        st.session_state["ml_res_X_scaled"] = X_scaled
        st.session_state["ml_res_y_enc"]    = y_enc

    # ── Results (persist across reruns) ───────
    if "ml_res_model" not in st.session_state:
        return

    model        = st.session_state["ml_res_model"]
    X_train      = st.session_state["ml_res_X_train"]
    X_test       = st.session_state["ml_res_X_test"]
    y_train      = st.session_state["ml_res_y_train"]
    y_test       = st.session_state["ml_res_y_test"]
    y_pred       = st.session_state["ml_res_y_pred"]
    y_pred_train = st.session_state["ml_res_y_pred_tr"]
    cv_scores    = st.session_state["ml_res_cv"]
    problem_type = st.session_state["ml_res_prob"]
    algo_name    = st.session_state["ml_res_algo"]
    feat_cols    = st.session_state["ml_res_X_cols"]
    le           = st.session_state["ml_res_le"]
    X_scaled     = st.session_state["ml_res_X_scaled"]
    y_enc        = st.session_state["ml_res_y_enc"]

    st.markdown("---")
    st.write("### 📊 Model Results")

    # ════════════════════════════════════════
    #  METRICS ROW
    # ════════════════════════════════════════
    if problem_type == "Classification":
        from sklearn.metrics import precision_score, recall_score, f1_score
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc  = accuracy_score(y_test,  y_pred)
        prec      = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec       = recall_score(y_test,  y_pred, average="weighted", zero_division=0)
        f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Train Accuracy",  f"{train_acc:.4f}")
        m2.metric("Test Accuracy",   f"{test_acc:.4f}",
                  delta=f"{test_acc - train_acc:+.4f}")
        m3.metric("Precision",       f"{prec:.4f}")
        m4.metric("Recall",          f"{rec:.4f}")
        m5.metric("F1 Score",        f"{f1:.4f}")
        m6.metric(f"CV ({len(cv_scores)}-fold)",
                  f"{cv_mean:.4f} ± {cv_std:.4f}")
    else:
        train_r2   = r2_score(y_train, y_pred_train)
        test_r2    = r2_score(y_test,  y_pred)
        train_rmse = mean_squared_error(y_train, y_pred_train) ** 0.5
        test_rmse  = mean_squared_error(y_test,  y_pred) ** 0.5
        test_mae   = mean_absolute_error(y_test, y_pred)
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Train R²",        f"{train_r2:.4f}")
        m2.metric("Test R²",         f"{test_r2:.4f}",
                  delta=f"{test_r2 - train_r2:+.4f}")
        m3.metric("Train RMSE",      f"{train_rmse:.4f}")
        m4.metric("Test RMSE",       f"{test_rmse:.4f}")
        m5.metric("Test MAE",        f"{test_mae:.4f}")
        m6.metric(f"CV ({len(cv_scores)}-fold)",
                  f"{cv_mean:.4f} ± {cv_std:.4f}")

    st.markdown("---")

    # ════════════════════════════════════════
    #  TABS
    # ════════════════════════════════════════
    if problem_type == "Classification":
        tab_names = [
            "Learning Curve", "Actual vs Predicted",
            "Confusion Matrix", "Classification Report",
            "CV Scores", "Feature Importance",
        ]
    else:
        tab_names = [
            "Learning Curve", "Actual vs Predicted",
            "Residuals", "Error Distribution",
            "CV Scores", "Feature Importance",
        ]

    tabs = st.tabs(tab_names)

    # ── Tab 1 — Learning Curve ────────────────
    with tabs[0]:
        st.write("##### Training vs Validation Score Across Dataset Sizes")
        st.caption(
            "A converging gap = good fit. "
            "Large persistent gap = overfitting. "
            "Both curves low = underfitting."
        )
        lc_scoring = "accuracy" if problem_type == "Classification" else "r2"
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_scaled, y_enc,
            cv=cv_folds,
            scoring=lc_scoring,
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
        )
        tr_mean  = train_scores.mean(axis=1)
        tr_std   = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std  = val_scores.std(axis=1)

        fig_lc, ax_lc = plt.subplots(figsize=(9, 5))
        ax_lc.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                           alpha=0.15, color="#2980B9")
        ax_lc.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                           alpha=0.15, color="#E74C3C")
        ax_lc.plot(train_sizes, tr_mean,  "o-", color="#2980B9",
                   linewidth=2, label="Training Score")
        ax_lc.plot(train_sizes, val_mean, "s-", color="#E74C3C",
                   linewidth=2, label="Validation Score")
        ax_lc.set(xlabel="Training Set Size",
                  ylabel="Accuracy" if problem_type == "Classification" else "R²",
                  title=f"Learning Curve — {algo_name}")
        ax_lc.legend(fontsize=10)
        ax_lc.grid(True, linestyle="--", alpha=0.4)
        ax_lc.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_lc)

        # Overfit / underfit diagnosis
        gap = float(tr_mean[-1] - val_mean[-1])
        if gap > 0.15:
            st.warning(
                f"⚠️ **Overfitting detected** — training score is {gap:.2f} higher than "
                f"validation. Try regularisation, more data, or pruning."
            )
        elif val_mean[-1] < 0.6:
            st.warning(
                "⚠️ **Underfitting** — both curves are low. "
                "Try a more complex model or more features."
            )
        else:
            st.success("✅ Model generalises well — training and validation scores are close.")

    # ── Tab 2 — Actual vs Predicted ───────────
    with tabs[1]:
        if problem_type == "Classification":
            fig_avp, ax_avp = plt.subplots(figsize=(8, 5))
            ax_avp.scatter(range(len(y_test)), y_test,
                           color="#2980B9", alpha=0.6, s=40, label="Actual")
            ax_avp.scatter(range(len(y_pred)), y_pred,
                           color="#E74C3C", alpha=0.6, s=40, marker="x",
                           linewidths=1.5, label="Predicted")
            ax_avp.set(xlabel="Sample Index", ylabel="Class",
                       title="Actual vs Predicted Labels")
            ax_avp.legend()
            ax_avp.grid(True, linestyle="--", alpha=0.3)
            ax_avp.spines[["top","right"]].set_visible(False)
        else:
            fig_avp, ax_avp = plt.subplots(figsize=(8, 5))
            ax_avp.scatter(y_test, y_pred, alpha=0.55, color="#3498DB",
                           edgecolors="white", linewidths=0.3, s=55, label="Test")
            ax_avp.scatter(y_train, y_pred_train, alpha=0.35, color="#E67E22",
                           edgecolors="white", linewidths=0.3, s=35, label="Train")
            lims = [
                min(y_test.min(), y_pred.min()),
                max(y_test.max(), y_pred.max()),
            ]
            ax_avp.plot(lims, lims, "r--", linewidth=1.5, label="Ideal (y = ŷ)")
            ax_avp.set(xlabel="Actual Values", ylabel="Predicted Values",
                       title=f"Actual vs Predicted — {algo_name}")
            ax_avp.legend()
            ax_avp.grid(True, linestyle="--", alpha=0.3)
            ax_avp.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_avp)

    # ── Tab 3 — Confusion Matrix / Residuals ──
    with tabs[2]:
        if problem_type == "Classification":
            st.write("##### Confusion Matrix")
            cm   = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            labels = le.classes_ if le else sorted(set(y_test))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                         display_labels=labels)
            disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
            ax_cm.set_title("Confusion Matrix")
            plt.tight_layout()
            st.pyplot(fig_cm)
        else:
            st.write("##### Residuals Plot (Predicted vs Residual)")
            residuals = y_test.values - y_pred
            fig_res, axes_res = plt.subplots(1, 2, figsize=(12, 4))
            # Residuals vs predicted
            axes_res[0].scatter(y_pred, residuals, alpha=0.55, color="#3498DB",
                                edgecolors="white", linewidths=0.3, s=50)
            axes_res[0].axhline(0, color="#E74C3C", linestyle="--", linewidth=1.5)
            axes_res[0].set(xlabel="Predicted Values", ylabel="Residuals",
                            title="Residuals vs Predicted")
            axes_res[0].grid(True, linestyle="--", alpha=0.3)
            axes_res[0].spines[["top","right"]].set_visible(False)
            # Residuals vs index
            axes_res[1].plot(residuals, "o", alpha=0.5, color="#8E44AD",
                             markersize=4)
            axes_res[1].axhline(0, color="#E74C3C", linestyle="--", linewidth=1.5)
            axes_res[1].set(xlabel="Sample Index", ylabel="Residuals",
                            title="Residuals vs Sample Index")
            axes_res[1].grid(True, linestyle="--", alpha=0.3)
            axes_res[1].spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_res)

    # ── Tab 4 — Classification Report / Error Distribution ──
    with tabs[3]:
        if problem_type == "Classification":
            st.write("##### Classification Report")
            labels_disp = le.classes_ if le else None
            report_dict = classification_report(
                y_test, y_pred, target_names=labels_disp,
                output_dict=True, zero_division=0,
            )
            report_df = pd.DataFrame(report_dict).T
            st.dataframe(report_df.style.background_gradient(cmap="Greens", axis=0),
                         use_container_width=True)
        else:
            st.write("##### Residuals Distribution")
            residuals = y_test.values - y_pred
            fig_ed, axes_ed = plt.subplots(1, 2, figsize=(12, 4))
            # Histogram
            axes_ed[0].hist(residuals, bins=30, color="#3498DB",
                            edgecolor="white", alpha=0.8)
            axes_ed[0].axvline(0, color="#E74C3C", linestyle="--", linewidth=1.5)
            axes_ed[0].set(xlabel="Residual", ylabel="Frequency",
                           title="Residuals Histogram")
            axes_ed[0].spines[["top","right"]].set_visible(False)
            # Q-Q plot proxy (sorted residuals vs normal quantiles)
            import scipy.stats as stats
            (osm, osr), (slope, intercept, _) = stats.probplot(residuals)
            axes_ed[1].scatter(osm, osr, alpha=0.6, color="#8E44AD",
                               edgecolors="white", s=30)
            axes_ed[1].plot(osm, slope * np.array(osm) + intercept,
                            color="#E74C3C", linestyle="--", linewidth=1.5)
            axes_ed[1].set(xlabel="Theoretical Quantiles",
                           ylabel="Sample Quantiles", title="Q-Q Plot of Residuals")
            axes_ed[1].spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_ed)

    # ── Tab 5 — CV Scores ─────────────────────
    with tabs[4]:
        st.write(f"##### {cv_folds}-Fold Cross-Validation Scores")
        cv_df = pd.DataFrame({
            "Fold":  [f"Fold {i+1}" for i in range(len(cv_scores))],
            "Score": cv_scores,
        })
        fig_cv, ax_cv = plt.subplots(figsize=(8, 4))
        bar_colors = [
            "#27AE60" if s >= cv_scores.mean() else "#E74C3C"
            for s in cv_scores
        ]
        ax_cv.bar(cv_df["Fold"], cv_df["Score"], color=bar_colors,
                  edgecolor="white", linewidth=0.8)
        ax_cv.axhline(cv_scores.mean(), color="#2C3E50", linestyle="--",
                      linewidth=1.5, label=f"Mean = {cv_scores.mean():.4f}")
        ax_cv.fill_between(
            range(len(cv_scores)),
            cv_scores.mean() - cv_scores.std(),
            cv_scores.mean() + cv_scores.std(),
            alpha=0.12, color="#2C3E50", label=f"±1 std ({cv_scores.std():.4f})",
        )
        ax_cv.set(xlabel="Fold", ylabel=cv_scoring.capitalize(),
                  title=f"Cross-Validation — {algo_name}")
        ax_cv.legend(fontsize=9)
        ax_cv.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_cv)

        # Table
        cv_df["vs Mean"] = cv_df["Score"] - cv_scores.mean()
        st.dataframe(
            cv_df.style
                .format({"Score": "{:.4f}", "vs Mean": "{:+.4f}"})
                .background_gradient(subset=["Score"], cmap="RdYlGn"),
            use_container_width=True,
        )

    # ── Tab 6 — Feature Importance ────────────
    with tabs[5]:
        st.write("##### Feature Importance")
        importance = None

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importance = np.abs(coef).flatten()[:len(feat_cols)]

        if importance is not None and len(importance) == len(feat_cols):
            fi_df = (
                pd.DataFrame({"Feature": feat_cols, "Importance": importance})
                .sort_values("Importance", ascending=False)
                .reset_index(drop=True)
            )
            top_n = min(20, len(fi_df))
            fi_top = fi_df.head(top_n)

            fig_fi, ax_fi = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))[::-1]
            ax_fi.barh(fi_top["Feature"][::-1], fi_top["Importance"][::-1],
                       color=colors, edgecolor="white")
            ax_fi.set(xlabel="Importance", title=f"Top {top_n} Feature Importances")
            ax_fi.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_fi)

            st.dataframe(fi_df.style.background_gradient(
                subset=["Importance"], cmap="YlOrRd"),
                use_container_width=True,
            )
        else:
            st.info(
                "Feature importance is not directly available for this model. "
                "Try Decision Tree, Random Forest, or Gradient Boosting for built-in importances, "
                "or Linear/Lasso Regression for coefficient-based importance."
            )


# ─────────────────────────────────────────────
#  FEATURE 7 — OUTLIER DETECTION
#              + PCA ANALYSIS
#              + PCA + CLUSTERING PIPELINE
# ─────────────────────────────────────────────
def show_outlier_detection(data: pd.DataFrame):
    st.subheader("🚨 Outlier Detection, PCA & Clustering")
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns

    if numeric_columns.empty:
        st.warning("No numeric columns available.")
        return

    # ════════════════════════════════════════
    #  SECTION A — TRAIN-TEST SPLIT
    # ════════════════════════════════════════
    st.write("### 📂 Train-Test Split")
    test_pct  = st.slider("Test size %", 10, 50, 20, 5, key="split_slider")
    test_size = test_pct / 100

    if st.button("Split Data", key="split_btn"):
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        st.session_state["train_data"] = train_data
        st.session_state["test_data"]  = test_data
        st.success(f"Split done → Train: {len(train_data)} rows | Test: {len(test_data)} rows")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Training Data (sample)")
            st.dataframe(train_data.head(5), use_container_width=True)
        with c2:
            st.write("Test Data (sample)")
            st.dataframe(test_data.head(5), use_container_width=True)

    st.markdown("---")

    # ════════════════════════════════════════
    #  SECTION B — PCA ANALYSIS
    # ════════════════════════════════════════
    st.write("### 🔬 PCA Analysis")

    if "train_data" not in st.session_state:
        st.info("Split the data first to enable PCA analysis.")
    else:
        all_num_cols = st.session_state["train_data"].select_dtypes(
            include=["float64", "int64"]
        ).columns.tolist()

        feat_mode = st.radio(
            "Feature selection for PCA:",
            ["Use all numeric features", "Select specific features"],
            key="pca_feat_mode",
        )
        pca_features = (
            st.multiselect("Select features:", all_num_cols, key="pca_feats")
            if feat_mode == "Select specific features"
            else all_num_cols
        )

        if pca_features:
            max_comp = min(len(pca_features), 10)
            n_comp   = st.slider("PCA components:", 2, max_comp, min(3, max_comp), key="pca_n")

            if st.button("▶️ Run PCA on Training Data", key="run_pca_btn"):
                X_train = st.session_state["train_data"][pca_features].copy()
                if X_train.isnull().sum().sum() > 0:
                    st.warning("Missing values found – filling with column means.")
                    X_train = X_train.fillna(X_train.mean())

                scaler     = StandardScaler()
                X_scaled   = scaler.fit_transform(X_train)
                pca        = PCA(n_components=n_comp)
                pca_result = pca.fit_transform(X_scaled)

                st.session_state.update({
                    "pca_model":    pca,
                    "pca_scaler":   scaler,
                    "pca_features": pca_features,
                })
                pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_comp)])
                st.session_state["train_pca_df"] = pca_df

                expl_var = pca.explained_variance_ratio_
                cum_var  = np.cumsum(expl_var)

                # Explained variance chart
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(range(1, len(expl_var) + 1), expl_var, alpha=0.7, label="Individual")
                ax.step(range(1, len(cum_var) + 1), cum_var, where="mid",
                        color="red", label="Cumulative")
                ax.set(xlabel="Principal Component", ylabel="Explained Variance Ratio",
                       title="Explained Variance by PC")
                ax.legend()
                st.pyplot(fig)

                # 3D plot
                if n_comp >= 3:
                    fig3d = px.scatter_3d(
                        pca_df, x="PC1", y="PC2", z="PC3",
                        title="PCA – First 3 Principal Components (Training Data)",
                        opacity=0.7,
                    )
                    st.plotly_chart(fig3d, use_container_width=True)

                # 2D plot
                fig2, ax2 = plt.subplots(figsize=(9, 6))
                ax2.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.7)
                ax2.set(xlabel="PC1", ylabel="PC2", title="PCA 2D – Training Data")
                ax2.grid(True, linestyle="--", alpha=0.6)
                st.pyplot(fig2)

                # Feature loadings
                loadings = pd.DataFrame(
                    pca.components_.T,
                    columns=[f"PC{i+1}" for i in range(n_comp)],
                    index=pca_features,
                )
                st.write("**Feature Loadings:**")
                st.dataframe(loadings, use_container_width=True)

            # Apply PCA to test data
            if "pca_model" in st.session_state:
                if st.button("▶️ Apply PCA on Test Data", key="pca_test_btn"):
                    X_test = st.session_state["test_data"][
                        st.session_state["pca_features"]
                    ].copy()
                    if X_test.isnull().sum().sum() > 0:
                        X_test = X_test.fillna(X_test.mean())

                    X_test_scaled = st.session_state["pca_scaler"].transform(X_test)
                    test_pca_res  = st.session_state["pca_model"].transform(X_test_scaled)
                    n_test_comp   = test_pca_res.shape[1]
                    test_pca_df   = pd.DataFrame(
                        test_pca_res, columns=[f"PC{i+1}" for i in range(n_test_comp)]
                    )
                    st.session_state["test_pca_df"] = test_pca_df

                    st.write("PCA Results for Test Data (first 5 rows):")
                    st.dataframe(test_pca_df.head(), use_container_width=True)

                    if "train_pca_df" in st.session_state and n_test_comp >= 3:
                        comp_fig = go.Figure([
                            go.Scatter3d(
                                x=st.session_state["train_pca_df"]["PC1"],
                                y=st.session_state["train_pca_df"]["PC2"],
                                z=st.session_state["train_pca_df"]["PC3"],
                                mode="markers",
                                marker=dict(size=4, color="blue", opacity=0.7),
                                name="Training",
                            ),
                            go.Scatter3d(
                                x=test_pca_df["PC1"],
                                y=test_pca_df["PC2"],
                                z=test_pca_df["PC3"],
                                mode="markers",
                                marker=dict(size=4, color="red", opacity=0.7),
                                name="Test",
                            ),
                        ])
                        comp_fig.update_layout(
                            title="Train vs Test PCA (3D)", width=800, height=800
                        )
                        st.plotly_chart(comp_fig, use_container_width=True)

    st.markdown("---")

    # ════════════════════════════════════════
    #  SECTION C — PCA + CLUSTERING PIPELINE
    # ════════════════════════════════════════
    st.write("### 🚀 PCA + Clustering Pipeline")

    with st.expander("📖 Why combine PCA and Clustering?", expanded=False):
        st.markdown("""
**High-dimensional data problem** — Clustering algorithms (K-Means, DBSCAN) suffer from the
*curse of dimensionality*: noisy and irrelevant features distort distance metrics.

**PCA solves this by** keeping maximum variance and removing redundant features.

**Mathematical insight — Z = X · W**

| Symbol | Meaning |
|--------|---------|
| **X** | Original data matrix |
| **W** | Eigenvectors (principal components) |
| **Z** | Reduced data — clustering applied here |

**When to use:**

| Situation | Recommended? |
|-----------|-------------|
| High-dimensional data | ✅ Yes |
| Noisy dataset | ✅ Yes |
| Visualisation needed | ✅ Yes |
| Already low-dimensional | ❌ Not necessary |

⚠️ **Don't over-reduce** — use `PCA(n_components=0.95)` to retain 95% variance automatically.

⚠️ **PCA is linear** — for non-linear structure consider t-SNE or UMAP instead.
        """)

    if "train_pca_df" not in st.session_state:
        st.info("▲ Run PCA on Training Data (Section B above) first to unlock clustering.")
    else:
        pca_df      = st.session_state["train_pca_df"].copy()
        X_pca       = pca_df.values
        actual_comp = X_pca.shape[1]

        st.write("#### Select Clustering Algorithm")
        algo = st.selectbox("Algorithm:", ["K-Means", "DBSCAN"], key="cluster_algo_sel")

        # K-Means controls
        if algo == "K-Means":
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                n_clusters = st.slider("Number of clusters (k):", 2, 15, 3, key="kmeans_k")
            with col_k2:
                show_elbow = st.checkbox("Show Elbow Curve", value=True, key="elbow_chk")

            if show_elbow:
                inertias = []
                k_range  = range(2, min(16, len(X_pca)))
                for k in k_range:
                    inertias.append(
                        KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_pca).inertia_
                    )
                fig_elb, ax_e = plt.subplots(figsize=(8, 3))
                ax_e.plot(list(k_range), inertias, "bo-", linewidth=2)
                ax_e.axvline(x=n_clusters, color="red", linestyle="--",
                             label=f"Selected k={n_clusters}")
                ax_e.set(xlabel="k", ylabel="Inertia", title="Elbow Curve — Optimal k")
                ax_e.legend()
                ax_e.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig_elb)

        # DBSCAN controls
        else:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                eps_val     = st.slider("ε (eps):", 0.1, 5.0, 0.5, 0.1, key="dbscan_eps")
            with col_d2:
                min_samples = st.slider("min_samples:", 2, 20, 5, key="dbscan_min")

        if st.button("▶️ Run Clustering", key="run_cluster_btn"):
            if algo == "K-Means":
                cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            else:
                cluster_model = DBSCAN(eps=eps_val, min_samples=min_samples)

            labels = cluster_model.fit_predict(X_pca)
            st.session_state["cluster_labels"]    = labels
            st.session_state["cluster_algo_used"] = algo
            if algo == "K-Means":
                st.session_state["cluster_model"] = cluster_model
            else:
                n_found = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                st.info(f"DBSCAN found **{n_found}** cluster(s) and **{n_noise}** noise point(s).")

        # Results
        if "cluster_labels" in st.session_state:
            labels      = st.session_state["cluster_labels"]
            algo_used   = st.session_state["cluster_algo_used"]
            pca_df_res  = st.session_state["train_pca_df"].copy()
            X_pca_res   = pca_df_res.values
            comp_count  = X_pca_res.shape[1]
            pca_df_res["Cluster"] = labels.astype(str)

            n_found     = len([l for l in set(labels) if l != -1])
            noise_count = list(labels).count(-1)

            m1, m2, m3 = st.columns(3)
            m1.metric("Clusters Found", n_found)
            m2.metric("Noise Points",   noise_count)
            m3.metric("Total Samples",  len(labels))

            t1, t2, t3, t4 = st.tabs(
                ["2D Scatter", "3D Scatter", "Cluster Distribution", "Cluster Stats"]
            )

            # Colour palette — one distinct colour per cluster
            unique_labels   = sorted(set(labels))
            n_unique        = len(unique_labels)
            PALETTE_HEX     = [
                "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
                "#E377C2", "#17BECF", "#BCBD22", "#9467BD", "#D62728",
            ]
            # Map each label to a colour; noise (-1) always grey
            label_to_color  = {}
            colour_idx      = 0
            for lbl in unique_labels:
                if lbl == -1:
                    label_to_color[lbl] = "#AAAAAA"
                else:
                    label_to_color[lbl] = PALETTE_HEX[colour_idx % len(PALETTE_HEX)]
                    colour_idx += 1
            point_colors = [label_to_color[l] for l in labels]

            with t1:
                fig2d, ax2d = plt.subplots(figsize=(9, 6))
                for lbl in unique_labels:
                    mask = labels == lbl
                    lbl_name = f"Cluster {lbl}" if lbl != -1 else "Noise"
                    ax2d.scatter(
                        X_pca_res[mask, 0], X_pca_res[mask, 1],
                        color=label_to_color[lbl], alpha=0.80,
                        edgecolors="white", linewidths=0.4, s=65, label=lbl_name,
                    )
                if algo_used == "K-Means" and "cluster_model" in st.session_state:
                    centres = st.session_state["cluster_model"].cluster_centers_
                    ax2d.scatter(centres[:, 0], centres[:, 1],
                                 c="black", marker="X", s=220, zorder=6, label="Centroids")
                ax2d.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left",
                            frameon=True, fontsize=9)
                ax2d.set(xlabel="PC1", ylabel="PC2",
                         title=f"PCA + {algo_used} — 2D Scatter")
                ax2d.grid(True, linestyle="--", alpha=0.35)
                ax2d.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig2d)

            with t2:
                if comp_count >= 3:
                    fig3d = px.scatter_3d(
                        pca_df_res, x="PC1", y="PC2", z="PC3",
                        color="Cluster",
                        title=f"PCA + {algo_used} — 3D Scatter",
                        opacity=0.85,
                        color_discrete_sequence=PALETTE_HEX,
                    )
                    fig3d.update_traces(marker=dict(size=4, line=dict(width=0.3, color="white")))
                    fig3d.update_layout(
                        width=800, height=700,
                        scene=dict(
                            xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
                            bgcolor="rgb(245,245,250)",
                        ),
                        legend_title_text="Cluster",
                    )
                    st.plotly_chart(fig3d, use_container_width=True)
                else:
                    st.info("3D scatter requires ≥ 3 PCA components. "
                            "Increase the component slider in Section B above.")

            with t3:
                label_names = [f"Cluster {l}" if l != -1 else "Noise" for l in labels]
                count_df    = pd.Series(label_names).value_counts().reset_index()
                count_df.columns = ["Cluster", "Count"]
                bar_colors  = [
                    label_to_color[-1]
                    if row["Cluster"] == "Noise"
                    else PALETTE_HEX[i % len(PALETTE_HEX)]
                    for i, row in count_df.iterrows()
                ]
                fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
                bars = ax_bar.bar(count_df["Cluster"], count_df["Count"],
                                  color=bar_colors, edgecolor="white", linewidth=0.8)
                for bar, val in zip(bars, count_df["Count"]):
                    ax_bar.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3, str(val),
                        ha="center", va="bottom", fontsize=11, fontweight="bold",
                    )
                ax_bar.set(xlabel="Cluster", ylabel="Number of Points",
                           title="Cluster Size Distribution")
                ax_bar.spines[["top", "right"]].set_visible(False)
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                st.pyplot(fig_bar)

            with t4:
                # ✅ FIX: use train_data (same length as labels) not full data
                train_df             = st.session_state["train_data"].copy().reset_index(drop=True)
                train_df["Cluster"]  = labels          # lengths now always match
                feat_cols = st.session_state.get("pca_features", numeric_columns.tolist())
                for lbl in unique_labels:
                    label_name = f"Cluster {lbl}" if lbl != -1 else "🔴 Noise Points"
                    subset     = train_df[train_df["Cluster"] == lbl][feat_cols]
                    with st.expander(f"**{label_name}** — {len(subset)} samples"):
                        st.dataframe(subset.describe().T, use_container_width=True)

            # ── Download (training rows + cluster label + PC scores) ──
            export_df = st.session_state["train_data"].copy().reset_index(drop=True)
            export_df["Cluster_Label"] = labels
            for i in range(comp_count):
                export_df[f"PC{i+1}"] = X_pca_res[:, i]
            st.download_button(
                "⬇️ Download Training Data with Cluster Labels & PCA Scores",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="pca_clustered_data.csv",
                mime="text/csv",
            )

    st.markdown("---")

    # ════════════════════════════════════════
    #  SECTION D — OUTLIER DETECTION & HANDLING
    # ════════════════════════════════════════
    st.write("### 📦 Outlier Detection & Handling")

    # ── Dataset selector ─────────────────────
    data_source = st.radio(
        "Dataset to analyse:",
        ["Full Dataset", "Training Data", "Test Data"],
        disabled="train_data" not in st.session_state,
        key="outlier_source",
        horizontal=True,
    )
    display_data = {
        "Full Dataset":  data,
        "Training Data": st.session_state.get("train_data", data),
        "Test Data":     st.session_state.get("test_data", data),
    }[data_source].copy()

    # ── Column selector ──────────────────────
    od_cols = st.multiselect(
        "Select numeric columns:", numeric_columns, key="outlier_cols"
    )
    if not od_cols:
        st.info("Select one or more numeric columns to begin outlier analysis.")
        return

    # ════════════════════════════════════
    # STEP 1 — DETECTION
    # ════════════════════════════════════
    st.write("#### 🔍 Step 1 — Detect Outliers")

    detect_method = st.radio(
        "Detection method:",
        ["Z-Score  (|Z| > threshold)", "IQR  (Tukey fences)"],
        horizontal=True,
        key="detect_method",
    )

    if "Z-Score" in detect_method:
        z_thresh = st.slider("Z-score threshold:", 1.5, 5.0, 3.0, 0.1, key="z_thresh")
    else:
        iqr_mult = st.slider("IQR multiplier (k):", 1.0, 3.0, 1.5, 0.25, key="iqr_mult")

    # Compute bounds & flag outliers per column
    bounds_info = {}
    for col in od_cols:
        series = display_data[col].dropna()
        if "Z-Score" in detect_method:
            z_scores = (display_data[col] - series.mean()) / series.std()
            mask     = z_scores.abs() > z_thresh
            lower    = series.mean() - z_thresh * series.std()
            upper    = series.mean() + z_thresh * series.std()
        else:
            Q1, Q3   = series.quantile(0.25), series.quantile(0.75)
            IQR      = Q3 - Q1
            lower    = Q1 - iqr_mult * IQR
            upper    = Q3 + iqr_mult * IQR
            mask     = (display_data[col] < lower) | (display_data[col] > upper)
        bounds_info[col] = {"lower": lower, "upper": upper, "mask": mask,
                            "count": int(mask.sum()), "pct": mask.mean() * 100}

    # Summary table
    summary_rows = [
        {
            "Column":          col,
            "Outlier Count":   v["count"],
            "Outlier %":       f"{v['pct']:.1f}%",
            "Lower Bound":     f"{v['lower']:.3f}",
            "Upper Bound":     f"{v['upper']:.3f}",
        }
        for col, v in bounds_info.items()
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # Visualisation — boxplot with outlier highlights
    st.write("##### Boxplot with Outlier Highlights")
    n_cols_plot = len(od_cols)
    fig_bp, axes_bp = plt.subplots(
        1, n_cols_plot, figsize=(max(5, 4 * n_cols_plot), 5), squeeze=False
    )
    for ax, col in zip(axes_bp[0], od_cols):
        v    = bounds_info[col]
        mask = v["mask"]
        # Box
        ax.boxplot(display_data[col].dropna(), patch_artist=True,
                   boxprops=dict(facecolor="#AED6F1", color="#2874A6"),
                   medianprops=dict(color="#E74C3C", linewidth=2),
                   whiskerprops=dict(color="#2874A6"),
                   capprops=dict(color="#2874A6"),
                   flierprops=dict(marker="o", markerfacecolor="#E74C3C",
                                   markersize=5, alpha=0.5))
        # Bound lines
        ax.axhline(v["lower"], color="#E67E22", linestyle="--", linewidth=1.2,
                   label=f"Lower ({v['lower']:.2f})")
        ax.axhline(v["upper"], color="#E67E22", linestyle="--", linewidth=1.2,
                   label=f"Upper ({v['upper']:.2f})")
        ax.set_title(f"{col}\n({v['count']} outliers, {v['pct']:.1f}%)", fontsize=10)
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_bp)

    st.markdown("---")

    # ════════════════════════════════════
    # STEP 2 — HANDLING METHOD
    # ════════════════════════════════════
    st.write("#### 🛠️ Step 2 — Handle Outliers")

    with st.expander("📖 Method guide", expanded=False):
        guide_df = pd.DataFrame({
            "Method": [
                "🔥 Removal (Trimming)", "✂️ Capping (Winsorization)",
                "🔄 Log Transform", "🔄 Square-Root Transform",
                "📏 Robust Scaling", "🤖 Isolation Forest",
                "🤖 DBSCAN", "📊 Imputation (Mean/Median/Mode)",
                "📉 Binning", "🚫 Treat Separately",
            ],
            "Best for": [
                "Data-entry errors, few outliers",
                "Keep data size, limit extreme impact",
                "Right-skewed data (finance, sensor)",
                "Moderate right skew, no zeros",
                "Many outliers, need scaled features",
                "Automatic multivariate anomaly detection",
                "Cluster-based, unknown outlier count",
                "Preserve size, replace extreme values",
                "Model doesn't need exact values",
                "Fraud / anomaly detection use-cases",
            ],
            "Risk": [
                "Loses data", "Distorts distribution tails",
                "Undefined for zero/negative values", "Less aggressive than log",
                "Does not remove outliers", "Needs tuning (contamination)",
                "Sensitive to eps parameter", "Can pull mean/median toward outliers",
                "Loses numeric precision", "Requires separate modelling",
            ],
        })
        st.dataframe(guide_df, use_container_width=True)

    handle_method = st.selectbox("Select handling method:", [
        "🔥 Removal (Trimming)",
        "✂️ Capping (Winsorization)",
        "🔄 Log Transform",
        "🔄 Square-Root Transform",
        "📏 Robust Scaling",
        "🤖 Isolation Forest",
        "🤖 DBSCAN",
        "📊 Imputation – Mean",
        "📊 Imputation – Median",
        "📊 Imputation – Mode",
        "📉 Binning",
        "🚫 Treat Separately (flag column)",
    ], key="handle_method")

    # Method-specific params
    if handle_method == "🤖 Isolation Forest":
        contamination = st.slider(
            "Contamination (expected outlier fraction):",
            0.01, 0.5, 0.05, 0.01, key="iso_cont"
        )
    if handle_method == "🤖 DBSCAN":
        db_eps  = st.slider("ε (eps):", 0.1, 5.0, 0.5, 0.1, key="od_dbscan_eps")
        db_mins = st.slider("min_samples:", 2, 20, 5, key="od_dbscan_min")
    if handle_method == "📉 Binning":
        n_bins = st.slider("Number of bins:", 3, 20, 5, key="od_bins")

    if st.button("▶️ Apply Outlier Handling", key="apply_outlier_btn"):
        handled = display_data.copy()
        method_log = []

        # ── 1. Removal ────────────────────────
        if handle_method == "🔥 Removal (Trimming)":
            before = len(handled)
            for col in od_cols:
                v = bounds_info[col]
                handled = handled[(handled[col] >= v["lower"]) & (handled[col] <= v["upper"])]
            after = len(handled)
            method_log.append(f"Removed {before - after} rows containing outliers.")

        # ── 2. Capping ────────────────────────
        elif handle_method == "✂️ Capping (Winsorization)":
            for col in od_cols:
                v = bounds_info[col]
                n_capped = int((handled[col] > v["upper"]).sum() +
                               (handled[col] < v["lower"]).sum())
                handled[col] = np.where(handled[col] > v["upper"], v["upper"], handled[col])
                handled[col] = np.where(handled[col] < v["lower"], v["lower"], handled[col])
                method_log.append(f"{col}: capped {n_capped} value(s).")

        # ── 3. Log Transform ──────────────────
        elif handle_method == "🔄 Log Transform":
            for col in od_cols:
                min_val = handled[col].min()
                shift   = abs(min_val) + 1 if min_val <= 0 else 0
                handled[col] = np.log(handled[col] + shift)
                method_log.append(f"{col}: log(x + {shift:.3f}) applied.")

        # ── 4. Square-Root Transform ──────────
        elif handle_method == "🔄 Square-Root Transform":
            for col in od_cols:
                min_val = handled[col].min()
                shift   = abs(min_val) if min_val < 0 else 0
                handled[col] = np.sqrt(handled[col] + shift)
                method_log.append(f"{col}: sqrt(x + {shift:.3f}) applied.")

        # ── 5. Robust Scaling ─────────────────
        elif handle_method == "📏 Robust Scaling":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            handled[od_cols] = scaler.fit_transform(handled[od_cols])
            method_log.append(f"RobustScaler applied to: {', '.join(od_cols)}.")

        # ── 6. Isolation Forest ───────────────
        elif handle_method == "🤖 Isolation Forest":
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination=contamination, random_state=42)
            preds = iso.fit_predict(handled[od_cols].fillna(handled[od_cols].mean()))
            n_flagged = int((preds == -1).sum())
            handled["IsoForest_Outlier"] = np.where(preds == -1, "Outlier", "Normal")
            method_log.append(
                f"Isolation Forest flagged {n_flagged} row(s) as outliers "
                f"(contamination={contamination}). Column 'IsoForest_Outlier' added."
            )

        # ── 7. DBSCAN ─────────────────────────
        elif handle_method == "🤖 DBSCAN":
            db = DBSCAN(eps=db_eps, min_samples=db_mins)
            preds = db.fit_predict(
                StandardScaler().fit_transform(handled[od_cols].fillna(handled[od_cols].mean()))
            )
            n_noise = int((preds == -1).sum())
            handled["DBSCAN_Label"] = preds
            handled["DBSCAN_Outlier"] = np.where(preds == -1, "Outlier", "Normal")
            method_log.append(
                f"DBSCAN flagged {n_noise} noise point(s) as outliers. "
                f"Columns 'DBSCAN_Label' and 'DBSCAN_Outlier' added."
            )

        # ── 8-10. Imputation ──────────────────
        elif handle_method in ("📊 Imputation – Mean",
                               "📊 Imputation – Median",
                               "📊 Imputation – Mode"):
            for col in od_cols:
                v = bounds_info[col]
                if "Mean"   in handle_method: fill = handled[col].mean()
                elif "Median" in handle_method: fill = handled[col].median()
                else:                           fill = handled[col].mode().iloc[0]
                n_replaced = int(bounds_info[col]["mask"].sum())
                handled.loc[bounds_info[col]["mask"], col] = fill
                method_log.append(
                    f"{col}: replaced {n_replaced} outlier(s) with "
                    f"{'mean' if 'Mean' in handle_method else 'median' if 'Median' in handle_method else 'mode'} "
                    f"({fill:.3f})."
                )

        # ── 11. Binning ───────────────────────
        elif handle_method == "📉 Binning":
            for col in od_cols:
                handled[f"{col}_binned"] = pd.cut(
                    handled[col], bins=n_bins,
                    labels=[f"B{i+1}" for i in range(n_bins)]
                )
                method_log.append(f"{col}: binned into {n_bins} equal-width bins → '{col}_binned' added.")

        # ── 12. Treat Separately ──────────────
        elif handle_method == "🚫 Treat Separately (flag column)":
            for col in od_cols:
                flag_col = f"{col}_outlier_flag"
                handled[flag_col] = bounds_info[col]["mask"].astype(int)
                method_log.append(
                    f"{col}: '{flag_col}' added "
                    f"(1 = outlier, {bounds_info[col]['count']} flagged)."
                )

        # ── Store & report ────────────────────
        st.session_state["outlier_handled_data"] = handled

        st.success("✅ Outlier handling applied successfully!")
        for log in method_log:
            st.write(f"  • {log}")

        col_a, col_b = st.columns(2)
        col_a.metric("Rows Before", len(display_data))
        col_b.metric("Rows After",  len(handled))

    # ── Results tabs ─────────────────────────
    if "outlier_handled_data" in st.session_state:
        handled = st.session_state["outlier_handled_data"]

        st.write("##### Before vs After Comparison")
        rt1, rt2, rt3 = st.tabs(["Boxplot Comparison", "Distribution", "Handled Data Preview"])

        # common columns that are still numeric in both
        common_num = [c for c in od_cols if c in handled.columns
                      and pd.api.types.is_numeric_dtype(handled[c])]

        with rt1:
            if common_num:
                fig_cmp, axes_cmp = plt.subplots(
                    2, len(common_num),
                    figsize=(max(5, 4 * len(common_num)), 8),
                    squeeze=False
                )
                for j, col in enumerate(common_num):
                    # Before
                    axes_cmp[0][j].boxplot(
                        display_data[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor="#FADBD8"),
                        medianprops=dict(color="#C0392B", linewidth=2),
                    )
                    axes_cmp[0][j].set_title(f"{col}\n(Before)", fontsize=9)
                    axes_cmp[0][j].spines[["top","right"]].set_visible(False)
                    # After
                    axes_cmp[1][j].boxplot(
                        handled[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor="#D5F5E3"),
                        medianprops=dict(color="#1E8449", linewidth=2),
                    )
                    axes_cmp[1][j].set_title(f"{col}\n(After)", fontsize=9)
                    axes_cmp[1][j].spines[["top","right"]].set_visible(False)
                plt.suptitle("Boxplot: Before (red) vs After (green)", fontsize=12, y=1.01)
                plt.tight_layout()
                st.pyplot(fig_cmp)
            else:
                st.info("No comparable numeric columns after transformation.")

        with rt2:
            if common_num:
                fig_dist, axes_dist = plt.subplots(
                    1, len(common_num),
                    figsize=(max(5, 4 * len(common_num)), 4),
                    squeeze=False
                )
                for j, col in enumerate(common_num):
                    axes_dist[0][j].hist(
                        display_data[col].dropna(), bins=30, alpha=0.55,
                        color="#E74C3C", label="Before", edgecolor="white"
                    )
                    axes_dist[0][j].hist(
                        handled[col].dropna(), bins=30, alpha=0.55,
                        color="#27AE60", label="After", edgecolor="white"
                    )
                    axes_dist[0][j].set_title(col, fontsize=9)
                    axes_dist[0][j].legend(fontsize=8)
                    axes_dist[0][j].spines[["top","right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_dist)

        with rt3:
            st.dataframe(handled.head(30), use_container_width=True)
            st.download_button(
                "⬇️ Download Handled Dataset",
                data=handled.to_csv(index=False).encode("utf-8"),
                file_name="outlier_handled_data.csv",
                mime="text/csv",
            )

        # Option to promote handled data as the working dataset
        st.markdown("---")
        if st.button("✅ Save as Active Dataset (use in all features)", key="promote_outlier"):
            st.session_state["cleaned_data"] = handled
            st.success("Saved! All features will now use the outlier-handled dataset.")
            st.rerun()


# ─────────────────────────────────────────────
#  FEATURE: DATA PROFILING REPORT
# ─────────────────────────────────────────────
def show_data_profiling(data: pd.DataFrame):
    st.subheader("📋 Data Profiling Report")
    st.caption("Auto-generated summary of every column — distributions, quality, correlations.")

    numeric_cols     = data.select_dtypes(include="number").columns.tolist()
    categorical_cols = data.select_dtypes(include="object").columns.tolist()
    total_cells      = data.shape[0] * data.shape[1]
    total_nulls      = data.isnull().sum().sum()
    duplicate_rows   = data.duplicated().sum()

    # ── Top-level KPIs ────────────────────────
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Rows",            data.shape[0])
    k2.metric("Columns",         data.shape[1])
    k3.metric("Numeric Cols",    len(numeric_cols))
    k4.metric("Categorical Cols",len(categorical_cols))
    k5.metric("Missing Cells",   f"{total_nulls} ({total_nulls/total_cells*100:.1f}%)")
    k6.metric("Duplicate Rows",  duplicate_rows)

    if duplicate_rows > 0:
        st.warning(f"⚠️ {duplicate_rows} duplicate row(s) found. Consider removing them.")

    st.markdown("---")

    # ── Per-column profile table ──────────────
    st.write("### 🗂️ Column-by-Column Profile")
    rows = []
    for col in data.columns:
        s          = data[col]
        null_c     = s.isnull().sum()
        unique_c   = s.nunique()
        dtype_str  = str(s.dtype)
        if pd.api.types.is_numeric_dtype(s):
            skewness   = round(s.skew(), 3)
            kurt       = round(s.kurtosis(), 3)
            if   abs(skewness) < 0.5: shape = "Normal"
            elif abs(skewness) < 1.0: shape = "Moderate skew"
            else:                     shape = "High skew"
            rows.append({
                "Column": col, "Type": dtype_str,
                "Non-Null": data.shape[0] - null_c,
                "Null %": f"{null_c/len(s)*100:.1f}%",
                "Unique": unique_c,
                "Mean": round(s.mean(), 4),
                "Std":  round(s.std(),  4),
                "Min":  round(s.min(),  4),
                "Max":  round(s.max(),  4),
                "Skewness": skewness,
                "Shape": shape,
                "Kurtosis": kurt,
            })
        else:
            top_val  = s.value_counts().index[0] if unique_c > 0 else "N/A"
            top_freq = s.value_counts().iloc[0]  if unique_c > 0 else 0
            rows.append({
                "Column": col, "Type": dtype_str,
                "Non-Null": data.shape[0] - null_c,
                "Null %": f"{null_c/len(s)*100:.1f}%",
                "Unique": unique_c,
                "Mean": "—", "Std": "—", "Min": "—", "Max": "—",
                "Skewness": "—", "Shape": f"Top: {top_val} ({top_freq}×)",
                "Kurtosis": "—",
            })

    profile_df = pd.DataFrame(rows)
    st.dataframe(profile_df, use_container_width=True)

    st.markdown("---")

    # ── High-correlation warning ──────────────
    if len(numeric_cols) >= 2:
        st.write("### 🔗 High-Correlation Pairs (|r| > 0.85)")
        corr_matrix = data[numeric_cols].corr().abs()
        upper       = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr = (
            upper.stack()
                 .reset_index()
                 .rename(columns={"level_0":"Feature A","level_1":"Feature B", 0:"Correlation"})
        )
        high_corr = high_corr[high_corr["Correlation"] > 0.85].sort_values(
            "Correlation", ascending=False
        )
        if not high_corr.empty:
            st.warning(
                f"⚠️ {len(high_corr)} highly correlated pair(s) found. "
                "Consider dropping one from each pair before ML."
            )
            st.dataframe(
                high_corr.style.background_gradient(subset=["Correlation"], cmap="Reds"),
                use_container_width=True,
            )
        else:
            st.success("✅ No feature pairs with |r| > 0.85 found.")

    st.markdown("---")

    # ── Class imbalance check ─────────────────
    st.write("### ⚖️ Potential Target Column Imbalance Check")
    for col in data.columns:
        if data[col].nunique() < 20:
            vc      = data[col].value_counts(normalize=True) * 100
            min_pct = vc.min()
            if min_pct < 15:
                st.warning(
                    f"**`{col}`** may be imbalanced — "
                    f"minority class is only {min_pct:.1f}% of data."
                )
                fig_ib, ax_ib = plt.subplots(figsize=(6, 3))
                ax_ib.bar(vc.index.astype(str), vc.values,
                          color=["#E74C3C" if v == vc.min() else "#3498DB" for v in vc.values],
                          edgecolor="white")
                ax_ib.set(xlabel=col, ylabel="%", title=f"Class Distribution — {col}")
                ax_ib.spines[["top","right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_ib)

    st.markdown("---")

    # ── Memory usage ──────────────────────────
    st.write("### 💾 Memory Usage")
    mem = data.memory_usage(deep=True)
    mem_df = pd.DataFrame({
        "Column":       ["(Index)"] + data.columns.tolist(),
        "Memory (KB)":  (mem.values / 1024).round(3),
    })
    total_kb = mem.sum() / 1024
    st.dataframe(mem_df, use_container_width=True)
    st.info(f"Total memory usage: **{total_kb:.2f} KB** ({total_kb/1024:.3f} MB)")

    # ── Distributions grid ────────────────────
    if numeric_cols:
        st.markdown("---")
        st.write("### 📊 Distribution Grid (Numeric Columns)")
        n      = len(numeric_cols)
        ncols  = 4
        nrows  = (n + ncols - 1) // ncols
        fig_dg, axes_dg = plt.subplots(
            nrows, ncols, figsize=(ncols * 3.5, nrows * 3), squeeze=False
        )
        for idx, col in enumerate(numeric_cols):
            ax = axes_dg[idx // ncols][idx % ncols]
            ax.hist(data[col].dropna(), bins=25, color="#3498DB",
                    edgecolor="white", alpha=0.85)
            skv = data[col].skew()
            ax.set_title(f"{col}\nskew={skv:.2f}", fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
            ax.tick_params(labelsize=7)
        # hide unused axes
        for idx in range(n, nrows * ncols):
            axes_dg[idx // ncols][idx % ncols].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_dg)


# ─────────────────────────────────────────────
#  FEATURE: FEATURE ENGINEERING
# ─────────────────────────────────────────────
def show_feature_engineering(data: pd.DataFrame):
    st.subheader("⚙️ Feature Engineering")
    st.caption(
        "Build new features from existing ones. "
        "Click **Apply & Save** at the bottom to push changes to the active dataset."
    )

    numeric_cols     = data.select_dtypes(include="number").columns.tolist()
    categorical_cols = data.select_dtypes(include="object").columns.tolist()
    all_cols         = data.columns.tolist()

    working = data.copy()   # all transforms accumulate here

    # ════════════════════════════════════════
    #  1. ENCODE CATEGORICAL COLUMNS
    # ════════════════════════════════════════
    st.write("### 1️⃣ Encode Categorical Columns")
    if not categorical_cols:
        st.info("No categorical columns found.")
    else:
        enc_cols   = st.multiselect("Select columns to encode:", categorical_cols, key="fe_enc_cols")
        enc_method = st.radio(
            "Encoding method:",
            ["One-Hot Encoding", "Label Encoding", "Ordinal (manual order)"],
            horizontal=True, key="fe_enc_method",
        )

        if enc_cols:
            ordinal_orders = {}
            if enc_method == "Ordinal (manual order)":
                for col in enc_cols:
                    uniq = data[col].dropna().unique().tolist()
                    order_input = st.text_input(
                        f"Order for `{col}` (comma-separated, low→high):",
                        value=", ".join(str(u) for u in uniq),
                        key=f"fe_ord_{col}",
                    )
                    ordinal_orders[col] = [v.strip() for v in order_input.split(",")]

            if st.button("Apply Encoding", key="fe_enc_btn"):
                for col in enc_cols:
                    if enc_method == "One-Hot Encoding":
                        dummies = pd.get_dummies(working[col], prefix=col, drop_first=False)
                        working = pd.concat([working.drop(columns=[col]), dummies], axis=1)
                    elif enc_method == "Label Encoding":
                        le = LabelEncoder()
                        working[f"{col}_encoded"] = le.fit_transform(working[col].astype(str))
                    else:  # Ordinal
                        order  = ordinal_orders.get(col, [])
                        mapper = {v: i for i, v in enumerate(order)}
                        working[f"{col}_ordinal"] = working[col].map(mapper)
                st.session_state["fe_working"] = working
                st.success(f"✅ Encoding applied to: {', '.join(enc_cols)}")

    st.markdown("---")

    # ════════════════════════════════════════
    #  2. POLYNOMIAL / INTERACTION FEATURES
    # ════════════════════════════════════════
    st.write("### 2️⃣ Polynomial & Interaction Features")
    if not numeric_cols:
        st.info("No numeric columns available.")
    else:
        poly_cols   = st.multiselect("Select numeric columns:", numeric_cols, key="fe_poly_cols")
        poly_degree = st.slider("Polynomial degree:", 2, 4, 2, key="fe_poly_deg")
        poly_inter  = st.checkbox("Include interaction terms only (no powers)", key="fe_poly_inter")

        if poly_cols and st.button("Apply Polynomial Features", key="fe_poly_btn"):
            pf  = PolynomialFeatures(
                degree=poly_degree,
                interaction_only=poly_inter,
                include_bias=False,
            )
            arr      = working[poly_cols].fillna(0).values
            poly_arr = pf.fit_transform(arr)          # fit first
            names    = pf.get_feature_names_out(poly_cols)  # then get names
            poly_df  = pd.DataFrame(poly_arr, columns=names, index=working.index)
            # only add NEW columns (not the originals repeated)
            new_cols = [c for c in poly_df.columns if c not in working.columns]
            working  = pd.concat([working, poly_df[new_cols]], axis=1)
            st.session_state["fe_working"] = working
            st.success(f"✅ Added {len(new_cols)} new polynomial/interaction columns.")

    st.markdown("---")

    # ════════════════════════════════════════
    #  3. MANUAL INTERACTION TERMS
    # ════════════════════════════════════════
    st.write("### 3️⃣ Manual Interaction Terms (col_A × col_B)")
    if len(numeric_cols) >= 2:
        ic1, ic2 = st.columns(2)
        with ic1:
            col_a = st.selectbox("Column A:", numeric_cols, key="fe_ia")
        with ic2:
            col_b = st.selectbox("Column B:", [c for c in numeric_cols if c != col_a],
                                 key="fe_ib")
        interact_op = st.radio(
            "Operation:", ["Multiply (A × B)", "Divide (A / B)", "Add (A + B)", "Subtract (A − B)"],
            horizontal=True, key="fe_iop",
        )
        if st.button("Add Interaction Feature", key="fe_inter_btn"):
            if   interact_op == "Multiply (A × B)": working[f"{col_a}_x_{col_b}"] = working[col_a] * working[col_b]
            elif interact_op == "Divide (A / B)":   working[f"{col_a}_div_{col_b}"] = working[col_a] / working[col_b].replace(0, np.nan)
            elif interact_op == "Add (A + B)":      working[f"{col_a}_plus_{col_b}"] = working[col_a] + working[col_b]
            else:                                   working[f"{col_a}_minus_{col_b}"] = working[col_a] - working[col_b]
            st.session_state["fe_working"] = working
            st.success(f"✅ Interaction column added.")

    st.markdown("---")

    # ════════════════════════════════════════
    #  4. MATHEMATICAL TRANSFORMATIONS
    # ════════════════════════════════════════
    st.write("### 4️⃣ Mathematical Transformations")
    trans_cols = st.multiselect("Select columns to transform:", numeric_cols, key="fe_trans_cols")
    trans_op   = st.selectbox("Transformation:", [
        "Log (log(x+1))", "Square Root (√x)", "Square (x²)",
        "Reciprocal (1/x)", "Z-Score Normalise", "Min-Max Scale (0–1)",
    ], key="fe_trans_op")

    if trans_cols and st.button("Apply Transformation", key="fe_trans_btn"):
        for col in trans_cols:
            if   trans_op == "Log (log(x+1))":
                shift = abs(working[col].min()) + 1 if working[col].min() <= 0 else 0
                working[f"{col}_log"] = np.log(working[col] + shift)
            elif trans_op == "Square Root (√x)":
                shift = abs(working[col].min()) if working[col].min() < 0 else 0
                working[f"{col}_sqrt"] = np.sqrt(working[col] + shift)
            elif trans_op == "Square (x²)":
                working[f"{col}_sq"]   = working[col] ** 2
            elif trans_op == "Reciprocal (1/x)":
                working[f"{col}_recip"] = 1 / working[col].replace(0, np.nan)
            elif trans_op == "Z-Score Normalise":
                working[f"{col}_zscore"] = (working[col] - working[col].mean()) / working[col].std()
            elif trans_op == "Min-Max Scale (0–1)":
                mn, mx = working[col].min(), working[col].max()
                working[f"{col}_minmax"] = (working[col] - mn) / (mx - mn + 1e-9)
        st.session_state["fe_working"] = working
        st.success(f"✅ Transformation applied to: {', '.join(trans_cols)}")

    st.markdown("---")

    # ════════════════════════════════════════
    #  5. BINNING
    # ════════════════════════════════════════
    st.write("### 5️⃣ Binning (Continuous → Categorical)")
    bin_col    = st.selectbox("Column to bin:", ["— select —"] + numeric_cols, key="fe_bin_col")
    bin_n      = st.slider("Number of bins:", 2, 20, 5, key="fe_bin_n")
    bin_labels = st.checkbox("Use custom labels (e.g. Low/Med/High)", key="fe_bin_lbl")
    if bin_labels:
        custom_labels_str = st.text_input(
            f"Enter {bin_n} comma-separated labels:",
            value=", ".join([f"B{i+1}" for i in range(bin_n)]),
            key="fe_bin_custom",
        )
        custom_labels = [v.strip() for v in custom_labels_str.split(",")][:bin_n]
    else:
        custom_labels = None

    if bin_col != "— select —" and st.button("Apply Binning", key="fe_bin_btn"):
        working[f"{bin_col}_binned"] = pd.cut(
            working[bin_col], bins=bin_n, labels=custom_labels
        )
        st.session_state["fe_working"] = working
        st.success(f"✅ `{bin_col}` binned into {bin_n} bins → `{bin_col}_binned`")

    st.markdown("---")

    # ════════════════════════════════════════
    #  6. DROP COLUMNS
    # ════════════════════════════════════════
    st.write("### 6️⃣ Drop Irrelevant / ID Columns")
    drop_cols = st.multiselect(
        "Select columns to drop (e.g. ID, name, index cols):",
        all_cols, key="fe_drop_cols",
    )
    if drop_cols and st.button("Drop Selected Columns", key="fe_drop_btn"):
        working = working.drop(columns=drop_cols, errors="ignore")
        st.session_state["fe_working"] = working
        st.success(f"✅ Dropped: {', '.join(drop_cols)}")

    st.markdown("---")

    # ── Pull latest working from session if buttons were clicked ──
    working = st.session_state.get("fe_working", data.copy())

    # ── Preview ───────────────────────────────
    st.write("### 📋 Current Dataset Preview")
    st.caption(f"Shape: {working.shape[0]} rows × {working.shape[1]} columns")
    new_col_names = [c for c in working.columns if c not in data.columns]
    if new_col_names:
        st.info(f"🆕 New columns added: {', '.join(new_col_names)}")
    st.dataframe(working.head(20), use_container_width=True)

    st.markdown("---")

    # ── Save as active dataset ────────────────
    if st.button("✅ Save Engineered Dataset as Active Data", type="primary", key="fe_save_btn"):
        st.session_state["cleaned_data"] = working
        if "fe_working" in st.session_state:
            del st.session_state["fe_working"]
        st.success(
            f"✅ Saved! Active dataset now has "
            f"{working.shape[0]} rows × {working.shape[1]} columns. "
            "All features (ML, Correlation, etc.) will use this data."
        )
        st.rerun()


# ─────────────────────────────────────────────
#  FEATURE: MULTI-MODEL COMPARISON
# ─────────────────────────────────────────────
def show_multi_model_comparison(data: pd.DataFrame):
    st.subheader("🏆 Multi-Model Comparison")
    st.caption("Train every algorithm at once and rank them side-by-side.")

    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        st.error("No numeric columns available.")
        return

    # ── Config ────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        target_col = st.selectbox("Target Column:", numeric_columns, key="mm_target")
    with c2:
        test_pct   = st.slider("Test size %:", 10, 40, 20, 5, key="mm_test")
    with c3:
        cv_folds   = st.slider("CV Folds:", 3, 10, 5, key="mm_cv")

    def detect(target):
        return "Classification" if target.dtype == "object" or target.nunique() < 20 else "Regression"

    problem_type = detect(data[target_col])
    st.info(f"**Detected Problem Type:** `{problem_type}`")

    # ── Model registry ────────────────────────
    if problem_type == "Classification":
        model_registry = {
            "K-Nearest Neighbors":      KNeighborsClassifier(),
            "Decision Tree":            DecisionTreeClassifier(random_state=42),
            "Random Forest":            RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting":        GradientBoostingClassifier(random_state=42),
            "SVM":                      SVC(probability=True),
            "MLP Neural Network":       MLPClassifier(max_iter=500, random_state=42),
        }
        score_label = "Accuracy"
        cv_scoring  = "accuracy"
    else:
        model_registry = {
            "Linear Regression":        LinearRegression(),
            "Ridge Regression":         Ridge(),
            "Lasso Regression":         Lasso(),
            "Decision Tree":            DecisionTreeRegressor(random_state=42),
            "Random Forest":            RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting":        GradientBoostingRegressor(random_state=42),
            "KNN Regressor":            KNeighborsRegressor(),
            "SVR":                      SVR(),
            "MLP Neural Network":       MLPRegressor(max_iter=500, random_state=42),
        }
        score_label = "R²"
        cv_scoring  = "r2"

    selected_models = st.multiselect(
        "Select models to compare (default = all):",
        list(model_registry.keys()),
        default=list(model_registry.keys()),
        key="mm_model_sel",
    )

    if st.button("▶️ Run Comparison", key="mm_run_btn", type="primary"):
        # Prepare data
        X_raw = data.drop(target_col, axis=1).select_dtypes(include="number")
        y_raw = data[target_col].dropna()
        X_raw = X_raw.loc[y_raw.index]
        imp   = SimpleImputer(strategy="mean")
        X_imp = pd.DataFrame(imp.fit_transform(X_raw), columns=X_raw.columns)
        scaler    = StandardScaler()
        X_scaled  = scaler.fit_transform(X_imp)
        le = None
        y_enc = y_raw.reset_index(drop=True)
        if problem_type == "Classification" and y_enc.dtype not in ["int64","int32"]:
            le    = LabelEncoder()
            y_enc = pd.Series(le.fit_transform(y_enc))

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y_enc, test_size=test_pct/100, random_state=42
        )

        results = []
        progress = st.progress(0, text="Training models…")
        n_models = len(selected_models)

        for i, name in enumerate(selected_models):
            model = model_registry[name]
            try:
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                cv_sc  = cross_val_score(model, X_scaled, y_enc,
                                         cv=cv_folds, scoring=cv_scoring)
                if problem_type == "Classification":
                    train_sc = accuracy_score(y_tr, model.predict(X_tr))
                    test_sc  = accuracy_score(y_te, y_pred)
                    extra    = {
                        "Precision": precision_score(y_te, y_pred, average="weighted", zero_division=0),
                        "Recall":    recall_score(y_te, y_pred, average="weighted", zero_division=0),
                        "F1":        f1_score(y_te, y_pred, average="weighted", zero_division=0),
                    }
                else:
                    train_sc = r2_score(y_tr, model.predict(X_tr))
                    test_sc  = r2_score(y_te, y_pred)
                    extra    = {
                        "MAE":  mean_absolute_error(y_te, y_pred),
                        "RMSE": mean_squared_error(y_te, y_pred) ** 0.5,
                    }
                results.append({
                    "Model":        name,
                    f"Train {score_label}": round(train_sc, 4),
                    f"Test {score_label}":  round(test_sc,  4),
                    "Overfit Gap":  round(train_sc - test_sc, 4),
                    f"CV Mean":     round(cv_sc.mean(), 4),
                    f"CV Std":      round(cv_sc.std(),  4),
                    **{k: round(v, 4) for k, v in extra.items()},
                    "_model_obj":   model,
                })
            except Exception as e:
                results.append({"Model": name, "Error": str(e)})

            progress.progress((i+1)/n_models, text=f"Trained: {name}")

        progress.empty()
        st.session_state["mm_results"] = results
        st.session_state["mm_problem"] = problem_type
        st.session_state["mm_score_label"] = score_label
        st.session_state["mm_X_tr"] = X_tr
        st.session_state["mm_X_te"] = X_te
        st.session_state["mm_y_tr"] = y_tr
        st.session_state["mm_y_te"] = y_te
        st.session_state["mm_scaler"] = scaler
        st.session_state["mm_imp"]    = imp

    # ── Results display ───────────────────────
    if "mm_results" not in st.session_state:
        return

    results      = st.session_state["mm_results"]
    problem_type = st.session_state["mm_problem"]
    score_label  = st.session_state["mm_score_label"]

    valid = [r for r in results if "Error" not in r]
    if not valid:
        st.error("All models failed. Check your data.")
        return

    df_res = pd.DataFrame([{k:v for k,v in r.items() if k != "_model_obj"} for r in valid])
    df_res = df_res.sort_values(f"Test {score_label}", ascending=False).reset_index(drop=True)
    df_res.insert(0, "Rank", range(1, len(df_res)+1))

    st.markdown("---")
    st.write("### 🏅 Leaderboard")

    # Colour the best row gold
    def highlight_best(row):
        return ["background-color: #FFF9C4; font-weight:bold"
                if row["Rank"] == 1 else "" for _ in row]

    st.dataframe(
        df_res.style.apply(highlight_best, axis=1)
              .background_gradient(subset=[f"Test {score_label}"], cmap="RdYlGn"),
        use_container_width=True,
    )

    # ── Bar chart ─────────────────────────────
    st.write("### 📊 Score Comparison")
    tab_bar, tab_radar, tab_overfit = st.tabs(
        ["Bar Chart", "Radar Chart", "Overfit Analysis"]
    )

    with tab_bar:
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        x       = np.arange(len(df_res))
        w       = 0.35
        colors_tr = ["#2980B9"] * len(df_res)
        colors_te = ["#27AE60" if i == 0 else "#E74C3C" for i in range(len(df_res))]
        b1 = ax_bar.bar(x - w/2, df_res[f"Train {score_label}"], w,
                        label=f"Train {score_label}", color=colors_tr, alpha=0.85)
        b2 = ax_bar.bar(x + w/2, df_res[f"Test {score_label}"],  w,
                        label=f"Test {score_label}",  color=colors_te, alpha=0.85)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(df_res["Model"], rotation=35, ha="right", fontsize=9)
        ax_bar.set(ylabel=score_label, title=f"Train vs Test {score_label} — All Models")
        ax_bar.legend()
        ax_bar.spines[["top","right"]].set_visible(False)
        # label bars
        for bar in list(b1) + list(b2):
            ax_bar.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{bar.get_height():.3f}",
                        ha="center", va="bottom", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig_bar)

    with tab_radar:
        if problem_type == "Classification":
            metrics = [f"Test {score_label}", "Precision", "Recall", "F1", "CV Mean"]
        else:
            metrics = [f"Test {score_label}", "CV Mean"]
        avail_metrics = [m for m in metrics if m in df_res.columns]
        if len(avail_metrics) >= 3:
            fig_r  = go.Figure()
            for _, row in df_res.iterrows():
                vals = [row[m] for m in avail_metrics]
                vals += [vals[0]]   # close the polygon
                fig_r.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=avail_metrics + [avail_metrics[0]],
                    name=row["Model"],
                    fill="toself",
                    opacity=0.5,
                ))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Metric Radar Chart — All Models",
                height=500,
            )
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("Need at least 3 metrics for radar chart (classification only).")

    with tab_overfit:
        fig_of, ax_of = plt.subplots(figsize=(10, 4))
        colors_of = ["#E74C3C" if g > 0.1 else "#27AE60" for g in df_res["Overfit Gap"]]
        ax_of.bar(df_res["Model"], df_res["Overfit Gap"],
                  color=colors_of, edgecolor="white")
        ax_of.axhline(0.1, color="orange", linestyle="--", linewidth=1.2,
                      label="Overfit threshold (0.10)")
        ax_of.axhline(0,   color="black",  linestyle="-",  linewidth=0.8)
        ax_of.set(xlabel="Model", ylabel="Train − Test Score",
                  title="Overfit Gap (lower is better)")
        ax_of.legend(fontsize=9)
        ax_of.spines[["top","right"]].set_visible(False)
        plt.xticks(rotation=35, ha="right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_of)

    # ── Save best model ───────────────────────
    st.markdown("---")
    best_name  = df_res.iloc[0]["Model"]
    best_obj   = next(r["_model_obj"] for r in valid if r["Model"] == best_name)
    st.success(f"🥇 Best model: **{best_name}** (Test {score_label} = {df_res.iloc[0][f'Test {score_label}']})")

    if st.button(f"💾 Save '{best_name}' as Active Model", key="mm_save_best"):
        st.session_state["ml_res_model"]   = best_obj
        st.session_state["ml_res_prob"]    = problem_type
        st.session_state["ml_res_algo"]    = best_name
        st.session_state["ml_res_X_cols"]  = list(
            data.drop(target_col, axis=1).select_dtypes(include="number").columns
        )
        st.session_state["ml_res_le"]      = None
        st.session_state["ml_res_scaler"]  = st.session_state.get("mm_scaler")
        st.success("Active model updated. Use **Model Export & Predict** to download or predict.")


# ─────────────────────────────────────────────
#  FEATURE: HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
def show_hyperparameter_tuning(data: pd.DataFrame):
    st.subheader("🔧 Hyperparameter Tuning")
    st.caption("GridSearchCV or RandomizedSearchCV to find the best parameters automatically.")

    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        st.error("No numeric columns available.")
        return

    # ── Config ────────────────────────────────
    h1, h2, h3 = st.columns(3)
    with h1:
        target_col   = st.selectbox("Target Column:", numeric_columns, key="ht_target")
    with h2:
        search_type  = st.radio("Search strategy:", ["Grid Search", "Random Search"],
                                horizontal=True, key="ht_search")
    with h3:
        cv_folds     = st.slider("CV Folds:", 3, 10, 5, key="ht_cv")

    def detect(target):
        return "Classification" if target.dtype == "object" or target.nunique() < 20 else "Regression"

    problem_type = detect(data[target_col])
    st.info(f"**Problem Type:** `{problem_type}`")

    # ── Algorithm + param grid ─────────────────
    st.write("#### Select Algorithm & Parameter Grid")

    if problem_type == "Classification":
        algo_options = [
            "Random Forest Classifier",
            "Gradient Boosting Classifier",
            "K-Nearest Neighbors",
            "Decision Tree",
            "SVM",
        ]
    else:
        algo_options = [
            "Random Forest Regressor",
            "Gradient Boosting Regressor",
            "Ridge Regression",
            "Lasso Regression",
            "Decision Tree Regressor",
        ]

    algo_name = st.selectbox("Algorithm:", algo_options, key="ht_algo")

    # Default param grids
    PARAM_GRIDS = {
        "Random Forest Classifier": {
            "n_estimators": [50, 100, 200],
            "max_depth":    [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "Gradient Boosting Classifier": {
            "n_estimators":   [50, 100, 200],
            "learning_rate":  [0.01, 0.1, 0.2],
            "max_depth":      [3, 5, 7],
        },
        "K-Nearest Neighbors": {
            "n_neighbors": [3, 5, 7, 10, 15],
            "weights":     ["uniform", "distance"],
            "metric":      ["euclidean", "manhattan"],
        },
        "Decision Tree": {
            "max_depth":         [None, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
            "criterion":         ["gini", "entropy"],
        },
        "SVM": {
            "C":      [0.1, 1, 10, 100],
            "kernel": ["rbf", "linear", "poly"],
        },
        "Random Forest Regressor": {
            "n_estimators": [50, 100, 200],
            "max_depth":    [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "Gradient Boosting Regressor": {
            "n_estimators":  [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth":     [3, 5, 7],
        },
        "Ridge Regression": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "fit_intercept": [True, False],
        },
        "Lasso Regression": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "max_iter": [1000, 5000],
        },
        "Decision Tree Regressor": {
            "max_depth":         [None, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
            "criterion":         ["squared_error", "friedman_mse"],
        },
    }

    default_grid_str = json.dumps(PARAM_GRIDS.get(algo_name, {}), indent=2)
    grid_input = st.text_area(
        "Parameter grid (JSON — edit freely):",
        value=default_grid_str,
        height=200,
        key="ht_grid_input",
    )

    n_iter = 20
    if search_type == "Random Search":
        n_iter = st.slider("Number of random iterations:", 5, 100, 20, key="ht_niter")

    if st.button("▶️ Run Tuning", key="ht_run_btn", type="primary"):
        try:
            param_grid = json.loads(grid_input)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON in parameter grid: {e}")
            return

        # Prepare data
        X_raw = data.drop(target_col, axis=1).select_dtypes(include="number")
        y_raw = data[target_col].dropna()
        X_raw = X_raw.loc[y_raw.index]
        imp   = SimpleImputer(strategy="mean")
        X_imp = pd.DataFrame(imp.fit_transform(X_raw), columns=X_raw.columns)
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_imp)
        y_enc = y_raw.reset_index(drop=True)
        le = None
        if problem_type == "Classification" and y_enc.dtype not in ["int64","int32"]:
            le    = LabelEncoder()
            y_enc = pd.Series(le.fit_transform(y_enc))

        # Build base estimator
        base_map = {
            "Random Forest Classifier":     RandomForestClassifier(random_state=42),
            "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
            "K-Nearest Neighbors":          KNeighborsClassifier(),
            "Decision Tree":                DecisionTreeClassifier(random_state=42),
            "SVM":                          SVC(),
            "Random Forest Regressor":      RandomForestRegressor(random_state=42),
            "Gradient Boosting Regressor":  GradientBoostingRegressor(random_state=42),
            "Ridge Regression":             Ridge(),
            "Lasso Regression":             Lasso(),
            "Decision Tree Regressor":      DecisionTreeRegressor(random_state=42),
        }
        base_model   = base_map[algo_name]
        cv_scoring   = "accuracy" if problem_type == "Classification" else "r2"

        with st.spinner(f"Running {search_type} — this may take a moment…"):
            if search_type == "Grid Search":
                searcher = GridSearchCV(
                    base_model, param_grid,
                    cv=cv_folds, scoring=cv_scoring,
                    n_jobs=-1, verbose=0, return_train_score=True,
                )
            else:
                searcher = RandomizedSearchCV(
                    base_model, param_grid,
                    n_iter=n_iter, cv=cv_folds, scoring=cv_scoring,
                    n_jobs=-1, verbose=0, return_train_score=True,
                    random_state=42,
                )
            searcher.fit(X_scaled, y_enc)

        st.session_state["ht_searcher"]    = searcher
        st.session_state["ht_problem"]     = problem_type
        st.session_state["ht_algo"]        = algo_name
        st.session_state["ht_cv_scoring"]  = cv_scoring
        st.session_state["ht_X_scaled"]    = X_scaled
        st.session_state["ht_y_enc"]       = y_enc
        st.session_state["ht_scaler"]      = scaler
        st.session_state["ht_imp"]         = imp
        st.session_state["ht_feat_cols"]   = X_raw.columns.tolist()

    if "ht_searcher" not in st.session_state:
        return

    searcher    = st.session_state["ht_searcher"]
    problem_type= st.session_state["ht_problem"]
    cv_scoring  = st.session_state["ht_cv_scoring"]
    algo_name   = st.session_state["ht_algo"]

    st.markdown("---")
    st.write("### 🏆 Tuning Results")

    # Best params box
    b1, b2 = st.columns(2)
    b1.success(f"**Best CV Score:** `{searcher.best_score_:.4f}`")
    b2.info(f"**Best Parameters:**\n```json\n{json.dumps(searcher.best_params_, indent=2)}\n```")

    # Full results table
    cv_results_df = pd.DataFrame(searcher.cv_results_)
    show_cols = (
        ["params", "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]
    )
    show_cols = [c for c in show_cols if c in cv_results_df.columns]
    cv_display = cv_results_df[show_cols].sort_values("rank_test_score").reset_index(drop=True)
    cv_display.columns = [c.replace("mean_","").replace("_"," ").title() for c in cv_display.columns]

    st.write("#### All Candidates")
    st.dataframe(
        cv_display.style.background_gradient(subset=["Test Score"], cmap="RdYlGn"),
        use_container_width=True,
    )

    # Score distribution plot
    st.write("#### CV Score Distribution Across Candidates")
    fig_ht, ax_ht = plt.subplots(figsize=(10, 4))
    scores_sorted = cv_results_df["mean_test_score"].sort_values(ascending=False).values
    colors_ht = ["#FFD700" if i == 0 else "#3498DB" for i in range(len(scores_sorted))]
    ax_ht.bar(range(len(scores_sorted)), scores_sorted, color=colors_ht, edgecolor="white")
    ax_ht.axhline(searcher.best_score_, color="#E74C3C", linestyle="--",
                  linewidth=1.5, label=f"Best = {searcher.best_score_:.4f}")
    ax_ht.set(xlabel="Candidate #", ylabel=cv_scoring.upper(),
              title=f"{search_type} — Score per Candidate ({algo_name})")
    ax_ht.legend()
    ax_ht.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_ht)

    # Save best tuned model
    st.markdown("---")
    if st.button("💾 Save Best Tuned Model as Active Model", key="ht_save_btn"):
        st.session_state["ml_res_model"]  = searcher.best_estimator_
        st.session_state["ml_res_prob"]   = problem_type
        st.session_state["ml_res_algo"]   = algo_name + " (Tuned)"
        st.session_state["ml_res_X_cols"] = st.session_state.get("ht_feat_cols", [])
        st.session_state["ml_res_le"]     = None
        st.session_state["ht_best_model"] = searcher.best_estimator_
        st.success("✅ Tuned model saved. Use **Model Export & Predict** to download or predict on new data.")


# ─────────────────────────────────────────────
#  FEATURE: MODEL EXPORT & PREDICT
# ─────────────────────────────────────────────
def show_model_export_predict(data: pd.DataFrame):
    st.subheader("📦 Model Export & Predict on New Data")
    st.caption("Download your trained model or upload a new CSV to get instant predictions.")

    # ── Check a model exists ──────────────────
    model_sources = {
        "ML Algorithm":           st.session_state.get("ml_res_model"),
        "Multi-Model Comparison":  st.session_state.get("ml_res_model"),
        "Hyperparameter Tuning":   st.session_state.get("ht_best_model"),
    }
    available = {k: v for k, v in model_sources.items() if v is not None}

    if not available:
        st.warning(
            "No trained model found. "
            "Go to **ML Algorithm**, **Multi-Model Comparison**, or **Hyperparameter Tuning** "
            "to train and save a model first."
        )
        return

    # Pick which model to use
    model_choice = st.selectbox("Use model from:", list(available.keys()), key="ep_model_src")
    model        = (
        st.session_state["ht_best_model"]
        if model_choice == "Hyperparameter Tuning"
        else st.session_state["ml_res_model"]
    )
    algo_name    = st.session_state.get("ml_res_algo", "Unknown")
    problem_type = st.session_state.get("ml_res_prob", "Unknown")
    feat_cols    = st.session_state.get("ml_res_X_cols", [])

    # Info banner
    st.info(
        f"**Active Model:** `{algo_name}` | "
        f"**Task:** `{problem_type}` | "
        f"**Features:** {len(feat_cols)} column(s)"
    )

    st.markdown("---")

    # ════════════════════════════════════════
    #  SECTION 1 — DOWNLOAD MODEL
    # ════════════════════════════════════════
    st.write("### 💾 Download Trained Model")

    d1, d2 = st.columns(2)

    with d1:
        st.write("**Pickle format (.pkl)**")
        model_bytes = pickle.dumps(model)
        st.download_button(
            "⬇️ Download model.pkl",
            data=model_bytes,
            file_name=f"{algo_name.replace(' ','_')}_model.pkl",
            mime="application/octet-stream",
            key="ep_dl_pkl",
        )

    with d2:
        st.write("**Model Info (.txt)**")
        info_lines = [
            f"Algorithm   : {algo_name}",
            f"Problem Type: {problem_type}",
            f"Features    : {', '.join(feat_cols)}",
            f"Params      : {model.get_params()}",
        ]
        st.download_button(
            "⬇️ Download model_info.txt",
            data="\n".join(info_lines).encode(),
            file_name="model_info.txt",
            mime="text/plain",
            key="ep_dl_txt",
        )

    # How to load code snippet
    with st.expander("📋 How to load and use this model in your own Python code"):
        st.code(
            f"""import pickle
import pandas as pd

# Load model
with open("{algo_name.replace(' ','_')}_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare new data (must have these columns):
# {feat_cols}
new_data = pd.read_csv("new_data.csv")[{feat_cols}]

# Predict
predictions = model.predict(new_data)
print(predictions)
""",
            language="python",
        )

    st.markdown("---")

    # ════════════════════════════════════════
    #  SECTION 2 — PREDICT ON NEW CSV
    # ════════════════════════════════════════
    st.write("### 🔮 Predict on New Data")

    if not feat_cols:
        st.warning("Feature columns are not recorded for this model. Retrain via ML Algorithm.")
        return

    st.info(
        f"Upload a CSV containing these **{len(feat_cols)}** feature columns "
        f"(target column not needed):\n\n`{', '.join(feat_cols)}`"
    )

    new_file = st.file_uploader("Upload new CSV for prediction:", type=["csv"], key="ep_upload")

    if new_file is not None:
        try:
            new_df = pd.read_csv(new_file)
            st.write(f"Uploaded: **{new_df.shape[0]} rows × {new_df.shape[1]} cols**")
            st.dataframe(new_df.head(5), use_container_width=True)

            # Check columns
            missing_cols = [c for c in feat_cols if c not in new_df.columns]
            extra_cols   = [c for c in new_df.columns if c not in feat_cols]

            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                return

            if extra_cols:
                st.warning(f"ℹ️ Extra columns will be ignored: {extra_cols}")

            X_new = new_df[feat_cols].copy()

            # Impute and scale using training artifacts if available
            imp_new = SimpleImputer(strategy="mean")
            X_new   = pd.DataFrame(
                imp_new.fit_transform(X_new), columns=feat_cols
            )

            # Try to use same scaler from training session
            scaler_key = (
                "ht_scaler" if model_choice == "Hyperparameter Tuning"
                else "mm_scaler" if model_choice == "Multi-Model Comparison"
                else None
            )
            if scaler_key and scaler_key in st.session_state:
                X_new_scaled = st.session_state[scaler_key].transform(X_new)
            else:
                # Fallback: standardise from the new data itself
                st.warning(
                    "⚠️ Original training scaler not found — "
                    "scaling from new data. For best results, retrain via ML Algorithm."
                )
                X_new_scaled = StandardScaler().fit_transform(X_new)

            if st.button("▶️ Generate Predictions", key="ep_predict_btn", type="primary"):
                preds = model.predict(X_new_scaled)

                # Decode labels if classification
                le = st.session_state.get("ml_res_le")
                if le is not None and problem_type == "Classification":
                    try:
                        preds_display = le.inverse_transform(preds.astype(int))
                    except Exception:
                        preds_display = preds
                else:
                    preds_display = preds

                result_df = new_df.copy()
                result_df["Prediction"] = preds_display

                # Prediction confidence for classifiers that support it
                if problem_type == "Classification" and hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(X_new_scaled)
                        result_df["Confidence"] = proba.max(axis=1).round(4)
                    except Exception:
                        pass

                st.session_state["ep_results"] = result_df

        except Exception as e:
            st.error(f"Error reading file: {e}")

    # ── Prediction results ────────────────────
    if "ep_results" in st.session_state:
        result_df = st.session_state["ep_results"]
        st.markdown("---")
        st.write("### ✅ Prediction Results")

        # Summary stats
        p1, p2 = st.columns(2)
        p1.metric("Total Predictions", len(result_df))
        if "Confidence" in result_df.columns:
            p2.metric("Avg Confidence", f"{result_df['Confidence'].mean():.2%}")

        st.dataframe(result_df, use_container_width=True)

        # Distribution of predictions
        st.write("#### Prediction Distribution")
        fig_pd, ax_pd = plt.subplots(figsize=(8, 4))
        if problem_type == "Classification":
            vc = pd.Series(result_df["Prediction"]).value_counts()
            ax_pd.bar(vc.index.astype(str), vc.values,
                      color="#3498DB", edgecolor="white")
            ax_pd.set(xlabel="Predicted Class", ylabel="Count",
                      title="Predicted Class Distribution")
        else:
            ax_pd.hist(result_df["Prediction"], bins=30,
                       color="#3498DB", edgecolor="white", alpha=0.85)
            ax_pd.axvline(result_df["Prediction"].mean(), color="#E74C3C",
                          linestyle="--", linewidth=1.5,
                          label=f"Mean = {result_df['Prediction'].mean():.3f}")
            ax_pd.legend()
            ax_pd.set(xlabel="Predicted Value", ylabel="Frequency",
                      title="Predicted Value Distribution")
        ax_pd.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_pd)

        # Download predictions
        st.download_button(
            "⬇️ Download Predictions CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
            key="ep_dl_preds",
        )


# ─────────────────────────────────────────────
#  FEATURE: CLASS IMBALANCE HANDLING
# ─────────────────────────────────────────────
def show_class_imbalance(data: pd.DataFrame):
    st.subheader("⚖️ Class Imbalance Handling")
    st.caption("Detect and fix imbalanced target columns before classification.")

    all_cols = data.columns.tolist()
    target_col = st.selectbox("Select target (label) column:", all_cols, key="imb_target")
    target     = data[target_col]

    # ── Class distribution ────────────────────
    st.write("### 📊 Current Class Distribution")
    vc      = target.value_counts()
    vc_pct  = (vc / len(target) * 100).round(2)

    d1, d2, d3 = st.columns(3)
    d1.metric("Total Samples",   len(target))
    d2.metric("Number of Classes", target.nunique())
    d3.metric("Minority Class %",  f"{vc_pct.min():.1f}%")

    imb_ratio = vc.max() / vc.min() if vc.min() > 0 else float("inf")
    if imb_ratio > 3:
        st.warning(
            f"⚠️ Imbalance ratio **{imb_ratio:.1f}:1** — the majority class has "
            f"{imb_ratio:.1f}× more samples than the minority class."
        )
    else:
        st.success(f"✅ Classes are relatively balanced (ratio = {imb_ratio:.1f}:1).")

    # Distribution chart
    fig_ib, axes_ib = plt.subplots(1, 2, figsize=(12, 4))
    colors_ib = ["#E74C3C" if v == vc.min() else "#3498DB" for v in vc.values]
    axes_ib[0].bar(vc.index.astype(str), vc.values, color=colors_ib, edgecolor="white")
    axes_ib[0].set(title="Class Counts", xlabel=target_col, ylabel="Count")
    axes_ib[0].spines[["top","right"]].set_visible(False)
    axes_ib[1].pie(vc.values, labels=vc.index.astype(str),
                   autopct="%1.1f%%", startangle=90,
                   colors=["#3498DB","#E74C3C","#2ECC71","#F39C12","#9B59B6"])
    axes_ib[1].set_title("Class Proportions")
    plt.tight_layout()
    st.pyplot(fig_ib)

    st.markdown("---")

    # ── Method selection ──────────────────────
    st.write("### 🛠️ Handling Method")

    with st.expander("📖 Method Guide"):
        guide = pd.DataFrame({
            "Method": ["SMOTE", "Random Oversampling", "Random Undersampling",
                       "Class Weights", "SMOTE + Undersampling (Combined)"],
            "How it works": [
                "Generates synthetic minority samples using KNN interpolation",
                "Duplicates random minority samples",
                "Removes random majority samples",
                "Adjusts model's loss penalty, no resampling",
                "SMOTE minority + undersample majority to a balanced ratio",
            ],
            "Best for": [
                "Medium datasets, structured data",
                "Small datasets, quick fix",
                "Very large datasets",
                "Use during model training (pass class_weight='balanced')",
                "Severe imbalance, avoids over-duplication",
            ],
        })
        st.dataframe(guide, use_container_width=True)

    method = st.selectbox("Select method:", [
        "SMOTE (Synthetic Minority Oversampling)",
        "Random Oversampling",
        "Random Undersampling",
        "Combined SMOTE + Undersampling",
        "Show Class Weights Only (no resampling)",
    ], key="imb_method")

    # Feature columns
    feature_cols = [c for c in data.select_dtypes(include="number").columns if c != target_col]
    if not feature_cols:
        st.error("No numeric feature columns found.")
        return

    if method != "Show Class Weights Only (no resampling)":
        sampling_strategy = st.slider(
            "Target ratio for minority class (0.5 = 50% of majority):",
            0.1, 1.0, 0.5, 0.1, key="imb_ratio_slider"
        )

    if st.button("▶️ Apply", key="imb_apply_btn", type="primary"):
        X = data[feature_cols].copy()
        y = target.copy()

        # Drop nulls
        valid = y.dropna().index
        X, y  = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)

        # Impute
        imp = SimpleImputer(strategy="mean")
        X   = pd.DataFrame(imp.fit_transform(X), columns=feature_cols)

        le = LabelEncoder()
        y_enc = pd.Series(le.fit_transform(y.astype(str)))

        try:
            if method == "SMOTE (Synthetic Minority Oversampling)":
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
                X_res, y_res = sampler.fit_resample(X, y_enc)

            elif method == "Random Oversampling":
                from imblearn.over_sampling import RandomOverSampler
                sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
                X_res, y_res = sampler.fit_resample(X, y_enc)

            elif method == "Random Undersampling":
                from imblearn.under_sampling import RandomUnderSampler
                sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
                X_res, y_res = sampler.fit_resample(X, y_enc)

            elif method == "Combined SMOTE + Undersampling":
                from imblearn.combine import SMOTETomek
                sampler = SMOTETomek(random_state=42)
                X_res, y_res = sampler.fit_resample(X, y_enc)

            elif method == "Show Class Weights Only (no resampling)":
                from sklearn.utils.class_weight import compute_class_weight
                classes    = np.unique(y_enc)
                weights    = compute_class_weight("balanced", classes=classes, y=y_enc)
                weight_map = dict(zip(le.inverse_transform(classes), weights.round(4)))
                st.success("Class weights computed (use as `class_weight` param in your model):")
                st.json(weight_map)
                st.info(
                    "Example usage:\n```python\n"
                    "RandomForestClassifier(class_weight='balanced')\n"
                    "SVC(class_weight='balanced')\n```"
                )
                return

            # Decode labels back
            y_res_decoded = pd.Series(le.inverse_transform(y_res.astype(int)))
            result_df     = pd.DataFrame(X_res, columns=feature_cols)
            result_df[target_col] = y_res_decoded

            st.session_state["imb_result"] = result_df

        except ImportError:
            st.error(
                "❌ `imbalanced-learn` is not installed.\n\n"
                "Run: `pip install imbalanced-learn`"
            )
            return

    if "imb_result" in st.session_state:
        result_df = st.session_state["imb_result"]
        new_vc    = result_df[target_col].value_counts()

        st.markdown("---")
        st.write("### ✅ After Resampling")
        b1, b2, b3 = st.columns(3)
        b1.metric("Samples Before", len(data))
        b2.metric("Samples After",  len(result_df))
        b3.metric("New Imbalance Ratio",
                  f"{new_vc.max()/new_vc.min():.1f}:1" if new_vc.min() > 0 else "N/A")

        # Side-by-side comparison
        fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(12, 4))
        axes_cmp[0].bar(vc.index.astype(str), vc.values,
                        color="#E74C3C", edgecolor="white", alpha=0.85)
        axes_cmp[0].set(title="Before Resampling", xlabel=target_col, ylabel="Count")
        axes_cmp[0].spines[["top","right"]].set_visible(False)
        axes_cmp[1].bar(new_vc.index.astype(str), new_vc.values,
                        color="#27AE60", edgecolor="white", alpha=0.85)
        axes_cmp[1].set(title="After Resampling", xlabel=target_col, ylabel="Count")
        axes_cmp[1].spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_cmp)

        st.dataframe(result_df.head(20), use_container_width=True)

        # Save + download
        col_sv, col_dl = st.columns(2)
        with col_sv:
            if st.button("✅ Save as Active Dataset", key="imb_save_btn", type="primary"):
                st.session_state["cleaned_data"] = result_df
                st.success("Saved! All features will use the resampled dataset.")
                st.rerun()
        with col_dl:
            st.download_button(
                "⬇️ Download Resampled CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="resampled_data.csv",
                mime="text/csv",
                key="imb_dl_btn",
            )


# ─────────────────────────────────────────────
#  FEATURE: TIME SERIES ANALYSIS
# ─────────────────────────────────────────────
def show_time_series(data: pd.DataFrame):
    st.subheader("📈 Time Series Analysis")
    st.caption("Detect datetime columns and analyse trends, seasonality, and stationarity.")

    # ── Detect / select datetime column ───────
    datetime_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()
    # Also check object columns that look like dates
    for col in data.select_dtypes(include="object").columns:
        try:
            pd.to_datetime(data[col].dropna().iloc[:5])
            if col not in datetime_cols:
                datetime_cols.append(col)
        except Exception:
            pass

    numeric_cols = data.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found for time series analysis.")
        return

    ts1, ts2 = st.columns(2)
    with ts1:
        if datetime_cols:
            time_col = st.selectbox("Select datetime column:", datetime_cols, key="ts_time_col")
        else:
            st.warning("No datetime column detected. Using row index as time axis.")
            time_col = None
    with ts2:
        value_col = st.selectbox("Select value column to analyse:", numeric_cols, key="ts_val_col")

    # Parse time column
    if time_col:
        try:
            ts_data = data[[time_col, value_col]].copy()
            ts_data[time_col] = pd.to_datetime(ts_data[time_col])
            ts_data = ts_data.sort_values(time_col).reset_index(drop=True)
            ts_data = ts_data.dropna()
            time_index = ts_data[time_col]
            values     = ts_data[value_col]
        except Exception as e:
            st.error(f"Could not parse datetime column: {e}")
            return
    else:
        ts_data    = data[[value_col]].dropna().reset_index(drop=True)
        time_index = ts_data.index
        values     = ts_data[value_col]

    st.markdown("---")

    # ── Tab layout ────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "📉 Line Plot", "🔄 Rolling Stats",
        "🧩 Decomposition", "📊 Stationarity Test", "📐 Lag / Autocorrelation"
    ])

    with t1:
        st.write(f"#### Time Series — `{value_col}`")
        fig_ts, ax_ts = plt.subplots(figsize=(12, 4))
        ax_ts.plot(time_index, values, color="#2980B9", linewidth=1.2, label=value_col)
        ax_ts.set(xlabel="Time", ylabel=value_col, title=f"{value_col} Over Time")
        ax_ts.legend()
        ax_ts.grid(True, linestyle="--", alpha=0.4)
        ax_ts.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_ts)

        # Basic stats
        s1,s2,s3,s4 = st.columns(4)
        s1.metric("Mean",   f"{values.mean():.3f}")
        s2.metric("Std",    f"{values.std():.3f}")
        s3.metric("Min",    f"{values.min():.3f}")
        s4.metric("Max",    f"{values.max():.3f}")

    with t2:
        st.write("#### Rolling Mean & Standard Deviation")
        window = st.slider("Rolling window size:", 2, min(100, len(values)//2), 7, key="ts_window")
        rolling_mean = values.rolling(window).mean()
        rolling_std  = values.rolling(window).std()

        fig_roll, ax_roll = plt.subplots(figsize=(12, 4))
        ax_roll.plot(time_index, values,       color="#BDC3C7", linewidth=1,   label="Original", alpha=0.7)
        ax_roll.plot(time_index, rolling_mean, color="#2980B9", linewidth=2,   label=f"Rolling Mean ({window})")
        ax_roll.fill_between(
            time_index,
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.2, color="#2980B9", label=f"±1 Std ({window})"
        )
        ax_roll.plot(time_index, rolling_std,  color="#E74C3C", linewidth=1.5,
                     linestyle="--", label=f"Rolling Std ({window})")
        ax_roll.set(xlabel="Time", ylabel=value_col, title="Rolling Statistics")
        ax_roll.legend(fontsize=9)
        ax_roll.grid(True, linestyle="--", alpha=0.4)
        ax_roll.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_roll)

    with t3:
        st.write("#### Seasonal Decomposition (Trend + Seasonal + Residual)")
        if len(values) < 14:
            st.warning("Need at least 14 data points for decomposition.")
        else:
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                period = st.slider("Seasonal period:", 2, min(365, len(values)//2), 12, key="ts_period")
                decomp_model = st.radio("Model type:", ["additive", "multiplicative"],
                                        horizontal=True, key="ts_decomp_model")

                # multiplicative needs all positive values
                if decomp_model == "multiplicative" and (values <= 0).any():
                    st.warning("Multiplicative model requires all positive values. Switching to additive.")
                    decomp_model = "additive"

                result = seasonal_decompose(values.values, model=decomp_model, period=period)

                fig_dc, axes_dc = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                components = [
                    (values.values,        "Original",   "#2C3E50"),
                    (result.trend,         "Trend",      "#2980B9"),
                    (result.seasonal,      "Seasonal",   "#27AE60"),
                    (result.resid,         "Residual",   "#E74C3C"),
                ]
                for ax, (comp, label, color) in zip(axes_dc, components):
                    ax.plot(comp, color=color, linewidth=1.2)
                    ax.set_ylabel(label, fontsize=9)
                    ax.grid(True, linestyle="--", alpha=0.4)
                    ax.spines[["top","right"]].set_visible(False)
                axes_dc[0].set_title(f"Seasonal Decomposition ({decomp_model})", fontsize=12)
                plt.tight_layout()
                st.pyplot(fig_dc)

            except ImportError:
                st.error("Install statsmodels: `pip install statsmodels`")

    with t4:
        st.write("#### Stationarity Test (Augmented Dickey-Fuller)")
        st.caption("H₀: Series has a unit root (non-stationary). Reject H₀ if p-value < 0.05.")
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(values.dropna())
            adf_stat, p_val, lags_used, n_obs, crit_vals, _ = adf_result

            r1,r2,r3,r4 = st.columns(4)
            r1.metric("ADF Statistic", f"{adf_stat:.4f}")
            r2.metric("p-value",       f"{p_val:.4f}")
            r3.metric("Lags Used",     lags_used)
            r4.metric("N Observations",n_obs)

            st.write("**Critical Values:**")
            crit_df = pd.DataFrame(
                {"Significance": list(crit_vals.keys()),
                 "Critical Value": [round(v, 4) for v in crit_vals.values()]}
            )
            st.dataframe(crit_df, use_container_width=True)

            if p_val < 0.05:
                st.success(
                    f"✅ **Stationary** — p-value ({p_val:.4f}) < 0.05. "
                    "Reject H₀. The series does not have a unit root."
                )
            else:
                st.warning(
                    f"⚠️ **Non-stationary** — p-value ({p_val:.4f}) ≥ 0.05. "
                    "Cannot reject H₀. Consider differencing or log transform."
                )
                if st.checkbox("Apply 1st-order differencing and retest", key="ts_diff"):
                    diff_values = values.diff().dropna()
                    adf2        = adfuller(diff_values)
                    st.write(f"**After differencing — ADF = {adf2[0]:.4f}, p = {adf2[1]:.4f}**")
                    if adf2[1] < 0.05:
                        st.success("✅ Differenced series is stationary.")
                    else:
                        st.warning("Still non-stationary. Try 2nd-order differencing or log transform.")

        except ImportError:
            st.error("Install statsmodels: `pip install statsmodels`")

    with t5:
        st.write("#### Lag Plot & Autocorrelation (ACF / PACF)")
        lag_n = st.slider("Number of lags:", 5, min(50, len(values)//2), 20, key="ts_lag_n")
        try:
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

            fig_lag, axes_lag = plt.subplots(1, 3, figsize=(15, 4))

            # Lag plot (value[t] vs value[t-1])
            pd.plotting.lag_plot(values, lag=1, ax=axes_lag[0])
            axes_lag[0].set_title("Lag Plot (lag=1)")
            axes_lag[0].spines[["top","right"]].set_visible(False)

            # ACF
            plot_acf(values.dropna(), lags=lag_n, ax=axes_lag[1], color="#2980B9")
            axes_lag[1].set_title("ACF")
            axes_lag[1].spines[["top","right"]].set_visible(False)

            # PACF
            plot_pacf(values.dropna(), lags=lag_n, ax=axes_lag[2],
                      method="ywm", color="#E74C3C")
            axes_lag[2].set_title("PACF")
            axes_lag[2].spines[["top","right"]].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig_lag)

        except ImportError:
            st.error("Install statsmodels: `pip install statsmodels`")


# ─────────────────────────────────────────────
#  FEATURE: GENERATE REPORT
# ─────────────────────────────────────────────
def show_generate_report(data: pd.DataFrame, raw_data: pd.DataFrame):
    st.subheader("📄 Generate Analysis Report")
    st.caption("Auto-compile a summary of your entire analysis pipeline into a downloadable text report.")

    st.write("### ✏️ Report Settings")
    r1, r2 = st.columns(2)
    with r1:
        report_title    = st.text_input("Report Title:", value="DIAT Data Analysis Report", key="rpt_title")
        analyst_name    = st.text_input("Analyst Name:", value="", key="rpt_analyst")
    with r2:
        dataset_name    = st.text_input("Dataset Name:", value="Uploaded Dataset", key="rpt_dataset")
        include_model   = st.checkbox("Include ML Model Results", value=True, key="rpt_model")
        include_cluster = st.checkbox("Include Clustering Results", value=True, key="rpt_cluster")

    st.markdown("---")
    st.write("### 👁️ Report Preview")

    # ── Build report content ──────────────────
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    def h(text, level=1):
        sep = "=" if level == 1 else "-"
        lines.append(sep * 60)
        lines.append(text.upper() if level == 1 else text)
        lines.append(sep * 60)

    def row(key, val):
        lines.append(f"  {key:<30}: {val}")

    # Title block
    h(report_title)
    lines.append(f"  Generated On  : {now}")
    lines.append(f"  Analyst       : {analyst_name or 'N/A'}")
    lines.append(f"  Dataset       : {dataset_name}")
    lines.append("")

    # 1. Dataset Summary
    h("1. Dataset Summary", 2)
    row("Raw Rows",              raw_data.shape[0])
    row("Raw Columns",           raw_data.shape[1])
    row("Active Rows",           data.shape[0])
    row("Active Columns",        data.shape[1])
    row("Numeric Columns",       len(data.select_dtypes(include="number").columns))
    row("Categorical Columns",   len(data.select_dtypes(include="object").columns))
    row("Total Null Cells (raw)",raw_data.isnull().sum().sum())
    row("Total Null Cells (active)", data.isnull().sum().sum())
    row("Duplicate Rows (raw)", raw_data.duplicated().sum())
    lines.append("")

    # 2. Column Info
    h("2. Column Overview", 2)
    for col in data.columns:
        s        = data[col]
        null_pct = s.isnull().mean() * 100
        if pd.api.types.is_numeric_dtype(s):
            lines.append(
                f"  {col:<25} | numeric   | "
                f"mean={s.mean():.3f} std={s.std():.3f} "
                f"null={null_pct:.1f}% skew={s.skew():.3f}"
            )
        else:
            lines.append(
                f"  {col:<25} | category  | "
                f"unique={s.nunique()} null={null_pct:.1f}% "
                f"top={s.value_counts().index[0] if s.nunique()>0 else 'N/A'}"
            )
    lines.append("")

    # 3. EDA / Cleaning
    h("3. EDA & Data Cleaning", 2)
    if "cleaned_data" in st.session_state:
        row("Cleaning Applied",    "Yes")
        row("Rows After Cleaning", st.session_state["cleaned_data"].shape[0])
        row("Remaining Nulls",     st.session_state["cleaned_data"].isnull().sum().sum())
    else:
        row("Cleaning Applied", "No")
    lines.append("")

    # 4. Outlier Handling
    h("4. Outlier Handling", 2)
    if "outlier_handled_data" in st.session_state:
        oh = st.session_state["outlier_handled_data"]
        row("Outlier Handling Applied", "Yes")
        row("Rows After Handling",       oh.shape[0])
    else:
        row("Outlier Handling Applied", "No")
    lines.append("")

    # 5. Feature Engineering
    h("5. Feature Engineering", 2)
    if "fe_working" in st.session_state:
        fe = st.session_state["fe_working"]
        new_cols = [c for c in fe.columns if c not in raw_data.columns]
        row("Feature Engineering Applied", "Yes")
        row("New Columns Added",            len(new_cols))
        row("New Column Names",             ", ".join(new_cols[:10]) + ("…" if len(new_cols)>10 else ""))
    else:
        row("Feature Engineering Applied", "No")
    lines.append("")

    # 6. Class Imbalance
    h("6. Class Imbalance Handling", 2)
    if "imb_result" in st.session_state:
        row("Imbalance Handling Applied", "Yes")
        row("Resampled Rows",              st.session_state["imb_result"].shape[0])
    else:
        row("Imbalance Handling Applied", "No")
    lines.append("")

    # 7. ML Model Results
    if include_model:
        h("7. ML Model Results", 2)
        if "ml_res_model" in st.session_state:
            row("Algorithm",     st.session_state.get("ml_res_algo",  "N/A"))
            row("Problem Type",  st.session_state.get("ml_res_prob",  "N/A"))
            row("Feature Count", len(st.session_state.get("ml_res_X_cols", [])))
            # CV score
            if "ml_res_cv" in st.session_state:
                cv = st.session_state["ml_res_cv"]
                row("CV Score (mean ± std)", f"{cv.mean():.4f} ± {cv.std():.4f}")
        else:
            row("Model Trained", "No")

        if "mm_results" in st.session_state:
            lines.append("")
            lines.append("  Multi-Model Comparison Leaderboard:")
            valid_mm = [r for r in st.session_state["mm_results"] if "Error" not in r]
            prob_mm  = st.session_state.get("mm_problem", "")
            sl       = st.session_state.get("mm_score_label", "Score")
            for i, r in enumerate(
                sorted(valid_mm, key=lambda x: x.get(f"Test {sl}", 0), reverse=True)[:5], 1
            ):
                lines.append(
                    f"    {i}. {r['Model']:<35} Test {sl} = {r.get(f'Test {sl}', 'N/A')}"
                )

        if "ht_searcher" in st.session_state:
            lines.append("")
            lines.append("  Hyperparameter Tuning:")
            row("  Algorithm",  st.session_state.get("ht_algo", "N/A"))
            row("  Best Score", f"{st.session_state['ht_searcher'].best_score_:.4f}")
            row("  Best Params",str(st.session_state["ht_searcher"].best_params_))
        lines.append("")

    # 8. Clustering
    if include_cluster:
        h("8. Clustering Results", 2)
        if "cluster_labels" in st.session_state:
            labels   = st.session_state["cluster_labels"]
            n_clust  = len([l for l in set(labels) if l != -1])
            n_noise  = list(labels).count(-1)
            row("Algorithm Used",  st.session_state.get("cluster_algo_used", "N/A"))
            row("Clusters Found",  n_clust)
            row("Noise Points",    n_noise)
            row("Total Points",    len(labels))
        else:
            row("Clustering Applied", "No")
        lines.append("")

    # 9. Recommendations
    h("9. Recommendations", 2)
    nulls_left = data.isnull().sum().sum()
    if nulls_left > 0:
        lines.append(f"  • Dataset still has {nulls_left} null(s). Run EDA → Null Handling.")
    if "cleaned_data" not in st.session_state:
        lines.append("  • No cleaning applied yet. Run EDA → Null Value Handling.")
    if "ml_res_model" not in st.session_state:
        lines.append("  • No model has been trained yet. Go to ML Algorithm or Multi-Model Comparison.")

    num_cols = data.select_dtypes(include="number").columns
    if len(num_cols) >= 2:
        corr = data[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = upper.stack()
        high_corr = high_corr[high_corr > 0.85]
        if not high_corr.empty:
            lines.append(
                f"  • {len(high_corr)} highly correlated pair(s) found (|r|>0.85). "
                "Consider dropping redundant features."
            )

    if not any(l.startswith("  •") for l in lines[-8:]):
        lines.append("  • Dataset looks clean and model-ready. ✅")
    lines.append("")
    h("END OF REPORT")

    report_text = "\n".join(lines)

    # ── Preview in expander ───────────────────
    with st.expander("👁️ Full Report Preview", expanded=True):
        st.text(report_text)

    st.markdown("---")

    # ── Download buttons ──────────────────────
    st.write("### ⬇️ Download Report")
    dl1, dl2 = st.columns(2)

    with dl1:
        st.download_button(
            "📄 Download as .txt",
            data=report_text.encode("utf-8"),
            file_name=f"{report_title.replace(' ','_')}.txt",
            mime="text/plain",
            key="rpt_dl_txt",
        )

    with dl2:
        # Generate a simple HTML report for nicer formatting
        html_lines = [
            "<html><head>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:40px;color:#2C3E50;}",
            "h1{color:#2980B9;border-bottom:2px solid #2980B9;padding-bottom:8px;}",
            "h2{color:#2C3E50;margin-top:30px;}",
            "table{border-collapse:collapse;width:100%;margin-top:10px;}",
            "th{background:#2980B9;color:white;padding:8px;text-align:left;}",
            "td{padding:6px 8px;border-bottom:1px solid #ECF0F1;}",
            "tr:nth-child(even){background:#F8F9FA;}",
            ".good{color:#27AE60;font-weight:bold;}",
            ".warn{color:#E74C3C;font-weight:bold;}",
            "</style></head><body>",
            f"<h1>{report_title}</h1>",
            f"<p><b>Generated:</b> {now} &nbsp;|&nbsp; <b>Analyst:</b> {analyst_name or 'N/A'} "
            f"&nbsp;|&nbsp; <b>Dataset:</b> {dataset_name}</p>",
            "<h2>Dataset Summary</h2>",
            "<table><tr><th>Metric</th><th>Value</th></tr>",
            f"<tr><td>Active Rows</td><td>{data.shape[0]}</td></tr>",
            f"<tr><td>Active Columns</td><td>{data.shape[1]}</td></tr>",
            f"<tr><td>Null Cells (active)</td>"
            f"<td class='{'good' if data.isnull().sum().sum()==0 else 'warn'}'>"
            f"{data.isnull().sum().sum()}</td></tr>",
            f"<tr><td>Duplicate Rows (raw)</td><td>{raw_data.duplicated().sum()}</td></tr>",
            "</table>",
        ]

        if include_model and "ml_res_model" in st.session_state:
            html_lines += [
                "<h2>ML Model Results</h2>",
                "<table><tr><th>Parameter</th><th>Value</th></tr>",
                f"<tr><td>Algorithm</td><td>{st.session_state.get('ml_res_algo','N/A')}</td></tr>",
                f"<tr><td>Problem Type</td><td>{st.session_state.get('ml_res_prob','N/A')}</td></tr>",
                "</table>",
            ]

        html_lines.append("</body></html>")
        html_report = "\n".join(html_lines)

        st.download_button(
            "🌐 Download as .html",
            data=html_report.encode("utf-8"),
            file_name=f"{report_title.replace(' ','_')}.html",
            mime="text/html",
            key="rpt_dl_html",
        )


# ─────────────────────────────────────────────
#  SIDEBAR & MAIN ROUTING
# ─────────────────────────────────────────────
file_mode     = st.sidebar.radio("Select File Type:", list(FILE_EXTENSIONS.keys()))
uploaded_file = st.sidebar.file_uploader(
    f"Upload a {file_mode} file", type=FILE_EXTENSIONS[file_mode]
)

if uploaded_file is not None:
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")
    raw_data = process_file(uploaded_file, file_mode)

    if raw_data is not None:

        # ── Decide which dataset to use ───────
        # Evaluated on EVERY rerun so the switch takes effect immediately
        # after the user clicks "Apply & Save Cleaned Data" in the EDA step.
        is_cleaned = "cleaned_data" in st.session_state
        data       = st.session_state["cleaned_data"] if is_cleaned else raw_data

        # ── Sidebar data-status banner ─────────
        st.sidebar.markdown("---")
        if is_cleaned:
            cleaned = st.session_state["cleaned_data"]
            st.sidebar.success(
                f"✅ **Using Processed Data**\n\n"
                f"Rows: {cleaned.shape[0]} | Cols: {cleaned.shape[1]}\n\n"
                f"Nulls remaining: {cleaned.isnull().sum().sum()}"
            )
            if st.sidebar.button("↩️ Reset to Raw Data"):
                del st.session_state["cleaned_data"]
                st.rerun()
        else:
            st.sidebar.warning(
                f"⚠️ **Using Raw Data**\n\n"
                f"Rows: {raw_data.shape[0]} | Cols: {raw_data.shape[1]}\n\n"
                f"Nulls: {raw_data.isnull().sum().sum()}\n\n"
                f"Go to **EDA – Null Value Handling** to clean data."
            )
        st.sidebar.markdown("---")

        # ── Main content area ─────────────────
        # Show which data version is active at the top of every page
        with st.container():
            banner_col1, banner_col2 = st.columns([3, 1])
            with banner_col1:
                if is_cleaned:
                    st.info(
                        f"✅ **Processed data active** — "
                        f"{data.shape[0]} rows × {data.shape[1]} cols | "
                        f"{data.isnull().sum().sum()} nulls remaining"
                    )
                else:
                    st.warning(
                        f"⚠️ **Raw data active** — "
                        f"{data.shape[0]} rows × {data.shape[1]} cols | "
                        f"{data.isnull().sum().sum()} null(s) present. "
                        f"Use **EDA – Null Value Handling** to clean first."
                    )

        st.sidebar.markdown("---")

        # ── Pipeline Progress Tracker ──────────
        st.sidebar.write("### 🗺️ Pipeline Tracker")
        steps = {
            "📁 Data Uploaded":          True,
            "🔍 Nulls Handled":          "cleaned_data"      in st.session_state,
            "⚙️ Features Engineered":    "fe_working"         in st.session_state
                                          or "cleaned_data"   in st.session_state,
            "🚨 Outliers Handled":       "outlier_handled_data" in st.session_state,
            "⚖️ Imbalance Handled":      "imb_result"        in st.session_state,
            "🤖 Model Trained":          "ml_res_model"      in st.session_state,
            "🏆 Models Compared":        "mm_results"        in st.session_state,
            "🔧 Hyperparams Tuned":      "ht_searcher"       in st.session_state,
            "📦 Predictions Made":       "ep_results"        in st.session_state,
        }
        for step_name, done in steps.items():
            icon = "✅" if done else "⬜"
            st.sidebar.write(f"{icon} {step_name}")
        completed = sum(steps.values())
        st.sidebar.progress(completed / len(steps))
        st.sidebar.caption(f"{completed}/{len(steps)} steps completed")
        st.sidebar.markdown("---")

        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if "main_navigation_radio" not in st.session_state:
            st.session_state["main_navigation_radio"] = FEATURES[0]

        option = st.sidebar.radio("Select a Feature:", FEATURES, key="main_navigation_radio")

        # Groq API Key in sidebar for AI Agent Query if not in .env
        if option == "AI Agent Query" and not groq_api_key:
            st.sidebar.markdown("---")
            groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
            if not groq_api_key:
                st.sidebar.warning("Please enter your Groq API Key to use the AI Agent.")

        if option == "Data Overview":
            show_data_overview(data)

        elif option == "AI Agent Query":
            if groq_api_key:
                show_ai_agent_query(data, groq_api_key)
            else:
                st.info("Please enter your Groq API Key in the sidebar to start.")

        elif option == "Data Profiling Report":
            show_data_profiling(data)

        elif option == "EDA – Null Value Handling":
            show_eda_null_handling(raw_data)

        elif option == "Feature Engineering":
            show_feature_engineering(data)

        elif option == "Basic Statistics":
            show_basic_statistics(data)

        elif option == "Visualizations":
            show_visualizations(data)

        elif option == "Correlation Plot":
            show_correlation_plot(data)

        elif option == "Outlier Detection":
            show_outlier_detection(data)

        elif option == "ML Algorithm":
            show_ml_algorithm(data)

        elif option == "Multi-Model Comparison":
            show_multi_model_comparison(data)

        elif option == "Hyperparameter Tuning":
            show_hyperparameter_tuning(data)

        elif option == "Model Export & Predict":
            show_model_export_predict(data)

        elif option == "Class Imbalance Handling":
            show_class_imbalance(data)

        elif option == "Time Series Analysis":
            show_time_series(data)

        elif option == "Generate Report":
            show_generate_report(data, raw_data)

    else:
        st.error("Unable to process the uploaded file.")
else:
    st.sidebar.info(f"Please upload a {file_mode} file to begin.")