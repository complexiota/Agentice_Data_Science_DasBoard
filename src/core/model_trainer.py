import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor


# Algorithms that are inherently Regression models
_REGRESSION_ALGOS = {
    "Linear Regression",
    "Lasso Regression",
    "Gradient Boosting Regression (GBR)",
    "Decision Tree Regression",
    "Artificial Neural Networks (ANN)",
}

# Algorithms that are inherently Classification models
_CLASSIFICATION_ALGOS = {
    "K-Nearest Neighbors (KNN)",
    "Decision Tree",
    "Support Vector Machine (SVM)",
    "Random Forest Classifier",
}


def detect_problem_type(target: pd.Series) -> str:
    """
    Robustly detect whether a target column is for Classification or Regression.

    Rules (in priority order):
    1. If dtype is float  → Regression (continuous values).
    2. If dtype is object → Classification.
    3. If dtype is int AND nunique ≤ 10 AND min value ≥ 0 AND max value < 100 → Classification.
    4. Otherwise          → Regression.
    """
    if pd.api.types.is_float_dtype(target):
        return "Regression"
    if pd.api.types.is_object_dtype(target):
        return "Classification"
    # Integer column
    n_unique = target.nunique()
    if n_unique <= 10 and target.min() >= 0 and target.max() < 100:
        return "Classification"
    return "Regression"


def train_model(
    data: pd.DataFrame,
    target_column: str,
    selected_algorithm: str,
    problem_type: str,
    test_size_pct: int,
    cv_folds: int,
    hyperparams: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Trains a machine learning model based on the provided configuration.
    The problem_type is overridden if the chosen algorithm makes it unambiguous.
    """
    # ── Override problem_type based on algorithm family ───────────────────────
    if selected_algorithm in _REGRESSION_ALGOS:
        problem_type = "Regression"
    elif selected_algorithm in _CLASSIFICATION_ALGOS:
        problem_type = "Classification"

    # Prepare data
    X_raw = data.drop(target_column, axis=1).select_dtypes(include=["number"])
    y_raw = data[target_column]
    valid = y_raw.dropna().index
    X_raw, y_raw = X_raw.loc[valid], y_raw.loc[valid]

    imputer = SimpleImputer(strategy="mean")
    X_imp = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

    # Encode target for classification if needed
    y_enc = y_raw.reset_index(drop=True)
    le = None
    if problem_type == "Classification" and not pd.api.types.is_integer_dtype(y_enc):
        le = LabelEncoder()
        y_enc = pd.Series(le.fit_transform(y_enc))

    X_imp = X_imp.reset_index(drop=True)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=test_size_pct / 100, random_state=42
    )

    # ── Build model ───────────────────────────────────────────────────────────
    if selected_algorithm == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier(
            n_neighbors=hyperparams.get("knn_k", 5),
            weights=hyperparams.get("knn_weights", "uniform")
        )
    elif selected_algorithm == "Decision Tree":
        dt_depth = hyperparams.get("dt_depth", 0)
        model = DecisionTreeClassifier(
            max_depth=dt_depth if dt_depth > 0 else None,
            min_samples_split=hyperparams.get("dt_min_split", 2),
            random_state=42,
        )
    elif selected_algorithm == "Support Vector Machine (SVM)":
        model = SVC(
            C=hyperparams.get("svm_c", 1.0),
            kernel=hyperparams.get("svm_kernel", "rbf"),
            probability=True
        )
    elif selected_algorithm == "Random Forest Classifier":
        rf_depth = hyperparams.get("rf_depth", 0)
        model = RandomForestClassifier(
            n_estimators=hyperparams.get("rf_n", 100),
            max_depth=rf_depth if rf_depth > 0 else None,
            random_state=42,
        )
    elif selected_algorithm == "Linear Regression":
        model = LinearRegression()
    elif selected_algorithm == "Lasso Regression":
        model = Lasso(alpha=hyperparams.get("lasso_alpha", 1.0))
    elif selected_algorithm == "Gradient Boosting Regression (GBR)":
        model = GradientBoostingRegressor(
            n_estimators=hyperparams.get("gbr_n", 100),
            learning_rate=hyperparams.get("gbr_lr", 0.1),
            random_state=42
        )
    elif selected_algorithm == "Decision Tree Regression":
        dt_depth = hyperparams.get("dt_depth", 0)
        model = DecisionTreeRegressor(
            max_depth=dt_depth if dt_depth > 0 else None,
            min_samples_split=hyperparams.get("dt_min_split", 2),
            random_state=42,
        )
    elif selected_algorithm == "Artificial Neural Networks (ANN)":
        layers_str = hyperparams.get("ann_layers", "100,50")
        layers = tuple(int(x.strip()) for x in layers_str.split(",") if x.strip())
        model = MLPRegressor(
            hidden_layer_sizes=layers,
            learning_rate_init=hyperparams.get("ann_lr", 0.001),
            max_iter=500,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown algorithm: '{selected_algorithm}'")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # ── Cross-validation with the correct scoring metric ─────────────────────
    cv_scoring = "accuracy" if problem_type == "Classification" else "r2"
    cv_scores = cross_val_score(model, X_scaled, y_enc, cv=cv_folds, scoring=cv_scoring)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_train": y_pred_train,
        "cv_scores": cv_scores,
        "problem_type": problem_type,
        "algo_name": selected_algorithm,
        "feat_cols": X_imp.columns.tolist(),
        "le": le,
        "X_scaled": X_scaled,
        "y_enc": y_enc
    }
