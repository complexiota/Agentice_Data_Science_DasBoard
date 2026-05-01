import pandas as pd
from typing import List, Optional, Any

def clean_nulls(
    df_in: pd.DataFrame,
    method: str,
    selected_cols: List[str],
    num_const: float = 0.0,
    cat_const: str = "Unknown",
    knn_neighbors: int = 5
) -> pd.DataFrame:
    """
    Cleans null values in a pandas DataFrame based on the specified method.
    """
    df = df_in.copy()
    
    numeric_selected = [c for c in selected_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_selected = [c for c in selected_cols if not pd.api.types.is_numeric_dtype(df[c])]

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
