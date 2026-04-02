import pandas as pd
import numpy as np


def infer_target_type(df, target_col):
    if series.empty:
        return "Unknown"

    if series.nunique() == 2:
        return "Binary Classification"

    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 20:
            return "Multiclass Classification"
        return "Regression"

    return "Multiclass Classification"

def get_model_ready_features_df(
    df,
    target_col,
    continuous_numeric,
    discrete_numeric,
    categorical_cols,
    datetime_cols,
    dropped_corr_features=None
):
    if dropped_corr_features is None:
        dropped_corr_features = []

    rows = []

    for col in df.columns:
        if col == target_col:
            continue

        if col in datetime_cols:
            continue

        if col in dropped_corr_features:
            continue

        if df[col].nunique() <= 1:
            continue

        if col in continuous_numeric:
            role = "Continuous Numeric"
            encoding = "Scaling recommended"

        elif col in discrete_numeric:
            role = "Discrete Numeric"
            encoding = "Ordinal / One-hot encoding"

        elif col in categorical_cols:
            role = "Categorical"
            encoding = "One-hot / Target encoding"

        else:
            continue

        rows.append({
            "Feature": col,
            "Type": role,
            "Recommended Preprocessing": encoding
        })

    return pd.DataFrame(rows)

def get_model_recommendations_df(target_type):
    model_map = {
        "Regression": [
            ("Linear Regression", "Baseline model"),
            ("Ridge / Lasso", "Handles multicollinearity"),
            ("Random Forest Regressor", "Non-linear, robust"),
            ("XGBoost Regressor", "High performance"),
            ("LightGBM Regressor", "Fast & scalable")
        ],
        "Binary Classification": [
            ("Logistic Regression", "Baseline classifier"),
            ("Random Forest Classifier", "Handles non-linearity"),
            ("XGBoost Classifier", "High accuracy"),
            ("LightGBM Classifier", "Fast & efficient"),
            ("SVM", "Good for small datasets")
        ],
        "Multiclass Classification": [
            ("Multinomial Logistic Regression", "Baseline"),
            ("Random Forest", "Robust multiclass handling"),
            ("XGBoost", "Strong performance"),
            ("LightGBM", "Efficient multiclass"),
            ("CatBoost", "Great for categoricals")
        ]
    }

    models = model_map.get(target_type, [])

    return pd.DataFrame(
        models,
        columns=["Recommended Model", "Why Use It"]
    )

def get_training_plan(
    df,
    target_col,
    continuous_numeric,
    discrete_numeric,
    categorical_cols,
    datetime_cols,
    dropped_corr_features=None
):
    target_type = infer_target_type(df, target_col)

    feature_df = get_model_ready_features_df(
        df,
        target_col,
        continuous_numeric,
        discrete_numeric,
        categorical_cols,
        datetime_cols,
        dropped_corr_features
    )

    model_df = get_model_recommendations_df(target_type)

    return target_type, feature_df, model_df
