import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGDRegressor, SGDClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def train_baseline_model(
    df,
    target_col,
    feature_df,
    target_type,
    test_size=0.2,
    random_state=42,
    export_path="baseline_pipeline.joblib",
    hyperparams=None
):
    # -----------------------------
    # Prepare data
    # -----------------------------
    X = df[feature_df["Feature"]]
    y = df[target_col]

    numeric_features = feature_df[
        feature_df["Type"].isin(["Continuous Numeric", "Discrete Numeric"])
    ]["Feature"].tolist()

    categorical_features = feature_df[
        feature_df["Type"] == "Categorical"
    ]["Feature"].tolist()

    # -----------------------------
    # Preprocessing with configurable scaler
    # -----------------------------
    if hyperparams and "feature_scaling" in hyperparams:
        scaler_type = hyperparams["feature_scaling"]
        if scaler_type == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scaler_type == "RobustScaler":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
    else:
        scaler = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # -----------------------------
    # Choose model with hyperparameters
    # -----------------------------
    if target_type == "Regression":
        if hyperparams:
            regularization = hyperparams.get("regularization", "None")
            
            if regularization == "Ridge (L2)":
                model = Ridge(
                    alpha=hyperparams.get("alpha", 1.0),
                    fit_intercept=hyperparams.get("fit_intercept", True),
                    max_iter=hyperparams.get("max_iter", 1000)
                )
            elif regularization == "Lasso (L1)":
                model = Lasso(
                    alpha=hyperparams.get("alpha", 1.0),
                    fit_intercept=hyperparams.get("fit_intercept", True),
                    max_iter=hyperparams.get("max_iter", 1000)
                )
            elif regularization == "ElasticNet":
                model = ElasticNet(
                    alpha=hyperparams.get("alpha", 1.0),
                    l1_ratio=hyperparams.get("l1_ratio", 0.5),
                    fit_intercept=hyperparams.get("fit_intercept", True),
                    max_iter=hyperparams.get("max_iter", 1000)
                )
            else:
                model = LinearRegression(
                    fit_intercept=hyperparams.get("fit_intercept", True)
                )
        else:
            model = LinearRegression()
        task = "regression"

    elif target_type == "Binary Classification":
        if hyperparams:
            penalty = hyperparams.get("penalty", "l2")
            l1_ratio = hyperparams.get("l1_ratio", 0.5) if penalty == "elasticnet" else None
            
            model = LogisticRegression(
                max_iter=hyperparams.get("max_iter", 1000),
                C=hyperparams.get("C", 1.0),
                solver=hyperparams.get("solver", "lbfgs"),
                penalty=penalty,
                l1_ratio=l1_ratio,
                fit_intercept=hyperparams.get("fit_intercept", True)
            )
        else:
            model = LogisticRegression(max_iter=1000)
        task = "classification"

    else:  # Multiclass Classification
        if hyperparams:
            model = RandomForestClassifier(
                n_estimators=hyperparams.get("n_estimators", 100),
                max_depth=hyperparams.get("max_depth", None) if hyperparams.get("max_depth", 10) != 10 else None,
                min_samples_split=hyperparams.get("min_samples_split", 2),
                min_samples_leaf=hyperparams.get("min_samples_leaf", 1),
                max_features=hyperparams.get("max_features", "sqrt"),
                bootstrap=hyperparams.get("bootstrap", True),
                criterion=hyperparams.get("criterion", "gini"),
                random_state=random_state
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state
            )
        task = "classification"

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # -----------------------------
    # Train-test split
    # -----------------------------
    stratify = y if task == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    # -----------------------------
    # Train
    # -----------------------------
    pipeline.fit(X_train, y_train)

    # -----------------------------
    # Predict
    # -----------------------------
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # -----------------------------
    # Metrics
    # -----------------------------
    metrics = {}

    if task == "classification":
        metrics["Train Accuracy"] = accuracy_score(y_train, y_train_pred)
        metrics["Test Accuracy"] = accuracy_score(y_test, y_test_pred)
        metrics["Precision"] = precision_score(y_test, y_test_pred, average="weighted")
        metrics["Recall"] = recall_score(y_test, y_test_pred, average="weighted")
        metrics["F1 Score"] = f1_score(y_test, y_test_pred, average="weighted")

    else:
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        metrics["Train RMSE"] = train_rmse
        metrics["Test RMSE"] = test_rmse
        metrics["MAE"] = mean_absolute_error(y_test, y_test_pred)
        metrics["R2 Score"] = r2_score(y_test, y_test_pred)

    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])

    # -----------------------------
    # Train vs Test Performance Plot (ALL MODELS)
    # -----------------------------
    fig_perf, ax_perf = plt.subplots(figsize=(5, 4))

    if task == "classification":
        ax_perf.bar(
            ["Train Accuracy", "Test Accuracy"],
            [metrics["Train Accuracy"], metrics["Test Accuracy"]],
        )
        ax_perf.set_ylim(0, 1)
        ax_perf.set_title("Train vs Test Accuracy")

    else:
        ax_perf.bar(
            ["Train RMSE", "Test RMSE"],
            [metrics["Train RMSE"], metrics["Test RMSE"]],
        )
        ax_perf.set_title("Train vs Test RMSE")

    # -----------------------------
    # Classification Diagnostics (SIDE BY SIDE)
    # -----------------------------
    fig_cm = fig_report = None

    if task == "classification":
        cm = confusion_matrix(y_test, y_test_pred)
        report_df = pd.DataFrame(
            classification_report(y_test, y_test_pred, output_dict=True)
        ).transpose()

        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_title("Confusion Matrix")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")

        fig_report = report_df

    # -----------------------------
    # Export pipeline
    # -----------------------------
    joblib.dump(pipeline, export_path)

    return {
        "pipeline": pipeline,
        "metrics_df": metrics_df,
        "performance_fig": fig_perf,
        "confusion_matrix_fig": fig_cm,
        "classification_report_df": fig_report,
        "export_path": export_path
    }
