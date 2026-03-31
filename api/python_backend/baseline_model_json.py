import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB


def _as_float_list(values):
    arr = np.asarray(values)
    if arr.ndim == 0:
        return [float(arr)]
    return [float(v) for v in arr.flatten().tolist()]


def _friendly_feature_name(name):
    name = str(name)
    if name.startswith("num__"):
        return name.replace("num__", "", 1)
    if name.startswith("cat__"):
        return name.replace("cat__", "", 1)
    return name


def _extract_model_insights(pipeline):
    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")

    try:
        raw_features = list(preprocessor.get_feature_names_out())
    except Exception:
        raw_features = []
    feature_names = [_friendly_feature_name(f) for f in raw_features]

    details = {
        "Estimator": model.__class__.__name__,
        "Supports Intercept": hasattr(model, "intercept_"),
        "Supports Coefficients": hasattr(model, "coef_"),
        "Supports Feature Importances": hasattr(model, "feature_importances_"),
    }

    graph = None
    importance_vector = None

    if hasattr(model, "intercept_"):
        intercept_vals = _as_float_list(model.intercept_)
        details["Intercept"] = intercept_vals if len(intercept_vals) > 1 else intercept_vals[0]

    if hasattr(model, "coef_"):
        coef_arr = np.asarray(model.coef_)
        if coef_arr.ndim == 1:
            coef_vec = coef_arr
            details["Coefficient Mode"] = "single-output"
        else:
            coef_vec = np.mean(np.abs(coef_arr), axis=0)
            details["Coefficient Mode"] = "mean-absolute-across-classes"

        importance_vector = np.abs(coef_vec)
        details["Coefficient Count"] = int(coef_vec.shape[0])

    elif hasattr(model, "feature_importances_"):
        fi = np.asarray(model.feature_importances_)
        importance_vector = fi
        details["Importance Count"] = int(fi.shape[0])

    # Keep parameters concise to avoid huge payloads.
    compact_params = {}
    for key, value in model.get_params(deep=False).items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            compact_params[key] = value
    details["Parameters"] = compact_params

    if importance_vector is not None and len(importance_vector) and feature_names:
        pairs = list(zip(feature_names[: len(importance_vector)], importance_vector.tolist()))
        pairs.sort(key=lambda x: abs(float(x[1])), reverse=True)
        top = pairs[:12]
        graph = {
            "type": "bar",
            "title": "Top Feature Contributions",
            "x": [p[0] for p in top],
            "y": [float(p[1]) for p in top],
            "y_label": "Contribution",
        }

    return details, graph

def _to_feature_df(feature_df):
    return feature_df


def _get_scaler(scaler_name):
    mapping = {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "RobustScaler": RobustScaler,
    }
    return mapping.get(scaler_name, StandardScaler)()


def _get_model(target_type, model_name, random_state):
    if target_type == "Regression":
        mapping = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=1.0),
            "ElasticNet Regression": ElasticNet(alpha=1.0, l1_ratio=0.5),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=300, random_state=random_state),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=random_state),
            "AdaBoost Regressor": AdaBoostRegressor(random_state=random_state),
            "KNN Regressor": KNeighborsRegressor(),
            "SVR (Support Vector Regressor)": SVR(),
        }
        resolved = model_name if model_name in mapping else "Linear Regression"
        return mapping[resolved], "regression", resolved

    if target_type == "Binary Classification":
        mapping = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=random_state),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=300, random_state=random_state),
            "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=random_state),
            "AdaBoost Classifier": AdaBoostClassifier(random_state=random_state),
            "KNN Classifier": KNeighborsClassifier(),
            "SVM Classifier": SVC(),
            "Naive Bayes": GaussianNB(),
        }
        resolved = model_name if model_name in mapping else "Logistic Regression"
        return mapping[resolved], "classification", resolved

    mapping = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=random_state),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=random_state),
        "AdaBoost Classifier": AdaBoostClassifier(random_state=random_state),
        "KNN Classifier": KNeighborsClassifier(),
        "SVM Classifier": SVC(),
        "Naive Bayes": GaussianNB(),
    }
    resolved = model_name if model_name in mapping else "Random Forest Classifier"
    return mapping[resolved], "classification", resolved


def _train_pipeline(
    df,
    target_col,
    feature_df,
    target_type,
    test_size,
    random_state,
    model_name,
    scaler_name,
    selected_features=None,
):
    feature_df = _to_feature_df(feature_df)

    if selected_features:
        selected_set = {str(col) for col in selected_features}
        feature_df = feature_df[feature_df["Feature"].isin(selected_set)]

    if feature_df.empty:
        raise ValueError("No valid feature columns selected for training")

    X = df[feature_df["Feature"]]
    y = df[target_col]

    numeric_features = feature_df[
        feature_df["Type"].isin(["Continuous Numeric", "Discrete Numeric"])
    ]["Feature"].tolist()

    categorical_features = feature_df[
        feature_df["Type"] == "Categorical"
    ]["Feature"].tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", _get_scaler(scaler_name), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    model, task, resolved_model = _get_model(target_type, model_name, random_state)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    metrics = {}
    result = {
        "task": task,
        "model_name": resolved_model,
        "scaler_name": scaler_name,
        "pipeline": pipeline,
    }

    model_details, model_graph = _extract_model_insights(pipeline)
    result["model_details"] = model_details
    result["model_graph"] = model_graph

    if task == "classification":
        metrics["Train Accuracy"] = float(accuracy_score(y_train, y_train_pred))
        metrics["Test Accuracy"] = float(accuracy_score(y_test, y_test_pred))
        metrics["Precision"] = float(precision_score(y_test, y_test_pred, average="weighted", zero_division=0))
        metrics["Recall"] = float(recall_score(y_test, y_test_pred, average="weighted", zero_division=0))
        metrics["F1 Score"] = float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_test, y_test_pred)
        report_dict = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
        result["confusion_matrix"] = cm.tolist()
        result["classification_report"] = report_dict
    else:
        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        metrics["Train RMSE"] = train_rmse
        metrics["Test RMSE"] = test_rmse
        metrics["MAE"] = float(mean_absolute_error(y_test, y_test_pred))
        metrics["R2 Score"] = float(r2_score(y_test, y_test_pred))

    result["metrics"] = metrics
    return result


def train_baseline_model(
    df,
    target_col,
    feature_df,
    target_type,
    test_size=0.2,
    random_state=42,
    selected_features=None,
):
    default_model = "Linear Regression" if target_type == "Regression" else (
        "Logistic Regression" if target_type == "Binary Classification" else "Random Forest Classifier"
    )
    return _train_pipeline(
        df=df,
        target_col=target_col,
        feature_df=feature_df,
        target_type=target_type,
        test_size=test_size,
        random_state=random_state,
        model_name=default_model,
        scaler_name="StandardScaler",
        selected_features=selected_features,
    )


def train_custom_model(
    df,
    target_col,
    feature_df,
    target_type,
    model_name,
    scaler_name="StandardScaler",
    test_size=0.2,
    random_state=42,
    selected_features=None,
):
    return _train_pipeline(
        df=df,
        target_col=target_col,
        feature_df=feature_df,
        target_type=target_type,
        test_size=test_size,
        random_state=random_state,
        model_name=model_name,
        scaler_name=scaler_name,
        selected_features=selected_features,
    )
