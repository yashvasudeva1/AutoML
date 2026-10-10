import { MLTrainingResult } from "../types";

/**
 * Confusion Matrix calculation for Classification
 */
export function computeConfusionMatrix(
  predictions: number[],
  actual: number[],
  classLabels: string[]
) {
  const numClasses = classLabels.length;
  const matrix: number[][] = Array.from({ length: numClasses }, () => Array(numClasses).fill(0));

  for (let i = 0; i < predictions.length; i++) {
    const a = actual[i];
    const p = predictions[i];
    if (a >= 0 && a < numClasses && p >= 0 && p < numClasses) {
      matrix[a][p]++;
    }
  }

  const perClass: { [label: string]: { precision: number; recall: number; f1: number; support: number } } = {};
  for (let c = 0; c < numClasses; c++) {
    const label = classLabels[c] || `Class ${c}`;
    const tp = matrix[c][c];
    let fn = 0;
    for (let col = 0; col < numClasses; col++) {
      if (col !== c) fn += matrix[c][col];
    }
    let fp = 0;
    for (let row = 0; row < numClasses; row++) {
      if (row !== c) fp += matrix[row][c];
    }

    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    const support = tp + fn;

    perClass[label] = {
      precision: Number(precision.toFixed(4)),
      recall: Number(recall.toFixed(4)),
      f1: Number(f1.toFixed(4)),
      support
    };
  }

  return { matrix, labels: classLabels, perClass };
}

/**
 * Residual Analysis calculation for Regression
 */
export function computeResidualAnalysis(
  predictions: number[],
  actual: number[]
) {
  const residuals = actual.map((act, i) => {
    const pred = predictions[i];
    return {
      actual: Number(act.toFixed(4)),
      predicted: Number(pred.toFixed(4)),
      residual: Number((act - pred).toFixed(4))
    };
  });

  let sumRes = 0;
  let maxRes = 0;
  let sumSqRes = 0;
  for (const r of residuals) {
    const abs = Math.abs(r.residual);
    sumRes += abs;
    if (abs > maxRes) maxRes = abs;
    sumSqRes += Math.pow(r.residual, 2);
  }

  const meanResidual = Number((sumRes / Math.max(1, residuals.length)).toFixed(4));
  const stdResidual = Number(Math.sqrt(sumSqRes / Math.max(1, residuals.length)).toFixed(4));

  return {
    meanResidual,
    maxResidual: Number(maxRes.toFixed(4)),
    stdResidual,
    residuals: residuals.slice(0, 40)
  };
}

/**
 * Standalone, guaranteed offline Python Pipeline Script Generator
 */
export function generateLocalPythonPipeline(params: {
  targetColumn: string;
  features: string[];
  modelId: string;
  modelName: string;
  taskType: "Classification" | "Regression";
  hyperParams: Record<string, any>;
}): string {
  const { targetColumn, features, modelId, taskType, hyperParams } = params;

  let modelImport = "";
  let modelInstantiation = "";

  if (modelId === "gradient_boosting") {
    if (taskType === "Classification") {
      modelImport = "from sklearn.ensemble import GradientBoostingClassifier";
      modelInstantiation = `GradientBoostingClassifier(\n        n_estimators=${hyperParams.nEstimators || 20},\n        learning_rate=${hyperParams.learningRate || 0.1},\n        max_depth=${hyperParams.maxDepth || 3},\n        random_state=42\n    )`;
    } else {
      modelImport = "from sklearn.ensemble import GradientBoostingRegressor";
      modelInstantiation = `GradientBoostingRegressor(\n        n_estimators=${hyperParams.nEstimators || 20},\n        learning_rate=${hyperParams.learningRate || 0.1},\n        max_depth=${hyperParams.maxDepth || 3},\n        random_state=42\n    )`;
    }
  } else if (modelId === "random_forest") {
    if (taskType === "Classification") {
      modelImport = "from sklearn.ensemble import RandomForestClassifier";
      modelInstantiation = `RandomForestClassifier(\n        n_estimators=${hyperParams.nEstimators || 10},\n        max_depth=${hyperParams.maxDepth || 5},\n        min_samples_split=${hyperParams.minSamplesSplit || 2},\n        random_state=42\n    )`;
    } else {
      modelImport = "from sklearn.ensemble import RandomForestRegressor";
      modelInstantiation = `RandomForestRegressor(\n        n_estimators=${hyperParams.nEstimators || 10},\n        max_depth=${hyperParams.maxDepth || 5},\n        min_samples_split=${hyperParams.minSamplesSplit || 2},\n        random_state=42\n    )`;
    }
  } else if (modelId === "decision_tree") {
    if (taskType === "Classification") {
      modelImport = "from sklearn.tree import DecisionTreeClassifier";
      modelInstantiation = `DecisionTreeClassifier(\n        max_depth=${hyperParams.maxDepth || 5},\n        min_samples_split=${hyperParams.minSamplesSplit || 2},\n        random_state=42\n    )`;
    } else {
      modelImport = "from sklearn.tree import DecisionTreeRegressor";
      modelInstantiation = `DecisionTreeRegressor(\n        max_depth=${hyperParams.maxDepth || 5},\n        min_samples_split=${hyperParams.minSamplesSplit || 2},\n        random_state=42\n    )`;
    }
  } else if (modelId === "knn") {
    if (taskType === "Classification") {
      modelImport = "from sklearn.neighbors import KNeighborsClassifier";
      modelInstantiation = `KNeighborsClassifier(n_neighbors=${hyperParams.kNeighbors || 5}, weights='distance')`;
    } else {
      modelImport = "from sklearn.neighbors import KNeighborsRegressor";
      modelInstantiation = `KNeighborsRegressor(n_neighbors=${hyperParams.kNeighbors || 5}, weights='distance')`;
    }
  } else if (modelId === "naive_bayes") {
    modelImport = "from sklearn.naive_bayes import GaussianNB";
    modelInstantiation = "GaussianNB()";
  } else if (modelId === "elastic_net") {
    modelImport = "from sklearn.linear_model import ElasticNet";
    modelInstantiation = `ElasticNet(alpha=${hyperParams.l2Penalty || 0.1}, l1_ratio=${hyperParams.l1Ratio || 0.5}, random_state=42)`;
  } else if (modelId === "ridge") {
    modelImport = "from sklearn.linear_model import Ridge";
    modelInstantiation = `Ridge(alpha=${hyperParams.l2Penalty || 1.0})`;
  } else if (modelId === "regularized_logistic") {
    modelImport = "from sklearn.linear_model import LogisticRegression";
    modelInstantiation = `LogisticRegression(penalty='l2', C=${1 / (hyperParams.l2Penalty || 1.0)}, max_iter=${hyperParams.epochs || 200}, random_state=42)`;
  } else if (modelId === "logistic") {
    modelImport = "from sklearn.linear_model import LogisticRegression";
    modelInstantiation = `LogisticRegression(penalty=None, max_iter=${hyperParams.epochs || 200}, random_state=42)`;
  } else {
    modelImport = "from sklearn.linear_model import LinearRegression";
    modelInstantiation = "LinearRegression()";
  }

  const evalImports = taskType === "Classification"
    ? "from sklearn.metrics import classification_report, accuracy_score, confusion_matrix"
    : "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score";

  const evalCode = taskType === "Classification"
    ? `    # Evaluation Metrics\n    y_pred = pipeline.predict(X_test)\n    print("\\n=== Classification Performance ===")\n    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")\n    print("\\nDetailed Classification Report:")\n    print(classification_report(y_test, y_pred))\n    print("\\nConfusion Matrix:")\n    print(confusion_matrix(y_test, y_pred))`
    : `    # Evaluation Metrics\n    y_pred = pipeline.predict(X_test)\n    r2 = r2_score(y_test, y_pred)\n    mse = mean_squared_error(y_test, y_pred)\n    mae = mean_absolute_error(y_test, y_pred)\n    rmse = np.sqrt(mse)\n    print("\\n=== Regression Performance ===")\n    print(f"R² Score: {r2:.4f}")\n    print(f"Mean Squared Error (MSE): {mse:.4f}")\n    print(f"Mean Absolute Error (MAE): {mae:.4f}")\n    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")`;

  return `"""
AutoML & EDA Studio — Generated Python ML Pipeline
Target Feature: ${targetColumn}
Task Objective: ${taskType}
Model Algorithm: ${params.modelName}
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
${modelImport}
${evalImports}

def train_production_pipeline(data_path: str):
    # 1. Load Dataset
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # 2. Select Features & Target
    features = ${JSON.stringify(features)}
    target = "${targetColumn}"
    
    # Filter non-null rows for modeling
    df_clean = df[features + [target]].dropna(subset=[target])
    X = df_clean[features]
    y = df_clean[target]

    # 3. Detect column modalities
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    print(f"Dataset Shape: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # 4. Construct Preprocessing Pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # 5. Build Unified Pipeline with Chosen Estimator
    model = ${modelInstantiation}

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # 6. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 7. Train Model Pipeline
    print("\\nTraining ML pipeline...")
    pipeline.fit(X_train, y_train)
    print("Model training successfully completed.")

    # 8. 5-Fold Cross Validation
    cv_metric = "accuracy" if "${taskType}" == "Classification" else "r2"
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=cv_metric)
    print(f"\\n5-Fold CV {cv_metric.upper()} Score: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

${evalCode}

    return pipeline

if __name__ == "__main__":
    # Example usage:
    # pipeline = train_production_pipeline("your_dataset.csv")
    print("AutoML pipeline script compiled. Call train_production_pipeline('path/to/dataset.csv') to run.")
`;
}
