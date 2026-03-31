"""
Flask API server for the Dataset Analyzer.
Wraps existing Python modules, handles authentication, and provides
endpoints for CSV upload and analysis.
"""

import os
import sys
import json
import uuid
import io
import hashlib
import secrets
from urllib.parse import quote
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import jwt
import joblib
import requests as http_requests
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

# Add the project root to sys.path so we can import the analysis modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def _load_dotenv_file(path: str) -> None:
    """Lightweight .env loader that sets missing process env vars."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


_load_dotenv_file(os.path.join(PROJECT_ROOT, ".env"))

from column_identification import (
    classify_numeric_columns,
    detect_datetime_columns,
    detect_categorical_columns,
    is_id_like_numeric,
)
from utils import get_data_quality_metrics_df, clean_dataset, get_distribution_insights_df
from data_analysis import (
    get_categorical_descriptive_df,
    get_numerical_descriptive_df,
    get_numeric_correlation_diagnostics,
    get_pearson_corr_matrix,
    get_kendall_correlation_df,
    get_spearman_correlation_df,
    get_correlation_pairs_df,
    numeric_prescriptive_df,
    categorical_prescriptive_df,
    correlation_prescriptive_df,
    dataset_prescriptive_summary,
)
from models import (
    infer_target_type,
    get_model_ready_features_df,
    get_model_recommendations_df,
    get_training_plan,
)
from baseline_model_json import train_baseline_model, train_custom_model

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=os.path.join(PROJECT_ROOT, "frontend"), static_url_path="")


def _get_cors_allowed_origins():
    raw = (os.environ.get("CORS_ALLOWED_ORIGINS") or "").strip()
    if not raw:
        return ["http://127.0.0.1:5000", "http://localhost:5000"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


CORS(
    app,
    resources={r"/api/*": {"origins": _get_cors_allowed_origins()}},
)

SECRET_KEY = (os.environ.get("APP_SECRET_KEY") or secrets.token_hex(32)).strip()
GOOGLE_CLIENT_ID = os.environ.get(
    "GOOGLE_CLIENT_ID",
    "",
).strip()
GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "",
).strip()

import db

# In-memory cache for active sessions; dataset source of truth is SQLite.
DATASET_STORE = {}
PIPELINE_STORE = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def create_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=24),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "Missing token"}), 401
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            request.user_id = data["sub"]
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated


def df_to_json(df):
    """Convert a DataFrame to a JSON-serialisable list of dicts, handling NaN."""
    return json.loads(df.to_json(orient="records"))


def classify_columns(df):
    """Run the full column classification pipeline on a DataFrame."""
    object_cols = df.select_dtypes(include="object").columns.tolist()
    native_numeric = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    converted_numeric, discrete_numeric, continuous_numeric = classify_numeric_columns(
        df, object_cols
    )
    numeric_cols = list(set(native_numeric + converted_numeric))
    datetime_cols = detect_datetime_columns(df)
    categorical_cols = detect_categorical_columns(df)
    categorical_cols = list(set(categorical_cols) - set(datetime_cols) - set(numeric_cols))

    column_type_map = {}
    for col in df.columns:
        if col in continuous_numeric:
            column_type_map[col] = "Continuous Numeric"
        elif col in discrete_numeric:
            column_type_map[col] = "Discrete Numeric"
        elif col in datetime_cols:
            column_type_map[col] = "Date / Time"
        elif col in categorical_cols:
            column_type_map[col] = "Categorical"
        else:
            column_type_map[col] = "Unknown"

    return {
        "numeric_cols": numeric_cols,
        "discrete_numeric": discrete_numeric,
        "continuous_numeric": continuous_numeric,
        "datetime_cols": datetime_cols,
        "categorical_cols": categorical_cols,
        "column_type_map": column_type_map,
    }


def _safe_filename_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")


def _pipeline_key(kind: str, dataset_id: str, target_col: str) -> str:
    safe_target = _safe_filename_part(target_col) or "target"
    return f"{kind}:{dataset_id}:{safe_target}"


def _dataset_df_or_404(entry):
    dataset_id = entry.get("id")
    mem = DATASET_STORE.get(dataset_id)
    if mem and isinstance(mem.get("df"), pd.DataFrame):
        return mem["df"].copy()

    blob = (entry or {}).get("data_blob")
    if blob:
        try:
            return pd.read_csv(io.BytesIO(blob))
        except Exception:
            pass

    filepath = (entry or {}).get("filepath")
    if filepath and os.path.exists(filepath):
        return pd.read_csv(filepath)

    return None


def _clean_df_for_training(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = clean_dataset(df.copy(), null_threshold=0.30)
    return cleaned_df


def _assess_fit(target_type: str, metrics: dict):
    if target_type == "Regression":
        train_rmse = metrics.get("Train RMSE")
        test_rmse = metrics.get("Test RMSE")
        if train_rmse is None or test_rmse is None:
            return {"status": "Unknown", "reason": "Missing RMSE metrics", "gap": None}

        gap = float(abs(test_rmse - train_rmse))
        ratio = float(test_rmse / max(train_rmse, 1e-8))
        if ratio > 1.5:
            status = "Overfitting"
            reason = "Test RMSE is significantly higher than Train RMSE."
        elif ratio < 0.9:
            status = "Underfitting"
            reason = "Train RMSE is unexpectedly higher than Test RMSE, suggesting weak fit."
        elif ratio <= 1.15:
            status = "Good Fit"
            reason = "Train and Test RMSE are closely aligned."
        else:
            status = "Mild Overfitting"
            reason = "Test RMSE is moderately higher than Train RMSE."

        return {"status": status, "reason": reason, "gap": round(gap, 6), "ratio": round(ratio, 6)}

    train_acc = metrics.get("Train Accuracy")
    test_acc = metrics.get("Test Accuracy")
    if train_acc is None or test_acc is None:
        return {"status": "Unknown", "reason": "Missing accuracy metrics", "gap": None}

    gap = float(train_acc - test_acc)
    abs_gap = float(abs(gap))
    if gap > 0.12:
        status = "Overfitting"
        reason = "Train accuracy is much higher than Test accuracy."
    elif test_acc < 0.6 and abs_gap < 0.08:
        status = "Underfitting"
        reason = "Both Train and Test accuracy are low and close."
    elif abs_gap <= 0.05:
        status = "Good Fit"
        reason = "Train and Test accuracy are closely aligned."
    else:
        status = "Mild Overfitting"
        reason = "Train accuracy is moderately higher than Test accuracy."

    return {"status": status, "reason": reason, "gap": round(abs_gap, 6)}


def _build_dataset_chat_context(df: pd.DataFrame):
    """Build a compact JSON-safe context for dataset Q&A."""
    max_cols = 40
    selected_cols = df.columns.tolist()[:max_cols]
    cut_df = df[selected_cols].copy()

    missing = cut_df.isna().sum().sort_values(ascending=False)
    missing_top = [
        {"column": col, "missing_count": int(cnt)}
        for col, cnt in missing.items()
        if int(cnt) > 0
    ][:20]

    numeric_cols = cut_df.select_dtypes(include=[np.number]).columns.tolist()[:20]
    numeric_summary = {}
    if numeric_cols:
        desc = cut_df[numeric_cols].describe().round(4).fillna(0)
        numeric_summary = {col: {k: float(v) for k, v in vals.items()} for col, vals in desc.to_dict().items()}

    categorical_cols = cut_df.select_dtypes(exclude=[np.number]).columns.tolist()[:20]
    categorical_summary = []
    for col in categorical_cols:
        series = cut_df[col]
        mode_vals = series.mode(dropna=True)
        top_value = None
        if not mode_vals.empty:
            top_value = str(mode_vals.iloc[0])
        categorical_summary.append({
            "column": col,
            "unique": int(series.nunique(dropna=True)),
            "top_value": top_value,
        })

    sample_records = cut_df.head(8).replace({np.nan: None}).to_dict(orient="records")

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "included_columns": selected_cols,
        "missing_top": missing_top,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "sample_rows": sample_records,
    }


def _ask_gemini_about_dataset(question: str, dataset_context: dict):
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key is not configured.")

    context_text = json.dumps(dataset_context, ensure_ascii=True)
    if len(context_text) > 20000:
        context_text = context_text[:20000]

    prompt = (
        "You are a dataset analysis assistant. Answer ONLY based on the provided dataset context. "
        "If the context is insufficient, clearly say what is missing. Keep answers concise and practical.\n\n"
        f"Dataset context:\n{context_text}\n\n"
        f"User question:\n{question}"
    )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "maxOutputTokens": 512,
        },
    }

    resp = http_requests.post(url, json=payload, timeout=30)
    if resp.status_code >= 400:
        try:
            err = resp.json()
            err_msg = err.get("error", {}).get("message", "Gemini request failed")
        except Exception:
            err_msg = "Gemini request failed"
        raise RuntimeError(err_msg)

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("No response from Gemini")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "\n".join(p.get("text", "") for p in parts if p.get("text"))
    text = text.strip()
    if not text:
        raise RuntimeError("Empty response from Gemini")
    return text


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json()
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user_id = str(uuid.uuid4())
    success = db.create_user(user_id, name, email, hash_password(password))
    if not success:
        return jsonify({"error": "Email already registered"}), 409

    token = create_token(user_id)
    return jsonify({"token": token, "user": {"id": user_id, "name": name, "email": email}}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    user = db.get_user_by_email(email)
    if not user or user["password"] != hash_password(password):
        return jsonify({"error": "Invalid credentials"}), 401

    token = create_token(user["id"])
    return jsonify({
        "token": token,
        "user": {"id": user["id"], "name": user["name"], "email": user["email"]},
    })


@app.route("/api/auth/me", methods=["GET"])
@token_required
def me():
    user = db.get_user_by_id(request.user_id)
    if user:
        return jsonify({"user": {"id": user["id"], "name": user["name"], "email": user["email"]}})
    return jsonify({"error": "User not found"}), 404

@app.route("/api/auth/google", methods=["POST"])
def auth_google():
    data = request.get_json()
    raw_id_token = data.get("token", "")
    if not GOOGLE_CLIENT_ID:
        return jsonify({"error": "Google login is not configured on server"}), 503

    try:
        decoded = google_id_token.verify_oauth2_token(
            raw_id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
        email = decoded.get("email")
        name = decoded.get("name", "Google User")
        
        if not email:
            return jsonify({"error": "Google token missing email"}), 400
            
        user = db.get_user_by_email(email)
        if not user:
            user_id = str(uuid.uuid4())
            import secrets
            db.create_user(user_id, name, email, hash_password(secrets.token_hex(16)))
            user = db.get_user_by_id(user_id)
            
        token = create_token(user["id"])
        return jsonify({
            "token": token,
            "user": {"id": user["id"], "name": user["name"], "email": user["email"]},
        })
    except Exception as e:
        return jsonify({"error": f"Invalid Google token: {str(e)}"}), 401


@app.route("/api/auth/google-config", methods=["GET"])
def google_config():
    return jsonify({
        "enabled": bool(GOOGLE_CLIENT_ID),
        "client_id": GOOGLE_CLIENT_ID,
    })

@app.route("/api/user/profile", methods=["PUT"])
@token_required
def update_profile():
    data = request.get_json()
    name = data.get("name")
    password = data.get("password")
    
    hashed_pw = hash_password(password) if password else None
    db.update_user(request.user_id, name=name, password=hashed_pw)
    
    user = db.get_user_by_id(request.user_id)
    return jsonify({"success": True, "user": {"id": user["id"], "name": user["name"], "email": user["email"]}})

@app.route("/api/user/datasets", methods=["GET"])
@token_required
def get_my_datasets():
    datasets = db.get_user_dataset_groups(request.user_id)
    return jsonify({"datasets": datasets})



# ---------------------------------------------------------------------------
# Dataset endpoints
# ---------------------------------------------------------------------------

@app.route("/api/dataset/upload", methods=["POST"])
@token_required
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400

    dataset_id = str(uuid.uuid4())
    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"error": "Empty file"}), 400
    df = pd.read_csv(io.BytesIO(file_bytes))
    col_info = classify_columns(df)

    DATASET_STORE[dataset_id] = {
        "df": df.copy(),
        "filename": file.filename,
        "user_id": request.user_id,
    }

    rows, cols = df.shape
    missing_pct = round(df.isna().mean().mean() * 100, 2)
    memory_mb = round(df.memory_usage(deep=True).sum() / 1024**2, 2)

    summary_data = {
        "rows": int(rows),
        "columns": int(cols),
        "missing_pct": missing_pct,
        "memory_mb": memory_mb,
        "continuous_numeric": len(col_info["continuous_numeric"]),
        "discrete_numeric": len(col_info["discrete_numeric"]),
        "categorical": len(col_info["categorical_cols"]),
        "datetime": len(col_info["datetime_cols"]),
    }

    db.create_dataset(
        dataset_id=dataset_id,
        user_id=request.user_id,
        filename=file.filename,
        filepath="",
        summary=summary_data,
        col_info=col_info,
        data_blob=file_bytes,
    )

    rows, cols = df.shape
    missing_pct = round(df.isna().mean().mean() * 100, 2)
    memory_mb = round(df.memory_usage(deep=True).sum() / 1024**2, 2)

    # Column classification summary
    classification_rows = []
    for col in df.columns:
        unique_count = df[col].nunique()
        col_type = col_info["column_type_map"].get(col, "Unknown")

        if col in col_info["datetime_cols"]:
            reasoning = "Valid datetime values detected"
        elif col in col_info["discrete_numeric"]:
            reasoning = f"Integer with limited values ({unique_count})"
        elif col in col_info["continuous_numeric"]:
            reasoning = f"High cardinality numeric ({unique_count})"
        elif col in col_info["categorical_cols"]:
            reasoning = f"Low cardinality ({unique_count})"
        else:
            reasoning = "Ambiguous behavior"

        classification_rows.append({
            "column": col,
            "type": col_type,
            "reasoning": reasoning,
            "unique_values": int(unique_count),
        })

    return jsonify({
        "dataset_id": dataset_id,
        "filename": file.filename,
        "summary": {
            "rows": int(rows),
            "columns": int(cols),
            "missing_pct": missing_pct,
            "memory_mb": memory_mb,
            "continuous_numeric": len(col_info["continuous_numeric"]),
            "discrete_numeric": len(col_info["discrete_numeric"]),
            "categorical": len(col_info["categorical_cols"]),
            "datetime": len(col_info["datetime_cols"]),
        },
        "classification": classification_rows,
        "preview": df.head(10).fillna("").to_dict(orient="records"),
    })


@app.route("/api/dataset/<dataset_id>/overview", methods=["GET"])
@token_required
def dataset_overview(dataset_id):
    entry = db.get_dataset(dataset_id)
    if not entry:
        return jsonify({"error": "Dataset not found"}), 404

    df = _dataset_df_or_404(entry)
    if df is None:
        return jsonify({"error": "Dataset content not available in memory. Re-upload to continue."}), 410
    col_info = entry["col_info"]

    rows, cols = df.shape
    missing_pct = round(df.isna().mean().mean() * 100, 2)
    memory_mb = round(df.memory_usage(deep=True).sum() / 1024**2, 2)

    metrics_df = get_data_quality_metrics_df(df)
    cleaned_df = clean_dataset(df)

    return jsonify({
        "summary": {
            "rows": int(rows),
            "columns": int(cols),
            "missing_pct": missing_pct,
            "memory_mb": memory_mb,
        },
        "quality_metrics": df_to_json(metrics_df),
        "cleaned_preview": cleaned_df.head(50).fillna("").to_dict(orient="records"),
        "cleaned_shape": {"rows": int(cleaned_df.shape[0]), "columns": int(cleaned_df.shape[1])},
    })


@app.route("/api/dataset/<dataset_id>/resume", methods=["GET"])
@token_required
def resume_dataset(dataset_id):
    entry = db.get_dataset(dataset_id)
    if not entry:
        return jsonify({"error": "Dataset not found"}), 404
    if entry.get("user_id") != request.user_id:
        return jsonify({"error": "Unauthorized access to dataset"}), 403

    df = _dataset_df_or_404(entry)
    if df is None:
        return jsonify({"error": "Dataset content not available in memory. Re-upload to continue."}), 410
    col_info = entry["col_info"]

    rows, cols = df.shape
    missing_pct = round(df.isna().mean().mean() * 100, 2)
    memory_mb = round(df.memory_usage(deep=True).sum() / 1024**2, 2)

    classification_rows = []
    for col in df.columns:
        unique_count = df[col].nunique()
        col_type = col_info["column_type_map"].get(col, "Unknown")

        if col in col_info["datetime_cols"]:
            reasoning = "Valid datetime values detected"
        elif col in col_info["discrete_numeric"]:
            reasoning = f"Integer with limited values ({unique_count})"
        elif col in col_info["continuous_numeric"]:
            reasoning = f"High cardinality numeric ({unique_count})"
        elif col in col_info["categorical_cols"]:
            reasoning = f"Low cardinality ({unique_count})"
        else:
            reasoning = "Ambiguous behavior"

        classification_rows.append({
            "column": col,
            "type": col_type,
            "reasoning": reasoning,
            "unique_values": int(unique_count),
        })

    return jsonify({
        "dataset_id": dataset_id,
        "filename": entry["filename"],
        "summary": {
            "rows": int(rows),
            "columns": int(cols),
            "missing_pct": missing_pct,
            "memory_mb": memory_mb,
            "continuous_numeric": len(col_info["continuous_numeric"]),
            "discrete_numeric": len(col_info["discrete_numeric"]),
            "categorical": len(col_info["categorical_cols"]),
            "datetime": len(col_info["datetime_cols"]),
        },
        "classification": classification_rows,
        "preview": df.head(10).fillna("").to_dict(orient="records"),
    })


@app.route("/api/dataset/<dataset_id>/clean", methods=["POST"])
@token_required
def perform_clean(dataset_id):
    entry = db.get_dataset(dataset_id)
    if not entry:
        return jsonify({"error": "Dataset not found"}), 404

    df = _dataset_df_or_404(entry)
    if df is None:
        return jsonify({"error": "Dataset content not available in memory. Re-upload to continue."}), 410
    
    # 0.3 for 30% threshold
    cleaned_df = clean_dataset(df, null_threshold=0.30)
    
    # Check if empty
    if cleaned_df.empty:
        return jsonify({"error": "Cleaning resulted in an empty dataset"}), 400

    DATASET_STORE[dataset_id] = {
        "df": cleaned_df.copy(),
        "filename": entry["filename"],
        "user_id": entry.get("user_id"),
    }

    cleaned_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
    db.update_dataset_blob(dataset_id, cleaned_bytes)
    
    # Re-run column classification since contents changed
    col_info = classify_columns(cleaned_df)
    db.update_dataset_col_info(dataset_id, col_info)

    return jsonify({"success": True, "rows": len(cleaned_df)})


@app.route("/api/dataset/<dataset_id>/download", methods=["GET"])
@token_required
def download_cleaned(dataset_id):
    entry = db.get_dataset(dataset_id)
    if not entry:
        return jsonify({"error": "Dataset not found"}), 404
        
    df = _dataset_df_or_404(entry)
    if df is None:
        return jsonify({"error": "Dataset content not available in memory. Re-upload to continue."}), 410

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return send_file(
        io.BytesIO(csv_bytes),
        as_attachment=True,
        download_name=f"cleaned_{entry['filename']}",
        mimetype="text/csv",
    )


@app.route("/api/dataset/<dataset_id>/distribution", methods=["GET"])
@token_required
def dataset_distribution(dataset_id):
    entry = db.get_dataset(dataset_id)
    if not entry:
        return jsonify({"error": "Dataset not found"}), 404

    df = _dataset_df_or_404(entry)
    if df is None:
        return jsonify({"error": "Dataset content not available in memory. Re-upload to continue."}), 410
    col_info = entry["col_info"]

    distributions = []

    for col in df.columns[:50]: # limit to 50 columns
        col_type = col_info["column_type_map"].get(col, "Unknown")
        insights = get_distribution_insights_df(df, col, col_type)
        
        plot_data = {}
        if col in col_info["continuous_numeric"]:
            series = df[col].dropna()
            if not series.empty:
                counts, bins = np.histogram(series, bins=30)
                plot_data = {"type": "histogram", "x": bins[:-1].tolist(), "y": counts.tolist()}
        elif col in col_info["discrete_numeric"]:
            vc = df[col].value_counts().head(30)
            plot_data = {"type": "bar", "x": vc.index.astype(str).tolist(), "y": vc.values.tolist()}
        elif col in col_info["categorical_cols"]:
            vc = df[col].value_counts().head(30)
            plot_data = {"type": "bar", "x": vc.index.astype(str).tolist(), "y": vc.values.tolist()}
        elif col in col_info["datetime_cols"]:
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            if not parsed.empty:
                vc = parsed.dt.month_name().value_counts().reindex([
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ]).dropna()
                plot_data = {"type": "bar", "x": vc.index.tolist(), "y": vc.values.tolist()}

        distributions.append({
            "column": col,
            "type": col_type,
            "insights": df_to_json(insights) if not insights.empty else [],
            "plot": plot_data
        })

    return jsonify({"distributions": distributions})


@app.route("/api/dataset/<dataset_id>/analysis", methods=["GET"])
@token_required
def dataset_analysis(dataset_id):
    entry = db.get_dataset(dataset_id)
    if not entry:
        return jsonify({"error": "Dataset not found"}), 404

    df = _dataset_df_or_404(entry)
    if df is None:
        return jsonify({"error": "Dataset content not available in memory. Re-upload to continue."}), 410
    col_info = entry["col_info"]

    all_numeric = col_info["continuous_numeric"] + col_info["discrete_numeric"]
    cat_cols = col_info["categorical_cols"]
    cont_cols = col_info["continuous_numeric"]
    disc_cols = col_info["discrete_numeric"]

    result = {}

    # Descriptive
    if all_numeric:
        result["numerical_stats"] = df_to_json(get_numerical_descriptive_df(df, all_numeric).reset_index().rename(columns={"index": "column"}))
    else:
        result["numerical_stats"] = []

    if cat_cols:
        result["categorical_stats"] = df_to_json(get_categorical_descriptive_df(df, cat_cols))
    else:
        result["categorical_stats"] = []

    # Diagnostic
    if cont_cols and len(cont_cols) > 1:
        corr_matrix = get_pearson_corr_matrix(df, cont_cols)
        result["correlation_matrix"] = {
            "columns": cont_cols,
            "values": corr_matrix.round(4).values.tolist(),
        }

        spearman = get_spearman_correlation_df(df, cont_cols, threshold=0.6)
        result["spearman"] = df_to_json(spearman)

        kendall = get_kendall_correlation_df(df, cont_cols, threshold=0.5)
        result["kendall"] = df_to_json(kendall)
    else:
        result["correlation_matrix"] = None
        result["spearman"] = []
        result["kendall"] = []

    # Prescriptive
    if all_numeric:
        result["numeric_prescriptive"] = df_to_json(numeric_prescriptive_df(df, cont_cols, disc_cols))
    else:
        result["numeric_prescriptive"] = []

    if cat_cols:
        result["categorical_prescriptive"] = df_to_json(categorical_prescriptive_df(df, cat_cols))
    else:
        result["categorical_prescriptive"] = []

    if cont_cols and len(cont_cols) > 1:
        corr_diag = get_numeric_correlation_diagnostics(df, cont_cols, threshold=0.8)
        corr_pres = correlation_prescriptive_df(corr_diag)
        result["correlation_prescriptive"] = df_to_json(corr_pres)
    else:
        result["correlation_prescriptive"] = []

    result["dataset_prescriptive"] = df_to_json(dataset_prescriptive_summary(df))

    return jsonify(result)


@app.route("/api/dataset/<dataset_id>/models", methods=["GET"])
@token_required
def dataset_models(dataset_id):
    entry = db.get_dataset(dataset_id)
    if not entry:
        return jsonify({"error": "Dataset not found"}), 404

    df = _dataset_df_or_404(entry)
    if df is None:
        return jsonify({"error": "Dataset content not available in memory. Re-upload to continue."}), 410

    cleaned_df = _clean_df_for_training(df)
    if cleaned_df.empty:
        return jsonify({"error": "Cleaned dataset is empty. Upload a richer dataset or relax cleaning."}), 400

    col_info = entry["col_info"]
    target_col = request.args.get("target")

    if not target_col or target_col not in cleaned_df.columns:
        return jsonify({"columns": cleaned_df.columns.tolist()})

    target_type, feature_df, model_df = get_training_plan(
        df=cleaned_df,
        target_col=target_col,
        continuous_numeric=col_info["continuous_numeric"],
        discrete_numeric=col_info["discrete_numeric"],
        categorical_cols=col_info["categorical_cols"],
        datetime_cols=col_info["datetime_cols"],
    )

    return jsonify({
        "target_type": target_type,
        "features": df_to_json(feature_df),
        "recommendations": df_to_json(model_df),
    })


@app.route("/api/dataset/<dataset_id>/models/baseline", methods=["POST"])
@token_required
def train_dataset_baseline(dataset_id):
    entry = db.get_dataset(dataset_id)
    if not entry:
        return jsonify({"error": "Dataset not found"}), 404

    data = request.get_json(silent=True) or {}
    target_col = data.get("target")
    feature_columns = data.get("feature_columns") or []
    test_size = float(data.get("test_size", 0.2))
    random_state = int(data.get("random_state", 42))

    if not target_col:
        return jsonify({"error": "Target column is required"}), 400

    raw_df = _dataset_df_or_404(entry)
    if raw_df is None:
        return jsonify({"error": "Dataset content not available in memory. Re-upload to continue."}), 410

    df = _clean_df_for_training(raw_df)
    if df.empty:
        return jsonify({"error": "Cleaned dataset is empty. Upload a richer dataset or relax cleaning."}), 400

    if target_col not in df.columns:
        return jsonify({"error": "Invalid target column"}), 400

    col_info = entry["col_info"]
    target_type, feature_df, _ = get_training_plan(
        df=df,
        target_col=target_col,
        continuous_numeric=col_info["continuous_numeric"],
        discrete_numeric=col_info["discrete_numeric"],
        categorical_cols=col_info["categorical_cols"],
        datetime_cols=col_info["datetime_cols"],
    )

    if feature_df.empty:
        return jsonify({"error": "No model-ready features found for baseline training"}), 400

    valid_features = set(feature_df["Feature"].tolist())
    selected_features = [col for col in feature_columns if col in valid_features] if feature_columns else list(valid_features)
    if not selected_features:
        return jsonify({"error": "Select at least one valid X column"}), 400

    try:
        result = train_baseline_model(
            df=df,
            target_col=target_col,
            feature_df=feature_df,
            target_type=target_type,
            test_size=test_size,
            random_state=random_state,
            selected_features=selected_features,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    pipeline = result.pop("pipeline", None)
    if pipeline is not None:
        key = _pipeline_key("baseline", dataset_id, target_col)
        buffer = io.BytesIO()
        joblib.dump(pipeline, buffer)
        PIPELINE_STORE[key] = buffer.getvalue()

    fit_assessment = _assess_fit(target_type, result.get("metrics", {}))

    response = {
        "target": target_col,
        "target_type": target_type,
        "test_size": test_size,
        "random_state": random_state,
        "feature_columns": selected_features,
        "task": result.get("task"),
        "metrics": result.get("metrics", {}),
        "fit_assessment": fit_assessment,
        "model_details": result.get("model_details", {}),
        "model_graph": result.get("model_graph"),
        "pipeline_download_url": f"/api/dataset/{dataset_id}/models/baseline/download?target={quote(target_col, safe='')}",
    }

    if "confusion_matrix" in result:
        response["confusion_matrix"] = result["confusion_matrix"]
    if "classification_report" in result:
        response["classification_report"] = result["classification_report"]

    return jsonify(response)


@app.route("/api/dataset/<dataset_id>/models/baseline/download", methods=["GET"])
@token_required
def download_baseline_pipeline(dataset_id):
    target_col = request.args.get("target", "")
    if not target_col:
        return jsonify({"error": "Target query parameter is required"}), 400

    key = _pipeline_key("baseline", dataset_id, target_col)
    pipeline_bytes = PIPELINE_STORE.get(key)
    if not pipeline_bytes:
        return jsonify({"error": "Baseline pipeline not found. Train it first."}), 404

    safe_target = _safe_filename_part(target_col) or "target"
    return send_file(
        io.BytesIO(pipeline_bytes),
        as_attachment=True,
        download_name=f"baseline_{dataset_id}_{safe_target}.joblib",
        mimetype="application/octet-stream",
    )


@app.route("/api/dataset/<dataset_id>/models/custom", methods=["POST"])
@token_required
def train_dataset_custom(dataset_id):
    entry = db.get_dataset(dataset_id)
    if not entry:
        return jsonify({"error": "Dataset not found"}), 404

    data = request.get_json(silent=True) or {}
    target_col = data.get("target")
    feature_columns = data.get("feature_columns") or []
    model_name = data.get("model_name")
    scaler_name = data.get("scaler_name", "StandardScaler")
    test_size = float(data.get("test_size", 0.2))
    random_state = int(data.get("random_state", 42))

    if not target_col:
        return jsonify({"error": "Target column is required"}), 400
    if not model_name:
        return jsonify({"error": "Model name is required"}), 400

    raw_df = _dataset_df_or_404(entry)
    if raw_df is None:
        return jsonify({"error": "Dataset content not available in memory. Re-upload to continue."}), 410

    df = _clean_df_for_training(raw_df)
    if df.empty:
        return jsonify({"error": "Cleaned dataset is empty. Upload a richer dataset or relax cleaning."}), 400

    if target_col not in df.columns:
        return jsonify({"error": "Invalid target column"}), 400

    col_info = entry["col_info"]
    target_type, feature_df, _ = get_training_plan(
        df=df,
        target_col=target_col,
        continuous_numeric=col_info["continuous_numeric"],
        discrete_numeric=col_info["discrete_numeric"],
        categorical_cols=col_info["categorical_cols"],
        datetime_cols=col_info["datetime_cols"],
    )

    if feature_df.empty:
        return jsonify({"error": "No model-ready features found for custom training"}), 400

    valid_features = set(feature_df["Feature"].tolist())
    selected_features = [col for col in feature_columns if col in valid_features] if feature_columns else list(valid_features)
    if not selected_features:
        return jsonify({"error": "Select at least one valid X column"}), 400

    try:
        result = train_custom_model(
            df=df,
            target_col=target_col,
            feature_df=feature_df,
            target_type=target_type,
            model_name=model_name,
            scaler_name=scaler_name,
            test_size=test_size,
            random_state=random_state,
            selected_features=selected_features,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    pipeline = result.pop("pipeline", None)
    if pipeline is not None:
        key = _pipeline_key("custom", dataset_id, target_col)
        buffer = io.BytesIO()
        joblib.dump(pipeline, buffer)
        PIPELINE_STORE[key] = buffer.getvalue()

    fit_assessment = _assess_fit(target_type, result.get("metrics", {}))

    response = {
        "target": target_col,
        "target_type": target_type,
        "task": result.get("task"),
        "model_name": result.get("model_name"),
        "scaler_name": result.get("scaler_name"),
        "test_size": test_size,
        "random_state": random_state,
        "feature_columns": selected_features,
        "metrics": result.get("metrics", {}),
        "fit_assessment": fit_assessment,
        "model_details": result.get("model_details", {}),
        "model_graph": result.get("model_graph"),
        "pipeline_download_url": f"/api/dataset/{dataset_id}/models/custom/download?target={quote(target_col, safe='')}",
    }

    if "confusion_matrix" in result:
        response["confusion_matrix"] = result["confusion_matrix"]
    if "classification_report" in result:
        response["classification_report"] = result["classification_report"]

    return jsonify(response)


@app.route("/api/dataset/<dataset_id>/models/custom/download", methods=["GET"])
@token_required
def download_custom_pipeline(dataset_id):
    target_col = request.args.get("target", "")
    if not target_col:
        return jsonify({"error": "Target query parameter is required"}), 400

    key = _pipeline_key("custom", dataset_id, target_col)
    pipeline_bytes = PIPELINE_STORE.get(key)
    if not pipeline_bytes:
        return jsonify({"error": "Custom pipeline not found. Train it first."}), 404

    safe_target = _safe_filename_part(target_col) or "target"
    return send_file(
        io.BytesIO(pipeline_bytes),
        as_attachment=True,
        download_name=f"custom_{dataset_id}_{safe_target}.joblib",
        mimetype="application/octet-stream",
    )


@app.route("/api/dataset/<dataset_id>/chat", methods=["POST"])
@token_required
def chat_about_dataset(dataset_id):
    entry = db.get_dataset(dataset_id)
    if not entry:
        return jsonify({"error": "Dataset not found"}), 404

    raw_df = _dataset_df_or_404(entry)
    if raw_df is None:
        return jsonify({"error": "Dataset content not available in memory. Re-upload to continue."}), 410

    cleaned_df = _clean_df_for_training(raw_df)
    if cleaned_df.empty:
        return jsonify({"error": "Cleaned dataset is empty. Upload a richer dataset or relax cleaning."}), 400

    body = request.get_json(silent=True) or {}
    question = str(body.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        context = _build_dataset_chat_context(cleaned_df)
        answer = _ask_gemini_about_dataset(question, context)
        return jsonify({"answer": answer})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": f"Chat service failed: {str(exc)}"}), 502
    except Exception:
        return jsonify({"error": "Unexpected chatbot error"}), 500


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Server running on http://localhost:5000")
    app_debug = (os.environ.get("FLASK_DEBUG", "false").strip().lower() in ("1", "true", "yes", "on"))
    app_host = os.environ.get("FLASK_HOST", "127.0.0.1").strip() or "127.0.0.1"
    app_port = int(os.environ.get("PORT", "5000"))
    app.run(debug=app_debug, host=app_host, port=app_port)
