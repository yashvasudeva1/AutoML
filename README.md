# Dataset Cleaner & Analyzer

A full-stack web app for cleaning, analyzing, and modeling CSV datasets — built with Flask + vanilla JS.

Upload a CSV, let it profile and clean your data, train a model, and export a PDF report. There's also an AI chatbot that actually knows what's in your dataset.

---

## What it does

- Upload any CSV and get an instant data quality summary
- Auto-detects column types and flags issues
- Clean your data (missing values, outliers, etc.) and preview the result
- Distribution plots and correlation diagnostics via Plotly
- Pick a target column, get model recommendations, train baseline or custom models
- Download trained pipelines
- Generate PDF reports — per section or one combined report
- Dataset-aware chatbot (powered by Gemini) that answers questions about your actual data
- Auth via email/password or Google login, JWT-protected throughout

---

## Stack

**Backend:** Python 3.10+, Flask, SQLite (dev) / PostgreSQL (prod), pandas, numpy, scikit-learn, PyJWT, google-auth

**Frontend:** Plain HTML/CSS/JS, Plotly.js, jsPDF + AutoTable, Google Identity Services

---

## Project layout

```
Dataset Cleaner and Analyzer/
├─ api/
│  └─ python_backend/
│     ├─ server.py                 # Flask server and all API routes
│     ├─ db.py                     # DB schema and helpers
│     ├─ baseline_model_json.py    # Training pipelines
│     └─ app.db                    # Local SQLite file (gitignored)
├─ frontend/
│  ├─ index.html
│  ├─ styles.css
│  └─ app.js
├─ column_identification.py
├─ data_analysis.py
├─ models.py
├─ utils.py
├─ .env.example
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

---

## Getting started locally

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your environment
cp .env.example .env
# Edit .env with your actual keys

# 4. Run the server
python api/python_backend/server.py
```

Open `http://127.0.0.1:5000`. The Flask server handles both the API and frontend static files.

---

## Environment variables

| Variable | Required | Notes |
|---|---|---|
| `APP_SECRET_KEY` | Yes | Used for JWT signing |
| `GOOGLE_CLIENT_ID` | Yes | Google OAuth client ID |
| `GEMINI_API_KEY` | Yes | Powers the chatbot |
| `PORT` | No | Defaults to 5000 |
| `DATABASE_URL` | No | PostgreSQL URL — falls back to SQLite if unset |
| `CORS_ALLOWED_ORIGINS` | No | Comma-separated list of allowed origins |

---

## Docker

```bash
docker build -t dataset-cleaner-analyzer .
docker run -p 5000:5000 --env-file .env dataset-cleaner-analyzer
```

---

## PostgreSQL setup

SQLite is fine for local dev. For anything beyond that, point `DATABASE_URL` at a Postgres instance and the app will use it automatically — tables are created on startup.

**Quick local Postgres via Docker:**

```bash
docker run --name dca-postgres \
  -e POSTGRES_USER=dca_user \
  -e POSTGRES_PASSWORD=dca_pass \
  -e POSTGRES_DB=dca_db \
  -p 5432:5432 -d postgres:16
```

Then in `.env`:
```
DATABASE_URL=postgresql://dca_user:dca_pass@localhost:5432/dca_db
```

For production, use a managed provider (Render Postgres, Supabase, Neon, RDS) and plug in the connection URL.

**Migrating existing SQLite data:** export rows from `users` and `datasets`, insert into Postgres preserving `id` fields, verify counts before switching over.

---

## Deploying on Render

1. Push to GitHub
2. Create a new Web Service and connect your repo
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn --chdir api/python_backend server:app`
5. Add your env variables from `.env.example`
6. Deploy

Render works well for this setup — simple CI/CD from GitHub, straightforward env management, good Python support. Pair it with Render Postgres or Supabase for the database.

---

## API routes

**Auth**
- `POST /api/auth/signup`
- `POST /api/auth/login`
- `POST /api/auth/google`
- `GET /api/auth/google-config`
- `GET /api/auth/me`

**User**
- `PUT /api/user/profile`
- `GET /api/user/datasets`

**Dataset**
- `POST /api/dataset/upload`
- `GET /api/dataset/<id>/resume`
- `GET /api/dataset/<id>/overview`
- `GET /api/dataset/<id>/analysis`
- `GET /api/dataset/<id>/distribution`
- `POST /api/dataset/<id>/clean`
- `GET /api/dataset/<id>/download`

**Models**
- `GET /api/dataset/<id>/models`
- `POST /api/dataset/<id>/models/baseline`
- `GET /api/dataset/<id>/models/baseline/download`
- `POST /api/dataset/<id>/models/custom`
- `GET /api/dataset/<id>/models/custom/download`

**Chat**
- `POST /api/dataset/<id>/chat`

---

## Before pushing to GitHub

- Don't commit `.env` — it's in `.gitignore` but double-check
- Rotate any API keys that were ever hardcoded anywhere
- Keep `app.db` and anything in `uploads/` out of git

---

## Known rough edges / things to improve

- Rate limiting on auth and chat endpoints would be good to add
- The combined PDF report only includes model sections if you've trained models in the same session — this is a known quirk
- No automated tests yet (pytest + Playwright would be the move)
- Logging and error tracking (Sentry) would help in production
- Moving dataset blobs to object storage (S3/R2) would be worth it at any real scale

---

## Troubleshooting

**Login not working** — check that `APP_SECRET_KEY` is set and hasn't changed between restarts. Also verify DB file permissions.

**Google login fails** — make sure your deployed origin is listed in Google Cloud OAuth settings, and `GOOGLE_CLIENT_ID` matches.

**Chatbot errors** — almost always a missing or invalid `GEMINI_API_KEY`.

**Combined report missing model sections** — train your baseline or custom model first, and make sure you're on the same dataset ID when generating the report.

---
