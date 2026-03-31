# Dataset Cleaner and Analyzer

A full-stack data intelligence application for CSV datasets.

This project provides:
- User authentication (email/password + Google login)
- Dataset upload, profiling, cleaning, and analysis
- Model recommendation and training (baseline + custom)
- Model diagnostics (fit checks, metrics, train-vs-test, model details)
- Dataset-aware AI chatbot (Gemini)
- Rich PDF report generation (section reports + combined report)

## 1. Tech Stack

Backend:
- Python 3.10+
- Flask
- SQLite (metadata + persisted dataset blobs)
- pandas, numpy, scikit-learn
- JWT auth (PyJWT)
- Google token verification (google-auth)

Frontend:
- Vanilla HTML/CSS/JavaScript
- Plotly.js for charts
- jsPDF + AutoTable for report generation
- Google Identity Services

## 2. Current Project Structure

```text
Dataset Cleaner and Analyzer/
├─ api/
│  └─ python_backend/
│     ├─ server.py                 # Main Flask server + API endpoints
│     ├─ db.py                     # SQLite schema and DB operations
│     ├─ baseline_model_json.py    # Baseline/custom training pipelines
│     ├─ app.db                    # SQLite database file (local, ignored)
│     └─ __pycache__/
├─ frontend/
│  ├─ index.html                   # App markup
│  ├─ styles.css                   # App styles
│  ├─ app.js                       # Frontend app logic
│  └─ assets/
├─ uploads/                        # Optional local upload artifacts (ignored)
├─ column_identification.py        # Column typing utilities
├─ data_analysis.py                # Analysis helpers
├─ models.py                       # Model recommendation logic
├─ utils.py                        # Cleaning + utility functions
├─ Not Used/                       # Archived legacy files
├─ .env.example                    # Environment template
├─ .gitignore
├─ requirements.txt
├─ Dockerfile
├─ .dockerignore
└─ README.md
```

## 3. Features

### 3.1 Dataset Workflow
- Upload CSV
- Auto-detect column types
- Data quality summary
- Cleaning operations and cleaned preview
- Distribution analytics + correlation diagnostics

### 3.2 Modeling Workflow
- Select target column
- Auto model recommendations by task type
- Train baseline model
- Train custom model (model/scaler/test split/random state)
- Download trained pipeline
- View model details:
  - Intercept
  - Coefficients / feature importances
  - Compact estimator params
  - Model contribution graph

### 3.3 Reports
- Overview report
- Analysis report
- Models report
- Combined profile report (overview + analysis + graphs + model sections when available)

### 3.4 Authentication
- Email/password login + signup
- Google login/signup via Google Identity Services
- JWT-protected API endpoints

### 3.5 AI Chatbot
- Dataset-context grounded Q&A
- Gemini API integration via backend endpoint

## 4. Environment Variables

Create a local `.env` file from `.env.example`.

Required:
- `APP_SECRET_KEY`: Secret key for JWT signing
- `GOOGLE_CLIENT_ID`: Google OAuth client ID
- `GEMINI_API_KEY`: Gemini API key

Optional:
- `PORT`: Server port (default 5000)
- `DATABASE_URL`: PostgreSQL connection string (if unset, app falls back to local SQLite)
- `CORS_ALLOWED_ORIGINS`: Comma-separated allowed browser origins for API calls

Example:
- `CORS_ALLOWED_ORIGINS=https://your-frontend-domain.com,https://www.your-frontend-domain.com`

## 5. Local Development Setup

### 5.1 Create and Activate Virtual Environment
Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 5.3 Configure Environment

```bash
cp .env.example .env
```

Then edit `.env` with your credentials.

### 5.4 Run the Backend

```bash
python api/python_backend/server.py
```

App URL:
- `http://127.0.0.1:5000`

The Flask server serves both API and frontend static files.

## 6. Security Checklist Before GitHub Push

- Never commit `.env`
- Rotate any API keys that were previously hardcoded/exposed
- Keep database files (`app.db`) out of git
- Keep local uploads and temporary artifacts out of git

## 7. API Surface (High-Level)

Auth:
- `POST /api/auth/signup`
- `POST /api/auth/login`
- `POST /api/auth/google`
- `GET /api/auth/google-config`
- `GET /api/auth/me`

User/Profile:
- `PUT /api/user/profile`
- `GET /api/user/datasets`

Dataset:
- `POST /api/dataset/upload`
- `GET /api/dataset/<id>/resume`
- `GET /api/dataset/<id>/overview`
- `GET /api/dataset/<id>/analysis`
- `GET /api/dataset/<id>/distribution`
- `POST /api/dataset/<id>/clean`
- `GET /api/dataset/<id>/download`

Models:
- `GET /api/dataset/<id>/models`
- `POST /api/dataset/<id>/models/baseline`
- `GET /api/dataset/<id>/models/baseline/download`
- `POST /api/dataset/<id>/models/custom`
- `GET /api/dataset/<id>/models/custom/download`

Chat:
- `POST /api/dataset/<id>/chat`

## 8. Recommended Deployment Strategy

### Best Platform for This Project
For this architecture (Flask app serving frontend + API + report generation), the best practical path is:

- **Render Web Service** for app hosting
- **Managed Postgres** (Render/Supabase/Neon) for production database
- **Object storage** (S3/R2/Supabase Storage) for large dataset artifacts if needed

Why this is best:
- Very simple CI/CD from GitHub
- Easy environment variable management
- Good Python support
- Predictable deployment workflow for solo/small team projects

Important note:
- This project currently uses SQLite. SQLite is fine for local/dev, but for production use Postgres for reliability and concurrency.

## 9. Docker Deployment

Build:

```bash
docker build -t dataset-cleaner-analyzer .
```

Run:

```bash
docker run -p 5000:5000 --env-file .env dataset-cleaner-analyzer
```

## 9.1 Apply PostgreSQL (Scalable Setup)

The backend now supports both:
- SQLite (default when `DATABASE_URL` is not set)
- PostgreSQL (enabled automatically when `DATABASE_URL` is set)

### Quick local Postgres setup (Docker)

```bash
docker run --name dca-postgres \
  -e POSTGRES_USER=dca_user \
  -e POSTGRES_PASSWORD=dca_pass \
  -e POSTGRES_DB=dca_db \
  -p 5432:5432 -d postgres:16
```

Set in `.env`:

```env
DATABASE_URL=postgresql://dca_user:dca_pass@localhost:5432/dca_db
```

Then restart backend. Tables are auto-created at startup.

### Production Postgres setup

1. Create a managed Postgres database (Render Postgres, Supabase, Neon, or RDS).
2. Copy the provider connection URL into `DATABASE_URL`.
3. Redeploy the backend service.
4. Verify user signup/login and dataset upload paths.

### Data migration from SQLite to Postgres

If you have existing local SQLite data, migrate users/datasets once via script or SQL copy process.
Suggested approach:
- Export rows from SQLite `users` and `datasets` tables.
- Insert into Postgres preserving `id` fields.
- Validate row counts and sampled records before cutover.

## 10. Deploying on Render (Step-by-Step)

1. Push repo to GitHub.
2. Create a new Web Service in Render.
3. Connect your repository.
4. Set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn --chdir api/python_backend server:app`
5. Add environment variables from `.env.example`.
6. Deploy and open the public URL.

## 11. Recommended Production Improvements

- Migrate SQLite to Postgres
- Move dataset blobs to object storage
- Add rate limiting for auth and chat endpoints
- Add request logging and error monitoring (Sentry/OpenTelemetry)
- Add automated tests (pytest + Playwright)
- Add CI pipeline (lint, tests, build)

## 12. Troubleshooting

### Server starts but login fails
- Verify `APP_SECRET_KEY` is set and stable
- Verify DB file permissions

### Google auth button visible but fails
- Ensure `GOOGLE_CLIENT_ID` is correct
- Add deployed domain/origin in Google Cloud OAuth settings

### Chatbot returns config error
- Ensure `GEMINI_API_KEY` is set in environment

### Combined reports missing model sections
- Train baseline/custom models for the selected dataset in the same browser session
- Use the same dataset ID before generating combined report

## 13. License

Add your preferred license file (MIT recommended) before publishing publicly.
