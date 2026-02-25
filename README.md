# 🏥 AI-Powered Hybrid Clinical Decision Support & Remote Monitoring System

A production-ready Streamlit application combining clinical rule-based scoring (NEWS) with a **pre-trained Keras MLP neural network** (`risk_model.h5`) for explainable, 3-class health risk assessment.

---

## 🏗️ Model Architecture

| Property | Detail |
|----------|--------|
| Framework | TensorFlow / Keras (`.h5` format) |
| Type | Sequential MLP |
| Input features | **11** (see below) |
| Hidden layers | Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dropout(0.3) → Dense(16, relu) |
| Output | Dense(3, softmax) |
| Classes | `High (0)` · `Low (1)` · `Medium (2)` *(alphabetical LabelEncoder order)* |
| Scaler | MinMaxScaler on 6 numeric vitals |

### Feature Vector (11 inputs)

| # | Feature | Preprocessing |
|---|---------|---------------|
| 0 | Respiratory Rate | MinMaxScaled |
| 1 | Oxygen Saturation | MinMaxScaled |
| 2 | O2 Scale | MinMaxScaled |
| 3 | Systolic BP | MinMaxScaled |
| 4 | Heart Rate | MinMaxScaled |
| 5 | Temperature | MinMaxScaled |
| 6 | consciousness_C | OHE (base = A, drop_first) |
| 7 | consciousness_P | OHE |
| 8 | consciousness_U | OHE |
| 9 | consciousness_V | OHE |
| 10 | On_Oxygen | Binary 0/1 |

---

## 🚀 Quick Start (Local)

### 1. Clone & install

```bash
git clone https://github.com/your-username/clinical-dss.git
cd clinical-dss
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — SQLite works out of the box for local dev
```

### 3. Ensure model files are present

```
app/models/risk_model.h5   ← your trained Keras model
app/models/scaler.pkl      ← your fitted MinMaxScaler
```

These are already included in the repository.

### 4. Run

```bash
streamlit run app/main.py
```

---

## ☁️ Deploy to Streamlit Cloud

### Steps

1. **Push to GitHub** (including `app/models/` directory with both model files)

```bash
git add .
git commit -m "Deploy clinical DSS"
git push origin main
```

2. **Connect on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - New app → select repo → Main file: `app/main.py` → Deploy

3. **Add Secrets** (Settings → Secrets):

```toml
DATABASE_URL  = "postgresql://user:pass@host:5432/dbname"
SECRET_KEY    = "your-secret-key"
BCRYPT_ROUNDS = "12"
```

4. **Free PostgreSQL options**: [Neon](https://neon.tech) · [Supabase](https://supabase.com)

---

## 📁 Project Structure

```
.
├── app/
│   ├── main.py                  # Entry point
│   ├── config.py                # Env vars, model paths, class mappings
│   ├── database.py              # SQLAlchemy ORM (5 tables)
│   ├── auth.py                  # Bcrypt auth, sessions, audit logging
│   ├── utils.py                 # Two-layer clinical validation, global CSS
│   ├── risk_engine.py           # Hybrid: NEWS rules + Keras model + SHAP
│   ├── pdf_generator.py         # ReportLab clinical PDF
│   ├── models/
│   │   ├── risk_model.h5        # Pre-trained Keras MLP
│   │   └── scaler.pkl           # MinMaxScaler (6 vitals)
│   ├── pages/
│   │   ├── 1_Landing.py         # Healthcare SaaS landing page
│   │   ├── 2_Login.py
│   │   ├── 3_Register.py
│   │   ├── 4_Patient_Dashboard.py
│   │   ├── 5_Doctor_Dashboard.py
│   │   ├── 6_Admin_Dashboard.py
│   │   └── 7_Assessment.py      # Assessment form + result display
│   └── components/
│       ├── navbar.py            # Role-aware sidebar
│       ├── charts.py            # Plotly: trend, pie, radar, probability bar
│       └── alerts.py            # Risk alert banners
├── train_model.py               # Reference training script (for retraining)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 👤 Default Accounts (auto-seeded on first run)

| Role | Email | Password |
|------|-------|----------|
| Admin | admin@healthsystem.com | Admin@1234 |
| Doctor | doctor@healthsystem.com | Doctor@1234 |
| Patient | Register via the app | — |

> ⚠️ **Change all default passwords before production deployment!**

---

## 🔒 Security

- Bcrypt password hashing (configurable cost rounds)
- Role-based access control: `patient` · `doctor` · `admin`
- Streamlit session state management
- Environment variable secrets (never hardcoded)
- Full audit logging for every user action
- Two-layer vital sign validation (hard block + soft warnings)

---

## ⚠️ Disclaimer

This system provides **clinical decision support only**. It is not a substitute for professional medical judgment. Every assessment is mandatorily routed to a licensed physician for review before any clinical action is taken.
