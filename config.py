"""
config.py - Application configuration and environment variable management
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./health_monitor.db")

# ── Security ───────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
BCRYPT_ROUNDS = int(os.getenv("BCRYPT_ROUNDS", "12"))

# ── App metadata ───────────────────────────────────────────────────────────────
APP_NAME = "AI-Powered Hybrid Clinical Decision Support System"
APP_VERSION = "1.0.0"
APP_ICON = "🏥"

# ── Risk Levels  (3-class: Low / Medium / High) ────────────────────────────────
# The Keras model outputs 3 classes.
# "Normal" vitals from the original dataset were mapped to "Low" during training.
RISK_LEVELS = {
    "Low":    {"color": "#28a745", "bg": "#d4edda", "icon": "✅"},
    "Medium": {"color": "#ffc107", "bg": "#fff3cd", "icon": "⚠️"},
    "High":   {"color": "#dc3545", "bg": "#f8d7da", "icon": "🚨"},
}

# ── ML class-index mapping  ────────────────────────────────────────────────────
# sparse_categorical_crossentropy with LabelEncoder → alphabetical order
# High=0, Low=1, Medium=2
ML_CLASS_LABELS = {0: "High", 1: "Low", 2: "Medium"}

# ── Feature engineering constants ─────────────────────────────────────────────
# MinMaxScaler was fitted on these 6 columns (exact order matters)
SCALER_FEATURES = [
    "Respiratory_Rate", "Oxygen_Saturation", "O2_Scale",
    "Systolic_BP", "Heart_Rate", "Temperature",
]

# One-hot encoded consciousness: drop_first=True → base = 'A'
# Columns produced (alphabetical after dropping A):
CONSCIOUSNESS_OHE_COLS = ["consciousness_C", "consciousness_P",
                           "consciousness_U", "consciousness_V"]

# Full 11-feature vector:
# [RR, SpO2, O2Scale, SBP, HR, Temp (scaled), C, P, U, V (OHE), On_Oxygen]
MODEL_FEATURE_COUNT = 11

# ── Model file paths ───────────────────────────────────────────────────────────
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH  = os.path.join(_MODELS_DIR, "risk_model.h5")
SCALER_PATH = os.path.join(_MODELS_DIR, "scaler.pkl")

# ── Clinical hard-limit validation (physiologically impossible values) ──────────
CLINICAL_LIMITS = {
    "respiratory_rate":   (0,   70),
    "oxygen_saturation":  (50,  100),
    "systolic_bp":        (50,  300),
    "heart_rate":         (20,  250),
    "temperature":        (30.0, 44.0),
}

# ── Clinical soft-warning thresholds (dangerous but possible) ──────────────────
CLINICAL_WARNINGS = {
    "respiratory_rate":   (8,   30),
    "oxygen_saturation":  (85,  100),
    "systolic_bp":        (80,  220),
    "heart_rate":         (40,  180),
    "temperature":        (35.0, 40.5),
}

# ── Session state keys ─────────────────────────────────────────────────────────
SESSION_USER_ID    = "user_id"
SESSION_USER_ROLE  = "user_role"
SESSION_USER_NAME  = "user_name"
SESSION_USER_EMAIL = "user_email"
