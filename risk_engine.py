"""
risk_engine.py - Hybrid Rule-Based + Keras MLP Risk Classification Engine

Model details (from H5 inspection):
  Input  : 11 features
           [RR, SpO2, O2Scale, SBP, HR, Temp (MinMaxScaled),
            consciousness_C, consciousness_P, consciousness_U, consciousness_V (OHE, base=A),
            On_Oxygen]
  Layers : Dense(64,relu) → Dropout(0.3) → Dense(32,relu) → Dropout(0.3) → Dense(16,relu) → Dense(3,softmax)
  Output : 3 classes  — {0: High, 1: Low, 2: Medium}  (alphabetical LabelEncoder order)
  Scaler : MinMaxScaler on 6 numeric vitals
"""
import os
import pickle
import warnings
import numpy as np
import logging

from config import (
    MODEL_PATH, SCALER_PATH, ML_CLASS_LABELS,
    SCALER_FEATURES, CONSCIOUSNESS_OHE_COLS, MODEL_FEATURE_COUNT,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RULE-BASED SCORING  (NEWS-style simplified)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rule_score(vitals: dict) -> tuple:
    """
    Compute a simplified NEWS score from the raw vitals dict.
    Returns (score: int, abnormal_descriptions: list[str]).
    """
    score     = 0
    abnormals = []

    rr   = vitals.get("respiratory_rate",  18)
    spo2 = vitals.get("oxygen_saturation", 98)
    sbp  = vitals.get("systolic_bp",       120)
    hr   = vitals.get("heart_rate",        80)
    temp = vitals.get("temperature",       37.0)
    cons = vitals.get("consciousness",     "A")
    on_o2 = vitals.get("on_oxygen",        0)
    o2_sc = vitals.get("o2_scale",         1)

    # ── Respiratory Rate ──────────────────────────────────────────────────────
    if rr <= 8:
        score += 3; abnormals.append(f"Respiratory rate critically low ({rr} bpm)")
    elif rr <= 11:
        score += 1; abnormals.append(f"Respiratory rate low ({rr} bpm)")
    elif rr <= 20:
        pass  # normal
    elif rr <= 24:
        score += 2; abnormals.append(f"Respiratory rate elevated ({rr} bpm)")
    else:
        score += 3; abnormals.append(f"Respiratory rate critically high ({rr} bpm)")

    # ── Oxygen Saturation (Scale-aware) ───────────────────────────────────────
    if int(o2_sc) == 2:          # COPD / hypercapnic target 88–92 %
        if spo2 <= 83:
            score += 3; abnormals.append(f"SpO2 critically low for COPD scale ({spo2}%)")
        elif spo2 <= 85:
            score += 2; abnormals.append(f"SpO2 low for COPD scale ({spo2}%)")
        elif spo2 <= 87:
            score += 1; abnormals.append(f"SpO2 borderline for COPD scale ({spo2}%)")
        elif spo2 <= 92:
            pass  # target range
        else:
            score += 2
            abnormals.append(f"SpO2 above COPD target — hypercapnia risk ({spo2}%)")
    else:                        # Standard scale
        if spo2 <= 91:
            score += 3; abnormals.append(f"Oxygen saturation critically low ({spo2}%)")
        elif spo2 <= 93:
            score += 2; abnormals.append(f"Oxygen saturation low ({spo2}%)")
        elif spo2 <= 95:
            score += 1; abnormals.append(f"Oxygen saturation borderline ({spo2}%)")

    # ── Supplemental Oxygen ───────────────────────────────────────────────────
    if int(on_o2) == 1:
        score += 2; abnormals.append("Patient is on supplemental oxygen")

    # ── Systolic BP ───────────────────────────────────────────────────────────
    if sbp <= 90:
        score += 3; abnormals.append(f"Systolic BP critically low ({sbp} mmHg)")
    elif sbp <= 100:
        score += 2; abnormals.append(f"Systolic BP low ({sbp} mmHg)")
    elif sbp <= 110:
        score += 1; abnormals.append(f"Systolic BP borderline low ({sbp} mmHg)")
    elif sbp <= 219:
        pass
    else:
        score += 3; abnormals.append(f"Systolic BP critically high ({sbp} mmHg)")

    # ── Heart Rate ────────────────────────────────────────────────────────────
    if hr <= 40:
        score += 3; abnormals.append(f"Heart rate critically low ({hr} bpm)")
    elif hr <= 50:
        score += 1; abnormals.append(f"Heart rate low ({hr} bpm)")
    elif hr <= 90:
        pass
    elif hr <= 110:
        score += 1; abnormals.append(f"Heart rate elevated ({hr} bpm)")
    elif hr <= 130:
        score += 2; abnormals.append(f"Heart rate high ({hr} bpm)")
    else:
        score += 3; abnormals.append(f"Heart rate critically high ({hr} bpm)")

    # ── Temperature ───────────────────────────────────────────────────────────
    if temp <= 35.0:
        score += 3; abnormals.append(f"Temperature critically low — hypothermia risk ({temp}°C)")
    elif temp <= 36.0:
        score += 1; abnormals.append(f"Temperature low ({temp}°C)")
    elif temp <= 38.0:
        pass
    elif temp <= 39.0:
        score += 1; abnormals.append(f"Temperature elevated — fever ({temp}°C)")
    else:
        score += 2; abnormals.append(f"Temperature high — high fever ({temp}°C)")

    # ── Consciousness (ACVPU) ─────────────────────────────────────────────────
    if cons == "A":
        pass
    elif cons in ("C", "V"):
        score += 3; abnormals.append(f"Altered consciousness: {cons}")
    elif cons in ("P", "U"):
        score += 3; abnormals.append(f"Severely reduced consciousness: {cons}")

    return score, abnormals


def rule_score_to_risk(score: int) -> str:
    """Convert NEWS aggregate score → risk label."""
    if score <= 3:
        return "Low"
    elif score <= 6:
        return "Medium"
    else:
        return "High"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. KERAS MODEL + SCALER LOADING
# ═══════════════════════════════════════════════════════════════════════════════

_model_cache  = None
_scaler_cache = None


def _load_artifacts():
    """Load and cache the Keras model + MinMaxScaler. Returns (model, scaler)."""
    global _model_cache, _scaler_cache

    if _model_cache is not None:
        return _model_cache, _scaler_cache

    # ── Scaler ────────────────────────────────────────────────────────────────
    try:
        with open(SCALER_PATH, "rb") as f:
            _scaler_cache = pickle.load(f)
        logger.info("MinMaxScaler loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        _scaler_cache = None

    # ── Keras model ───────────────────────────────────────────────────────────
    try:
        # Lazy import so app doesn't crash if TF is not installed yet
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        _model_cache = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("Keras model loaded successfully.")
    except ImportError:
        logger.warning("TensorFlow not installed — ML predictions disabled.")
        _model_cache = None
    except Exception as e:
        logger.error(f"Failed to load Keras model: {e}")
        _model_cache = None

    return _model_cache, _scaler_cache


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING  (must exactly mirror training pre-processing)
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_vector(vitals: dict, scaler) -> np.ndarray:
    """
    Convert a raw vitals dict into the 11-dim feature vector the model expects.

    Feature vector layout:
        [0]  Respiratory_Rate   (MinMaxScaled)
        [1]  Oxygen_Saturation  (MinMaxScaled)
        [2]  O2_Scale           (MinMaxScaled)
        [3]  Systolic_BP        (MinMaxScaled)
        [4]  Heart_Rate         (MinMaxScaled)
        [5]  Temperature        (MinMaxScaled)
        [6]  consciousness_C    (0/1 OHE, base = A)
        [7]  consciousness_P    (0/1 OHE)
        [8]  consciousness_U    (0/1 OHE)
        [9]  consciousness_V    (0/1 OHE)
        [10] On_Oxygen          (binary 0/1)
    """
    cons = str(vitals.get("consciousness", "A")).upper()

    # ── Scale the 6 numeric vitals ────────────────────────────────────────────
    numeric = np.array([[
        vitals["respiratory_rate"],
        vitals["oxygen_saturation"],
        vitals["o2_scale"],
        vitals["systolic_bp"],
        vitals["heart_rate"],
        vitals["temperature"],
    ]], dtype=float)

    if scaler is not None:
        scaled = scaler.transform(numeric)[0]
    else:
        # Fallback: manual min-max using scaler's known data_min/data_max
        data_min = np.array([12.,  74.,  1.,  50.,  64., 35.6])
        data_max = np.array([40., 100.,  2., 144., 163., 41.8])
        scaled = (numeric[0] - data_min) / (data_max - data_min + 1e-8)
        scaled = np.clip(scaled, 0, 1)

    # ── One-hot encode consciousness (drop A as base) ─────────────────────────
    ohe = {
        "consciousness_C": 1 if cons == "C" else 0,
        "consciousness_P": 1 if cons == "P" else 0,
        "consciousness_U": 1 if cons == "U" else 0,
        "consciousness_V": 1 if cons == "V" else 0,
    }

    on_oxygen = int(vitals.get("on_oxygen", 0))

    feature_vec = np.concatenate([
        scaled,
        [ohe["consciousness_C"], ohe["consciousness_P"],
         ohe["consciousness_U"], ohe["consciousness_V"]],
        [on_oxygen],
    ]).astype(np.float32)

    assert feature_vec.shape == (MODEL_FEATURE_COUNT,), \
        f"Feature vector has wrong shape: {feature_vec.shape}"

    return feature_vec.reshape(1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ML PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def predict_keras(vitals: dict, model, scaler) -> tuple:
    """
    Run the Keras model and return (risk_label: str, confidence: float).
    Returns (None, None) on failure.
    """
    try:
        x = build_feature_vector(vitals, scaler)
        probs = model.predict(x, verbose=0)[0]           # shape (3,)
        class_idx  = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        risk_label = ML_CLASS_LABELS.get(class_idx, "Low")
        return risk_label, confidence, probs.tolist()
    except Exception as e:
        logger.error(f"Keras prediction error: {e}")
        return None, None, None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HYBRID DECISION  (higher risk wins when rule and ML disagree)
# ═══════════════════════════════════════════════════════════════════════════════

_RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def hybrid_decision(rule_risk: str, ml_risk: str | None) -> str:
    if ml_risk is None:
        return rule_risk
    return rule_risk if _RISK_ORDER[rule_risk] >= _RISK_ORDER.get(ml_risk, 0) else ml_risk


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SHAP EXPLAINABILITY  (on the Keras model via shap.GradientExplainer)
# ═══════════════════════════════════════════════════════════════════════════════

_FEATURE_NAMES = [
    "Respiratory Rate", "Oxygen Saturation", "O2 Scale",
    "Systolic BP", "Heart Rate", "Temperature",
    "Consciousness (C)", "Consciousness (P)",
    "Consciousness (U)", "Consciousness (V)",
    "On Oxygen",
]


def get_shap_explanation(vitals: dict, model, scaler, top_n: int = 4) -> list:
    """
    Return [(feature_name, shap_value), ...] for the top_n most influential features.
    Falls back to an empty list if SHAP or TF is unavailable.
    """
    try:
        import shap
        import tensorflow as tf

        x = build_feature_vector(vitals, scaler)           # shape (1, 11)

        # Use a zero background for speed (single sample explanation)
        background = np.zeros((1, MODEL_FEATURE_COUNT), dtype=np.float32)
        explainer  = shap.GradientExplainer(model, background)
        shap_vals  = explainer.shap_values(x)              # list of 3 arrays

        # Aggregate across output classes (mean absolute)
        if isinstance(shap_vals, list):
            importance = np.abs(np.array(shap_vals)).mean(axis=0)[0]
        else:
            importance = np.abs(shap_vals[0])

        top_idx = np.argsort(importance)[::-1][:top_n]
        return [(_FEATURE_NAMES[i], float(importance[i])) for i in top_idx]

    except Exception as e:
        logger.warning(f"SHAP explanation unavailable: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

_RECOMMENDATIONS = {
    "Low": (
        "✅ Vital signs are within an acceptable range. Continue routine monitoring. "
        "Reassess if new symptoms develop or your condition changes."
    ),
    "Medium": (
        "⚠️ Some vital signs require clinical attention. Increase monitoring frequency. "
        "Contact your care team and follow up within 24 hours."
    ),
    "High": (
        "🚨 Urgent clinical attention required. Contact your doctor or emergency services "
        "immediately. Do not delay — critical vital sign abnormalities detected."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_assessment(vitals: dict) -> dict:
    """
    Execute the complete hybrid assessment pipeline.

    Args:
        vitals: dict with keys:
            respiratory_rate, oxygen_saturation, o2_scale,
            systolic_bp, heart_rate, temperature,
            consciousness (A/C/V/P/U), on_oxygen (0/1)

    Returns:
        dict with: rule_score, ml_prediction, ml_probability, ml_class_probs,
                   final_risk, explanation, recommendation
    """
    # ── Step 1: Rule-based ────────────────────────────────────────────────────
    rule_score, abnormals = compute_rule_score(vitals)
    rule_risk = rule_score_to_risk(rule_score)

    # ── Step 2-4: Keras prediction ────────────────────────────────────────────
    model, scaler = _load_artifacts()
    ml_risk, ml_conf, ml_probs = predict_keras(vitals, model, scaler) \
        if model is not None else (None, None, None)

    # ── Step 5: Hybrid decision ───────────────────────────────────────────────
    final_risk = hybrid_decision(rule_risk, ml_risk)

    # ── Step 6: Explanation ───────────────────────────────────────────────────
    explanation_parts = []

    # Rule part
    if abnormals:
        explanation_parts.append(
            "**🔬 Rule-Based Analysis — Abnormal Vitals Detected:**\n" +
            "\n".join(f"- {a}" for a in abnormals)
        )
    else:
        explanation_parts.append(
            "**🔬 Rule-Based Analysis:** All vital signs are within clinically normal ranges."
        )

    # ML confidence
    if ml_risk is not None and ml_probs is not None:
        prob_str = (
            f"High: {ml_probs[0]:.1%} | Low: {ml_probs[1]:.1%} | Medium: {ml_probs[2]:.1%}"
        )
        explanation_parts.append(
            f"\n**🤖 AI Model Prediction:** {ml_risk} "
            f"(confidence: {ml_conf:.1%})\n"
            f"Class probabilities — {prob_str}"
        )

        # Hybrid resolution note
        if rule_risk != ml_risk:
            explanation_parts.append(
                f"\n**⚖️ Hybrid Resolution:** Rule-based predicted **{rule_risk}** "
                f"and AI model predicted **{ml_risk}**. "
                f"The higher risk level (**{final_risk}**) was adopted for patient safety."
            )

        # SHAP feature importance
        shap_feats = get_shap_explanation(vitals, model, scaler)
        if shap_feats:
            shap_lines = "\n".join(
                f"- {name}: {val:.4f}" for name, val in shap_feats
            )
            explanation_parts.append(
                f"\n**📊 Top Contributing Features (SHAP):**\n{shap_lines}"
            )
    else:
        explanation_parts.append(
            "\n**🤖 AI Model:** Unavailable — using rule-based assessment only. "
            "Ensure TensorFlow is installed and `risk_model.h5` is present in `/models`."
        )

    explanation   = "\n".join(explanation_parts)
    recommendation = _RECOMMENDATIONS[final_risk]

    return {
        "rule_score":      rule_score,
        "ml_prediction":   ml_risk  or "Unavailable",
        "ml_probability":  ml_conf  or 0.0,
        "ml_class_probs":  ml_probs or [],
        "final_risk":      final_risk,
        "explanation":     explanation,
        "recommendation":  recommendation,
    }
