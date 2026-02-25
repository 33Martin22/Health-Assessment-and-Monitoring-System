"""
pages/7_Assessment.py - Guided vital signs assessment form with Keras model integration
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database import init_db, get_session, Patient, Assessment
from auth import seed_admin, require_auth, log_action
from utils import inject_global_css, validate_vitals_hard, validate_vitals_soft
from components.navbar import render_navbar
from components.alerts import show_risk_alert, show_clinical_warnings
from risk_engine import run_full_assessment

st.set_page_config(
    page_title="Assessment | Clinical DSS",
    page_icon="📋", layout="wide"
)

@st.cache_resource
def startup():
    init_db(); seed_admin(); return True
startup()

inject_global_css()
render_navbar()

user    = require_auth(allowed_roles=["patient"])
db      = get_session()
patient = db.query(Patient).filter(Patient.user_id == user["id"]).first()

if not patient:
    st.error("Patient profile not found. Please contact the system administrator.")
    db.close()
    st.stop()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("# 📋 New Health Assessment")
st.markdown(
    "Enter your current vital signs below. All fields are required. "
    "Your assessment will be automatically forwarded to your doctor for review."
)
st.divider()

# ── Measurement guide ─────────────────────────────────────────────────────────
with st.expander("ℹ️ How to measure your vitals correctly", expanded=False):
    st.markdown("""
| Vital | Normal Range | How to Measure |
|-------|-------------|----------------|
| **Respiratory Rate** | 12–20 breaths/min | Count chest rises for 60 seconds |
| **Oxygen Saturation** | 95–100 % | Read from pulse oximeter fingertip clip |
| **Systolic BP** | 90–120 mmHg | Top number on blood pressure monitor |
| **Heart Rate** | 60–100 bpm | Pulse oximeter or manual pulse count |
| **Temperature** | 36.0–37.5 °C | Oral or axillary thermometer in Celsius |
| **Consciousness** | Alert (A) | A=Alert · C=Confused · V=Voice · P=Pain · U=Unresponsive |
    """)

# ── Assessment form ───────────────────────────────────────────────────────────
with st.form("assessment_form", clear_on_submit=False):

    # ── Respiratory & Oxygen ──────────────────────────────────────────────────
    st.markdown("### 🫁 Respiratory & Oxygen Status")
    col1, col2, col3 = st.columns(3)

    with col1:
        respiratory_rate = st.number_input(
            "Respiratory Rate (breaths/min)",
            min_value=1, max_value=100, value=18, step=1,
            help="Count how many times your chest rises in 1 full minute."
        )
    with col2:
        oxygen_saturation = st.number_input(
            "Oxygen Saturation — SpO₂ (%)",
            min_value=50, max_value=100, value=98, step=1,
            help="Reading from a pulse oximeter."
        )
    with col3:
        o2_scale = st.selectbox(
            "O₂ Measurement Scale",
            options=[1, 2],
            format_func=lambda x: (
                "Scale 1 — Standard (most patients)"
                if x == 1
                else "Scale 2 — COPD / Hypercapnic respiratory failure"
            ),
            help="Use Scale 2 only if you have confirmed hypercapnic respiratory failure (COPD)."
        )

    on_oxygen = st.checkbox(
        "I am currently using supplemental oxygen (O₂ therapy)",
        help="Tick if you are wearing a nasal cannula, mask, or any oxygen delivery device."
    )

    # ── Cardiovascular ────────────────────────────────────────────────────────
    st.markdown("### 💓 Cardiovascular Readings")
    col4, col5 = st.columns(2)

    with col4:
        systolic_bp = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=50, max_value=300, value=120, step=1,
            help="The top (larger) number shown on your blood pressure monitor."
        )
    with col5:
        heart_rate = st.number_input(
            "Heart Rate (bpm)",
            min_value=20, max_value=250, value=75, step=1,
            help="Beats per minute — use a pulse oximeter or count pulse at wrist."
        )

    # ── Temperature & Neurological ────────────────────────────────────────────
    st.markdown("### 🌡️ Temperature & Neurological Status")
    col6, col7 = st.columns(2)

    with col6:
        temperature = st.number_input(
            "Body Temperature (°C)",
            min_value=30.0, max_value=44.0, value=37.0, step=0.1,
            format="%.1f",
            help="Record in degrees Celsius. Normal oral temperature is 36.0–37.5 °C."
        )
    with col7:
        consciousness = st.selectbox(
            "Level of Consciousness (ACVPU scale)",
            options=["A", "C", "V", "P", "U"],
            format_func=lambda x: {
                "A": "A — Alert: fully awake and oriented",
                "C": "C — Confused: awake but disoriented",
                "V": "V — Voice: responds only to voice",
                "P": "P — Pain: responds only to painful stimulus",
                "U": "U — Unresponsive: no response to any stimulus",
            }[x],
            help="Choose the level that best describes your current state."
        )

    st.divider()
    st.warning(
        "⚠️ **Confirm accuracy**: By submitting, you confirm that all readings above are accurate. "
        "This assessment will be **automatically and mandatorily** sent to your assigned doctor."
    )
    submitted = st.form_submit_button(
        "🚀 Submit Assessment", use_container_width=True, type="primary"
    )

# ── Processing ────────────────────────────────────────────────────────────────
if submitted:
    vitals = {
        "respiratory_rate":  respiratory_rate,
        "oxygen_saturation": oxygen_saturation,
        "o2_scale":          o2_scale,
        "systolic_bp":       systolic_bp,
        "heart_rate":        heart_rate,
        "temperature":       temperature,
        "consciousness":     consciousness,
        "on_oxygen":         1 if on_oxygen else 0,
    }

    # Hard validation — block impossible values
    hard_errors = validate_vitals_hard(vitals)
    if hard_errors:
        for err in hard_errors:
            st.error(f"❌ {err}")
        st.stop()

    # Soft warnings — show but allow submission
    soft_warns = validate_vitals_soft(vitals)
    show_clinical_warnings(soft_warns)

    # Run hybrid assessment
    with st.spinner("🤖 Running hybrid AI risk analysis — please wait..."):
        result = run_full_assessment(vitals)

    # Persist to DB (mandatory — no skip option)
    new_assessment = Assessment(
        patient_id        = patient.id,
        respiratory_rate  = respiratory_rate,
        oxygen_saturation = oxygen_saturation,
        o2_scale          = o2_scale,
        systolic_bp       = systolic_bp,
        heart_rate        = heart_rate,
        temperature       = temperature,
        consciousness     = consciousness,
        on_oxygen         = 1 if on_oxygen else 0,
        rule_score        = result["rule_score"],
        ml_prediction     = result["ml_prediction"],
        ml_probability    = result["ml_probability"],
        final_risk        = result["final_risk"],
        explanation       = result["explanation"],
        recommendation    = result["recommendation"],
        status            = "pending",
    )
    db.add(new_assessment)
    db.commit()

    log_action(
        user["id"], "SUBMIT_ASSESSMENT",
        f"Patient submitted assessment #{new_assessment.id} — "
        f"Rule: {result['rule_score']} | ML: {result['ml_prediction']} | "
        f"Final Risk: {result['final_risk']}"
    )

    # ── Results display ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("## ✅ Assessment Complete")

    show_risk_alert(result["final_risk"], result["recommendation"])

    # AI class probability bar (if available)
    if result.get("ml_class_probs"):
        import plotly.graph_objects as go
        probs = result["ml_class_probs"]   # [High, Low, Medium]
        labels  = ["High", "Low", "Medium"]
        colors  = ["#dc3545", "#28a745", "#ffc107"]
        fig = go.Figure(go.Bar(
            x=labels, y=[p * 100 for p in probs],
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],
            textposition="auto",
        ))
        fig.update_layout(
            title="AI Model Class Probabilities",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Rule score & ML prediction summary
    col_r, col_m, col_f = st.columns(3)
    col_r.metric("Rule-Based Score (NEWS)", result["rule_score"])
    col_m.metric("AI Model Prediction", result["ml_prediction"])
    col_f.metric("Final Risk (Hybrid)", result["final_risk"])

    with st.expander("📖 Detailed Clinical Explanation", expanded=True):
        st.markdown(result["explanation"])

    st.info(
        "📤 Your assessment has been automatically saved and forwarded to your doctor. "
        "You will see their feedback in your dashboard once they review it."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📊 Go to My Dashboard", use_container_width=True, type="primary"):
            db.close()
            st.switch_page("pages/4_Patient_Dashboard.py")
    with col_b:
        if st.button("📋 Submit Another Assessment", use_container_width=True):
            st.rerun()

db.close()
