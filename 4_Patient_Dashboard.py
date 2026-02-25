"""
pages/4_Patient_Dashboard.py - Patient dashboard
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database import init_db, get_session, Patient, Assessment, DoctorNote, User
from auth import seed_admin, require_auth
from utils import inject_global_css
from components.navbar import render_navbar
from components.charts import risk_trend_chart, risk_distribution_pie, vitals_radar
from components.alerts import show_risk_alert
from pdf_generator import generate_assessment_pdf

st.set_page_config(
    page_title="Patient Dashboard | Clinical DSS",
    page_icon="📊", layout="wide"
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
    st.error("Patient profile not found. Please contact the administrator.")
    db.close()
    st.stop()

assessments = (
    db.query(Assessment)
    .filter(Assessment.patient_id == patient.id)
    .order_by(Assessment.created_at.desc())
    .all()
)
latest = assessments[0] if assessments else None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"# 📊 My Health Dashboard")
st.caption(
    f"👤 {user['name']} · "
    f"Age: {patient.age or 'N/A'} · "
    f"Gender: {patient.gender or 'N/A'} · "
    f"Conditions: {patient.underlying_conditions or 'None recorded'}"
)

col_btn, _ = st.columns([1, 4])
with col_btn:
    if st.button("➕ New Assessment", type="primary", use_container_width=True):
        db.close()
        st.switch_page("pages/7_Assessment.py")

st.divider()

# ── Summary cards ─────────────────────────────────────────────────────────────
total   = len(assessments)
pending = sum(1 for a in assessments if a.status == "pending")
risk_counts = {"Low": 0, "Medium": 0, "High": 0}
for a in assessments:
    risk_counts[a.final_risk] = risk_counts.get(a.final_risk, 0) + 1

c1, c2, c3, c4 = st.columns(4)
for col, val, label, clr in zip(
    [c1, c2, c3, c4],
    [total, pending, risk_counts["Low"], risk_counts["High"]],
    ["Total Assessments", "Pending Review", "Low Risk", "High Risk"],
    ["#1a237e", "#ffc107", "#28a745", "#dc3545"],
):
    col.markdown(
        f'<div class="metric-card">'
        f'<div style="font-size:2rem;font-weight:800;color:{clr};">{val}</div>'
        f'<div style="color:#555;">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Latest assessment panel ───────────────────────────────────────────────────
if latest:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 📋 Latest Assessment")
        show_risk_alert(latest.final_risk, latest.recommendation)

        # Metric summary row
        m1, m2, m3 = st.columns(3)
        m1.metric("Rule Score (NEWS)", latest.rule_score)
        m2.metric("AI Prediction",     latest.ml_prediction)
        m3.metric("Final Risk",        latest.final_risk)

        with st.expander("📖 Clinical Explanation", expanded=False):
            st.markdown(latest.explanation)

        st.caption(
            f"Submitted: {latest.created_at.strftime('%B %d, %Y at %H:%M') if latest.created_at else 'N/A'}"
            f" · Status: **{latest.status.title()}**"
        )

        # Doctor feedback
        notes = (
            db.query(DoctorNote, User)
            .join(User, DoctorNote.doctor_id == User.id)
            .filter(DoctorNote.assessment_id == latest.id)
            .all()
        )
        if notes:
            st.markdown("#### 🩺 Doctor Feedback")
            for note, doc in notes:
                date_str = note.created_at.strftime("%b %d, %Y") if note.created_at else ""
                st.info(f"**Dr. {doc.name}** ({date_str}):\n\n{note.note}")
        else:
            st.caption("⏳ Awaiting doctor review…")

        # PDF download
        patient_info = {
            "name":       user["name"],
            "email":      user["email"],
            "age":        patient.age,
            "gender":     patient.gender,
            "conditions": patient.underlying_conditions,
        }
        assessment_dict = {
            "id":                latest.id,
            "final_risk":        latest.final_risk,
            "respiratory_rate":  latest.respiratory_rate,
            "oxygen_saturation": latest.oxygen_saturation,
            "o2_scale":          latest.o2_scale,
            "systolic_bp":       latest.systolic_bp,
            "heart_rate":        latest.heart_rate,
            "temperature":       latest.temperature,
            "consciousness":     latest.consciousness,
            "on_oxygen":         latest.on_oxygen,
            "explanation":       latest.explanation,
            "recommendation":    latest.recommendation,
        }
        notes_dicts = [
            {
                "doctor_name": doc.name,
                "created_at":  n.created_at.strftime("%b %d, %Y") if n.created_at else "",
                "note":        n.note,
            }
            for n, doc in notes
        ]
        pdf_bytes = generate_assessment_pdf(patient_info, assessment_dict, notes_dicts)
        st.download_button(
            "📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"assessment_{latest.id}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with col_right:
        st.markdown("### 🕸️ Vitals Radar")
        vitals_dict = {
            "respiratory_rate": latest.respiratory_rate,
            "oxygen_saturation": latest.oxygen_saturation,
            "systolic_bp": latest.systolic_bp,
            "heart_rate": latest.heart_rate,
            "temperature": latest.temperature,
        }
        st.plotly_chart(vitals_radar(vitals_dict), use_container_width=True)

else:
    st.info(
        "📭 No assessments yet. Click **New Assessment** above to submit your first vital signs."
    )

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
if assessments:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Risk Trend Over Time")
        st.plotly_chart(risk_trend_chart(assessments), use_container_width=True)
    with col2:
        st.markdown("### 🥧 Risk Distribution")
        st.plotly_chart(risk_distribution_pie(risk_counts), use_container_width=True)

    # ── Assessment history ────────────────────────────────────────────────────
    st.markdown("### 📋 Full Assessment History")
    for a in assessments:
        date_str = a.created_at.strftime("%b %d, %Y %H:%M") if a.created_at else "N/A"
        risk_icon = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}.get(a.final_risk, "❓")
        with st.expander(
            f"{risk_icon} Assessment #{a.id} — {date_str} "
            f"| Risk: {a.final_risk} | Status: {a.status.title()}"
        ):
            v1, v2, v3 = st.columns(3)
            v1.metric("Respiratory Rate", f"{a.respiratory_rate} bpm")
            v1.metric("SpO₂",            f"{a.oxygen_saturation}%")
            v2.metric("Systolic BP",     f"{a.systolic_bp} mmHg")
            v2.metric("Heart Rate",      f"{a.heart_rate} bpm")
            v3.metric("Temperature",     f"{a.temperature}°C")
            v3.metric("Consciousness",   a.consciousness)
            st.caption(f"Rule Score: {a.rule_score} · AI: {a.ml_prediction} (conf: {a.ml_probability:.1%})")
            st.markdown(f"**Recommendation:** {a.recommendation}")

db.close()
