"""
components/alerts.py - Reusable alert / notification components (3-class model)
"""
import streamlit as st


_RISK_CONFIG = {
    "Low":    ("success", "✅ LOW RISK"),
    "Medium": ("warning", "⚠️ MEDIUM RISK"),
    "High":   ("error",   "🚨 HIGH RISK — URGENT ATTENTION REQUIRED"),
}


def show_risk_alert(risk_level: str, message: str = ""):
    """Display a colour-coded risk alert banner."""
    kind, label = _RISK_CONFIG.get(risk_level, ("info", f"RISK: {risk_level}"))
    full_msg = f"**{label}**" + (f" — {message}" if message else "")
    getattr(st, kind)(full_msg)


def show_clinical_warnings(warnings: list):
    """Show soft clinical warnings in a collapsible block."""
    if warnings:
        with st.expander("⚠️ Clinical Warnings — values outside normal range", expanded=True):
            for w in warnings:
                st.warning(w)


def show_pending_badge(count: int):
    """Inline badge for pending review count."""
    if count > 0:
        st.markdown(
            f'<span style="background:#dc3545;color:white;border-radius:12px;'
            f'padding:3px 12px;font-size:0.8rem;font-weight:600;">'
            f'🔔 {count} Pending Review{"s" if count > 1 else ""}</span>',
            unsafe_allow_html=True,
        )
