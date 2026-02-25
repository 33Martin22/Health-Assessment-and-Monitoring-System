"""
components/charts.py - Reusable Plotly chart components (3-class: Low / Medium / High)
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np


RISK_COLORS = {
    "Low":    "#28a745",
    "Medium": "#ffc107",
    "High":   "#dc3545",
}
RISK_NUM = {"Low": 1, "Medium": 2, "High": 3}


def risk_trend_chart(assessments: list) -> go.Figure:
    """Line + scatter chart of risk level over time."""
    if not assessments:
        return _empty_chart("No assessment data available yet")

    df = pd.DataFrame([{
        "Date":     a.created_at,
        "Risk":     a.final_risk,
        "Risk_num": RISK_NUM.get(a.final_risk, 0),
    } for a in assessments if a.created_at])

    if df.empty:
        return _empty_chart("No timestamped assessments found")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Risk_num"],
        mode="lines+markers",
        marker=dict(
            color=[RISK_COLORS.get(r, "#999") for r in df["Risk"]],
            size=12,
            line=dict(color="white", width=2),
        ),
        line=dict(color="#4A90D9", width=2, dash="dot"),
        text=df["Risk"],
        hovertemplate="<b>%{text}</b><br>%{x|%b %d, %Y}<extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(
            tickvals=[1, 2, 3],
            ticktext=["Low", "Medium", "High"],
            range=[0, 4],
            gridcolor="#f0f0f0",
        ),
        xaxis=dict(gridcolor="#f0f0f0"),
        xaxis_title="Date",
        yaxis_title="Risk Level",
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def risk_distribution_pie(counts: dict) -> go.Figure:
    """Donut pie chart of risk distribution."""
    filtered = {k: v for k, v in counts.items() if v > 0}
    if not filtered:
        return _empty_chart("No assessment data for distribution")

    labels = list(filtered.keys())
    values = list(filtered.values())

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker_colors=[RISK_COLORS.get(lbl, "#999") for lbl in labels],
        hole=0.42,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b>: %{value} assessments<extra></extra>",
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    return fig


def vitals_radar(vitals: dict) -> go.Figure:
    """Radar chart of normalised vitals."""
    norms = {
        "Resp Rate":  min(vitals.get("respiratory_rate",  18) / 30, 1.0),
        "SpO₂":       vitals.get("oxygen_saturation",     98) / 100,
        "Sys BP":     min(vitals.get("systolic_bp",       120) / 200, 1.0),
        "Heart Rate": min(vitals.get("heart_rate",        80) / 150, 1.0),
        "Temp":       min((vitals.get("temperature",      37) - 35) / 7, 1.0),
    }
    cats   = list(norms.keys()) + [list(norms.keys())[0]]   # close polygon
    vals   = list(norms.values()) + [list(norms.values())[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats,
        fill="toself",
        fillcolor="rgba(74, 144, 217, 0.20)",
        line=dict(color="#4A90D9", width=2),
        marker=dict(size=6, color="#1a237e"),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
        ),
        height=300,
        margin=dict(l=30, r=30, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def ml_probability_bar(probs: list) -> go.Figure:
    """
    Horizontal bar chart for the 3 Keras output probabilities.
    probs: [p_High, p_Low, p_Medium]  (alphabetical model output order)
    """
    labels  = ["High", "Low", "Medium"]
    colors  = [RISK_COLORS[l] for l in labels]
    pct     = [p * 100 for p in probs]

    fig = go.Figure(go.Bar(
        x=labels, y=pct,
        marker_color=colors,
        text=[f"{p:.1f}%" for p in pct],
        textposition="auto",
    ))
    fig.update_layout(
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=260,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _empty_chart(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color="#888"),
    )
    fig.update_layout(
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
