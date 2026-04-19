from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "analysis_summary.json"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_dataset.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

DISPLAY_LABELS = {
    "Drug MW": "Drug MW",
    "Drug TPSA": "Drug TPSA",
    "Drug LogP": "Drug LogP",
    "Polymer MW": "Polymer MW",
    "LA/GA": "LA/GA",
    "Initial Drug-to-Polymer Ratio": "Initial Drug-to-Polymer Ratio",
    "Particle Size": "Particle Size",
    "Drug Loading Capacity": "Drug Loading Capacity",
    "Drug Encapuslation Efficiency": "Drug Encapsulation Efficiency",
    "Solubility Enhancer Concentration": "Solubility Enhancer Concentration",
    "Time": "Time",
}


st.set_page_config(
    page_title="PharmaReleaseAI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@st.cache_data
def load_analysis():
    if not SUMMARY_PATH.exists():
        return None
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


def resolve_artifact_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    fallback = FIGURES_DIR / candidate.name
    if fallback.exists():
        return fallback
    return None


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        html {
            scroll-behavior: smooth;
        }
        body {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at 76% 68%, rgba(34, 211, 238, 0.22), transparent 0%, transparent 24%),
                radial-gradient(circle at 64% 84%, rgba(14, 165, 233, 0.18), transparent 0%, transparent 22%),
                radial-gradient(circle at 88% 14%, rgba(56, 189, 248, 0.16), transparent 0%, transparent 18%),
                radial-gradient(circle at 18% 18%, rgba(15, 23, 42, 0.35), transparent 0%, transparent 20%),
                linear-gradient(180deg, #02060d 0%, #06101c 26%, #071522 58%, #071828 100%);
            color: #f8fafc;
            overflow: hidden;
        }
        .stApp::before,
        .stApp::after {
            content: "";
            position: fixed;
            pointer-events: none;
            inset: 0;
            z-index: 0;
        }
        .stApp::before {
            background-image:
                radial-gradient(circle at 12% 22%, rgba(255,255,255,0.22) 0 1px, transparent 1.2px),
                radial-gradient(circle at 28% 80%, rgba(125,211,252,0.25) 0 1px, transparent 1.3px),
                radial-gradient(circle at 72% 26%, rgba(255,255,255,0.18) 0 1px, transparent 1.2px),
                radial-gradient(circle at 86% 58%, rgba(103,232,249,0.25) 0 1px, transparent 1.2px),
                radial-gradient(circle at 58% 12%, rgba(255,255,255,0.14) 0 1px, transparent 1.2px);
            background-size: 220px 220px, 260px 260px, 240px 240px, 300px 300px, 180px 180px;
            opacity: 0.35;
        }
        .stApp::after {
            background:
                radial-gradient(circle at 72% 78%, rgba(34, 211, 238, 0.15), transparent 0%, transparent 22%),
                radial-gradient(circle at 82% 72%, rgba(59, 130, 246, 0.12), transparent 0%, transparent 26%);
            filter: blur(24px);
        }
        .block-container {
            max-width: 1240px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            position: relative;
            z-index: 1;
        }
        h1, h2, h3, h4, h5, p, span, label, div {
            color: #f8fafc;
        }
        .hero-shell, .glass-card, .metric-card, .result-card, .mini-card, .about-card {
            border-radius: 28px;
            border: 1px solid rgba(103, 232, 249, 0.12);
            background: linear-gradient(180deg, rgba(7, 14, 27, 0.64), rgba(7, 18, 33, 0.42));
            box-shadow:
                0 18px 48px rgba(2, 6, 23, 0.44),
                inset 0 1px 0 rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(16px);
        }
        .hero-shell {
            padding: 2.7rem 2.4rem;
            min-height: 430px;
            background:
                radial-gradient(circle at 100% 16%, rgba(34, 211, 238, 0.16), transparent 0%, transparent 28%),
                linear-gradient(135deg, rgba(5, 11, 22, 0.86), rgba(8, 23, 41, 0.62));
            position: relative;
            overflow: hidden;
        }
        .hero-shell::after {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(115deg, transparent 0%, transparent 46%, rgba(255,255,255,0.03) 50%, transparent 54%, transparent 100%);
            pointer-events: none;
        }
        .eyebrow {
            display: inline-block;
            border-radius: 999px;
            padding: 0.38rem 0.8rem;
            border: 1px solid rgba(34, 211, 238, 0.24);
            color: #7dd3fc;
            background: rgba(34, 211, 238, 0.07);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 1rem;
            box-shadow: 0 0 24px rgba(34, 211, 238, 0.08);
        }
        .hero-title {
            font-size: 3.55rem;
            font-weight: 800;
            line-height: 0.98;
            letter-spacing: -0.04em;
            margin-bottom: 1rem;
            max-width: 620px;
        }
        .hero-subtitle {
            color: #dbe7f5;
            font-size: 1.05rem;
            line-height: 1.75;
            max-width: 440px;
            margin-bottom: 1.6rem;
        }
        .cta-row {
            display: flex;
            gap: 0.85rem;
            align-items: center;
            flex-wrap: wrap;
        }
        .cta-link, .cta-link-secondary {
            display: inline-block;
            text-decoration: none;
            border-radius: 18px;
            padding: 0.92rem 1.3rem;
            font-weight: 800;
            transition: all 0.2s ease;
        }
        .cta-link {
            color: #ecfeff !important;
            background: linear-gradient(180deg, rgba(11, 20, 34, 0.84), rgba(7, 18, 31, 0.76));
            border: 1px solid rgba(103, 232, 249, 0.34);
            box-shadow:
                0 0 0 1px rgba(103, 232, 249, 0.08) inset,
                0 0 28px rgba(34, 211, 238, 0.12);
        }
        .cta-link:hover {
            box-shadow:
                0 0 0 1px rgba(103, 232, 249, 0.18) inset,
                0 0 36px rgba(34, 211, 238, 0.18);
            transform: translateY(-1px);
        }
        .cta-link-secondary {
            color: #f8fafc !important;
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.92), rgba(249, 115, 22, 0.94));
            box-shadow: 0 14px 30px rgba(239, 68, 68, 0.24);
            border: 1px solid rgba(255, 255, 255, 0.06);
        }
        .hero-art {
            position: relative;
            min-height: 360px;
            border-radius: 34px;
            overflow: hidden;
            background:
                radial-gradient(circle at 75% 78%, rgba(34, 211, 238, 0.20), transparent 0%, transparent 23%),
                radial-gradient(circle at 72% 22%, rgba(34, 211, 238, 0.13), transparent 0%, transparent 22%),
                linear-gradient(180deg, rgba(6, 12, 23, 0.86), rgba(8, 18, 33, 0.64));
            border: 1px solid rgba(125, 211, 252, 0.12);
            box-shadow: inset 0 0 40px rgba(34, 211, 238, 0.05);
        }
        .hero-art-label {
            position: absolute;
            left: 18px;
            bottom: 18px;
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            background: rgba(8, 18, 33, 0.66);
            border: 1px solid rgba(103, 232, 249, 0.14);
            color: #cbeff8;
            font-size: 0.72rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            backdrop-filter: blur(10px);
        }
        .orb, .orb-small, .molecule-core, .molecule-node, .capsule-top, .capsule-bottom {
            position: absolute;
        }
        .orb {
            width: 310px;
            height: 310px;
            right: 12px;
            top: 36px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(34, 211, 238, 0.28), rgba(34, 211, 238, 0.02));
            filter: blur(10px);
        }
        .orb-small {
            width: 180px;
            height: 180px;
            left: 28px;
            bottom: 24px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.16), rgba(59, 130, 246, 0.01));
            filter: blur(12px);
        }
        .molecule-core {
            width: 92px;
            height: 92px;
            top: 86px;
            left: 70px;
            border-radius: 999px;
            border: 2px solid rgba(165, 243, 252, 0.7);
            box-shadow: 0 0 30px rgba(34, 211, 238, 0.26);
        }
        .molecule-node {
            width: 18px;
            height: 18px;
            border-radius: 999px;
            background: #67e8f9;
            box-shadow: 0 0 18px rgba(103, 232, 249, 0.7);
        }
        .node-1 { top: 78px; left: 210px; }
        .node-2 { top: 188px; left: 154px; }
        .node-3 { top: 150px; left: 38px; }
        .node-line {
            position: absolute;
            height: 2px;
            background: linear-gradient(90deg, rgba(103, 232, 249, 0.9), rgba(103, 232, 249, 0.12));
            transform-origin: left center;
        }
        .line-1 { width: 92px; top: 120px; left: 144px; transform: rotate(-14deg); }
        .line-2 { width: 74px; top: 164px; left: 113px; transform: rotate(49deg); }
        .line-3 { width: 84px; top: 144px; left: 55px; transform: rotate(-38deg); }
        .capsule-top, .capsule-bottom {
            width: 76px;
            height: 164px;
            border-radius: 999px;
            transform: rotate(26deg);
            top: 128px;
            right: 96px;
        }
        .capsule-top {
            background: linear-gradient(180deg, rgba(239, 68, 68, 0.92), rgba(251, 146, 60, 0.96));
            box-shadow: 0 0 42px rgba(248, 113, 113, 0.30);
        }
        .capsule-bottom {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(207, 250, 254, 0.92));
            right: 144px;
            box-shadow: 0 0 24px rgba(125, 211, 252, 0.14);
        }
        .particle {
            position: absolute;
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: rgba(236, 254, 255, 0.95);
            box-shadow: 0 0 20px rgba(103, 232, 249, 0.65);
        }
        .particle-1 { top: 150px; right: 84px; }
        .particle-2 { top: 172px; right: 72px; width: 6px; height: 6px; }
        .particle-3 { top: 196px; right: 92px; width: 7px; height: 7px; }
        .particle-4 { top: 184px; right: 54px; width: 5px; height: 5px; }
        .particle-5 { top: 214px; right: 76px; width: 4px; height: 4px; }
        .section-anchor {
            position: relative;
            top: -14px;
        }
        .section-title {
            font-size: 1.58rem;
            font-weight: 780;
            margin: 0.3rem 0 0.95rem 0;
            letter-spacing: -0.02em;
        }
        .section-copy {
            color: #cbd5e1;
            font-size: 0.95rem;
            margin-bottom: 0;
        }
        .metric-card {
            padding: 1.25rem 1.3rem;
            min-height: 142px;
        }
        .metric-label {
            color: #94a3b8;
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.4rem;
        }
        .metric-value {
            font-size: 1.95rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        .metric-note {
            color: #cbd5e1;
            font-size: 0.93rem;
        }
        .glass-card, .about-card {
            padding: 1.35rem 1.35rem;
            transition: box-shadow 0.2s ease, border-color 0.2s ease;
        }
        .glass-card:hover, .metric-card:hover, .result-card:hover, .about-card:hover {
            border-color: rgba(103, 232, 249, 0.18);
            box-shadow:
                0 22px 58px rgba(2, 6, 23, 0.44),
                0 0 36px rgba(34, 211, 238, 0.06);
        }
        .result-card {
            padding: 1.6rem 1.6rem 1.4rem 1.6rem;
            background:
                radial-gradient(circle at top right, rgba(34, 211, 238, 0.12), transparent 24%),
                linear-gradient(180deg, rgba(8, 21, 36, 0.78), rgba(7, 16, 29, 0.52));
        }
        .result-card-ready {
            animation: resultGlow 0.85s ease-out;
            border-color: rgba(103, 232, 249, 0.24);
            box-shadow:
                0 24px 60px rgba(2, 6, 23, 0.44),
                0 0 40px rgba(34, 211, 238, 0.12);
        }
        .result-tag {
            color: #99f6e4;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
        }
        .result-value {
            font-size: 3.65rem;
            font-weight: 800;
            line-height: 1;
            margin: 0.15rem 0 0.55rem 0;
        }
        .pill {
            display: inline-block;
            padding: 0.42rem 0.82rem;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 700;
            margin-right: 0.42rem;
            margin-bottom: 0.45rem;
        }
        .pill-cyan {
            background: rgba(34, 211, 238, 0.12);
            color: #a5f3fc;
            border: 1px solid rgba(34, 211, 238, 0.24);
        }
        .pill-green {
            background: rgba(16, 185, 129, 0.12);
            color: #a7f3d0;
            border: 1px solid rgba(16, 185, 129, 0.24);
        }
        .pill-orange {
            background: rgba(249, 115, 22, 0.12);
            color: #fdba74;
            border: 1px solid rgba(249, 115, 22, 0.24);
        }
        .pill-red {
            background: rgba(239, 68, 68, 0.12);
            color: #fca5a5;
            border: 1px solid rgba(239, 68, 68, 0.22);
        }
        div[data-testid="stNumberInput"] > div,
        div[data-testid="stSlider"] > div {
            background: linear-gradient(180deg, rgba(6, 14, 25, 0.92), rgba(8, 18, 31, 0.90));
            border-radius: 18px;
            border: 1px solid rgba(103, 232, 249, 0.10);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
        }
        div[data-testid="stSliderTickBarMin"],
        div[data-testid="stSliderTickBarMax"] {
            background: rgba(34, 211, 238, 0.24);
        }
        div[data-baseweb="slider"] div[role="slider"] {
            background: #67e8f9 !important;
            box-shadow: 0 0 18px rgba(103, 232, 249, 0.45);
            border-color: rgba(255, 255, 255, 0.88) !important;
        }
        .stButton > button {
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            font-weight: 800;
            padding: 0.88rem 1.2rem;
            color: white;
            background: linear-gradient(135deg, #ef4444, #f97316);
            box-shadow: 0 16px 30px rgba(239, 68, 68, 0.20);
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 20px 38px rgba(239, 68, 68, 0.24);
        }
        [data-testid="stDataFrame"] {
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid rgba(103, 232, 249, 0.12);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
        }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(6, 14, 25, 0.68), rgba(8, 18, 31, 0.52));
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            border: 1px solid rgba(103, 232, 249, 0.10);
        }
        .about-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
        }
        .about-stat {
            border-radius: 18px;
            padding: 1rem 1.05rem;
            background: linear-gradient(180deg, rgba(6, 14, 25, 0.72), rgba(8, 18, 31, 0.56));
            border: 1px solid rgba(103, 232, 249, 0.10);
        }
        .about-stat-value {
            font-size: 1.25rem;
            font-weight: 780;
            margin-top: 0.2rem;
        }
        .evidence-note {
            color: #9fb7c8;
            font-size: 0.84rem;
            margin-top: 0.7rem;
        }
        .predict-helper {
            color: #8ea9bc;
            font-size: 0.84rem;
            margin-top: 0.45rem;
            text-align: center;
        }
        @keyframes resultGlow {
            0% {
                transform: translateY(14px);
                opacity: 0.55;
                box-shadow:
                    0 12px 28px rgba(2, 6, 23, 0.28),
                    0 0 0 rgba(34, 211, 238, 0.0);
            }
            100% {
                transform: translateY(0);
                opacity: 1;
                box-shadow:
                    0 24px 60px rgba(2, 6, 23, 0.44),
                    0 0 40px rgba(34, 211, 238, 0.12);
            }
        }
        @media (max-width: 900px) {
            .hero-title {
                font-size: 2.7rem;
            }
            .hero-art {
                min-height: 280px;
            }
            .about-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card_metric(label: str, value: str, note: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def get_numeric_range(df: pd.DataFrame, column: str) -> tuple[float, float, float]:
    series = df[column].astype(float)
    return float(series.min()), float(series.max()), float(series.median())


def confidence_label(prediction_interval: float | None) -> str:
    if prediction_interval is None:
        return "Estimated"
    if prediction_interval <= 0.12:
        return "High"
    if prediction_interval <= 0.20:
        return "Moderate"
    return "Exploratory"


def release_status(release_percent: float) -> tuple[str, str]:
    if release_percent >= 75:
        return "Fast Release Profile", "pill-red"
    if release_percent >= 40:
        return "Moderate Release Profile", "pill-cyan"
    return "Sustained Release Profile", "pill-green"


def feature_importance_chart(importance_df: pd.DataFrame):
    chart_df = importance_df.head(8).copy()
    chart_df["feature"] = chart_df["feature"].replace(DISPLAY_LABELS)
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8)
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.value("#22d3ee"),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
        )
        .properties(height=320)
    )


def build_time_curve(
    model,
    feature_columns: list[str],
    current_inputs: dict[str, float],
    time_column: str,
    time_min: float,
    time_max: float,
) -> pd.DataFrame:
    sweep_times = np.linspace(time_min, time_max, 28)
    rows = []
    for time_value in sweep_times:
        row = {feature: current_inputs[feature] for feature in feature_columns}
        row[time_column] = float(time_value)
        rows.append(row)
    curve_df = pd.DataFrame(rows)
    curve_df["Predicted Release (%)"] = np.clip(model.predict(curve_df) * 100, 0, 100)
    curve_df[time_column] = curve_df[time_column].round(2)
    return curve_df


def time_curve_chart(curve_df: pd.DataFrame, time_column: str):
    return (
        alt.Chart(curve_df)
        .mark_line(color="#38bdf8", point=alt.OverlayMarkDef(color="#f97316", size=44))
        .encode(
            x=alt.X(f"{time_column}:Q", title="Time"),
            y=alt.Y("Predicted Release (%):Q", title="Predicted Release (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=[
                alt.Tooltip(f"{time_column}:Q", format=".2f"),
                alt.Tooltip("Predicted Release (%):Q", format=".2f"),
            ],
        )
        .properties(height=320)
    )


def render_hero(metrics: dict, rows: int) -> None:
    left, right = st.columns([1.2, 0.95], vertical_alignment="center")
    with left:
        st.markdown(
            """
            <div class="hero-shell">
                <div class="eyebrow">PharmaReleaseAI</div>
                <div class="hero-title">AI-Based Predictive<br/>Modelling of Drug Release</div>
                <div class="hero-subtitle">
                    Smarter formulation R&amp;D.<br/>
                    Faster predictions.<br/>
                    Better outcomes.
                </div>
                <div class="cta-row">
                    <a class="cta-link" href="#prediction-form">Start Prediction</a>
                    <a class="cta-link-secondary" href="#visual-insights">View Insights</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="hero-art">
                <div class="orb"></div>
                <div class="orb-small"></div>
                <div class="molecule-core"></div>
                <div class="molecule-node node-1"></div>
                <div class="molecule-node node-2"></div>
                <div class="molecule-node node-3"></div>
                <div class="node-line line-1"></div>
                <div class="node-line line-2"></div>
                <div class="node-line line-3"></div>
                <div class="capsule-top"></div>
                <div class="capsule-bottom"></div>
                <div class="particle particle-1"></div>
                <div class="particle particle-2"></div>
                <div class="particle particle-3"></div>
                <div class="particle particle-4"></div>
                <div class="particle particle-5"></div>
                <div class="hero-art-label">Decorative branding visual only</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(card_metric("Accuracy", f"R² = {metrics['R2']:.4f}", "Tuned holdout performance"), unsafe_allow_html=True)
    with c2:
        st.markdown(card_metric("Best Model", "Random Forest", "Selected from real model benchmarking"), unsafe_allow_html=True)
    with c3:
        st.markdown(card_metric("Dataset Rows", f"{rows}", "Processed formulation-time observations"), unsafe_allow_html=True)


def render_form(feature_columns: list[str], clean_df: pd.DataFrame) -> dict[str, float]:
    st.markdown("<div id='prediction-form' class='section-anchor'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction Form</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Time is retained because predicted Release (%) changes with the selected release time.</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    form_inputs: dict[str, float] = {}
    split = (len(feature_columns) + 1) // 2
    groups = [feature_columns[:split], feature_columns[split:]]

    for container, group in zip((left, right), groups):
        with container:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            for feature in group:
                minimum, maximum, default = get_numeric_range(clean_df, feature)
                step = max((maximum - minimum) / 150, 0.01)
                label = DISPLAY_LABELS.get(feature, feature)
                if feature == "Time":
                    form_inputs[feature] = st.slider(
                        label,
                        min_value=float(round(minimum, 2)),
                        max_value=float(round(maximum, 2)),
                        value=float(round(default, 2)),
                        step=float(round(max(step, 0.1), 2)),
                    )
                elif feature in {"Drug LogP", "LA/GA", "Initial Drug-to-Polymer Ratio", "Solubility Enhancer Concentration"}:
                    form_inputs[feature] = st.slider(
                        label,
                        min_value=float(round(minimum, 4)),
                        max_value=float(round(maximum, 4)),
                        value=float(round(default, 4)),
                        step=float(round(step, 4)),
                    )
                else:
                    form_inputs[feature] = st.number_input(
                        label,
                        min_value=float(round(minimum, 4)),
                        max_value=float(round(maximum, 4)),
                        value=float(round(default, 4)),
                        step=float(round(step, 4)),
                    )
            st.markdown("</div>", unsafe_allow_html=True)

    return form_inputs


def render_result(
    prediction_percent: float,
    time_value: float,
    confidence: str,
    status: str,
    status_class: str,
    highlight: bool = False,
) -> None:
    st.markdown("<div id='results-section' class='section-anchor'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Predicted Release</div>', unsafe_allow_html=True)
    result_col, bar_col = st.columns([1.3, 0.7], vertical_alignment="center")
    result_classes = "result-card result-card-ready" if highlight else "result-card"
    with result_col:
        st.markdown(
            f"""
            <div class="{result_classes}">
                <div class="result-tag">Model Output</div>
                <div class="result-value">{prediction_percent:.1f}%</div>
                <div>
                    <span class="pill pill-cyan">Confidence: {confidence}</span>
                    <span class="pill {status_class}">Predicted Release Behaviour: {status}</span>
                </div>
                <p class="section-copy" style="margin-top:0.8rem;">
                    At selected time = {time_value:.2f} hr, the model predicts a drug release of {prediction_percent:.1f}%.
                </p>
                <div class="evidence-note">Powered by trained Random Forest model on 4,909 formulation records.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with bar_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Release Progress**")
        st.progress(int(round(prediction_percent)))
        st.metric("Predicted Release", f"{prediction_percent:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)


def render_insights(
    analysis: dict,
    clean_df: pd.DataFrame,
    model,
    feature_columns: list[str],
    current_inputs: dict[str, float],
) -> None:
    st.markdown("<div id='visual-insights' class='section-anchor'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Visual Insights</div>', unsafe_allow_html=True)
    importance_df = pd.DataFrame(analysis["explainability_outputs"]["importance_table"])
    time_column = analysis["dataset_summary"]["time_column"] or "Time"
    time_min, time_max, _ = get_numeric_range(clean_df, time_column)
    curve_df = build_time_curve(model, feature_columns, current_inputs, time_column, time_min, time_max)

    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Feature Importance**")
        st.altair_chart(feature_importance_chart(importance_df), use_container_width=True)
        st.markdown('<div class="evidence-note">Powered by trained Random Forest model on 4,909 formulation records.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with top_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Time vs Predicted Release Curve**")
        st.altair_chart(time_curve_chart(curve_df, time_column), use_container_width=True)
        st.markdown('<div class="evidence-note">Powered by trained Random Forest model on 4,909 formulation records.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns([1.05, 1.15])
    with bottom_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Reference Dataset Sample**")
        sample_columns = feature_columns[:6] + [analysis["dataset_summary"]["target_column"]]
        st.dataframe(clean_df[sample_columns].head(5), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with bottom_right:
        heatmap_path = resolve_artifact_path(analysis["eda_outputs"].get("heatmap_path"))
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Correlation Heatmap**")
        if heatmap_path and heatmap_path.exists():
            st.image(str(heatmap_path), use_container_width=True)
        else:
            st.info("Correlation heatmap unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_about(metrics: dict) -> None:
    st.markdown('<div class="section-title">About Model</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="about-card">
            <div class="about-grid">
                <div class="about-stat">
                    <div class="metric-label">Best Model</div>
                    <div class="about-stat-value">Random Forest</div>
                </div>
                <div class="about-stat">
                    <div class="metric-label">R²</div>
                    <div class="about-stat-value">{metrics['R2']:.4f}</div>
                </div>
                <div class="about-stat">
                    <div class="metric-label">RMSE</div>
                    <div class="about-stat-value">{metrics['RMSE']:.4f}</div>
                </div>
                <div class="about-stat">
                    <div class="metric-label">MAE</div>
                    <div class="about-stat-value">{metrics['MAE']:.4f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Data Sources Used"):
        st.markdown("- Processed formulation dataset")
        st.markdown("- Trained ML model")
        st.markdown("- Real-time prediction engine")


def main() -> None:
    inject_styles()

    model = load_model()
    analysis = load_analysis()
    clean_df = load_dataset()

    if model is None or analysis is None or clean_df is None:
        st.error("Required project artifacts are missing. Run `python3 run_pipeline.py` before launching the app.")
        st.stop()

    st.session_state.setdefault("latest_prediction", None)
    st.session_state.setdefault("pending_prediction", False)
    st.session_state.setdefault("prediction_ready", False)
    st.session_state.setdefault("scroll_to_results", False)

    metrics = analysis["modelling_outputs"]["tuned_metrics"]
    feature_columns = analysis["modelling_outputs"]["feature_columns"]
    prediction_interval = analysis["modelling_outputs"].get("prediction_interval")
    time_column = analysis["dataset_summary"]["time_column"] or "Time"

    render_hero(metrics, analysis["dataset_summary"]["rows"])
    st.markdown("")

    current_inputs = render_form(feature_columns, clean_df)

    if st.session_state.latest_prediction is None:
        baseline_input = pd.DataFrame([current_inputs])
        baseline_prediction = float(np.clip(model.predict(baseline_input)[0] * 100, 0, 100))
        status, status_class = release_status(baseline_prediction)
        st.session_state.latest_prediction = {
            "prediction_percent": baseline_prediction,
            "time": current_inputs[time_column],
            "status": status,
            "status_class": status_class,
            "confidence": confidence_label(prediction_interval),
            "inputs": current_inputs,
        }

    button_label = "Predicting..." if st.session_state.pending_prediction else (
        "Prediction Ready ↓" if st.session_state.prediction_ready else "Predict Drug Release"
    )
    if st.button(
        button_label,
        type="primary",
        use_container_width=True,
        disabled=st.session_state.pending_prediction,
    ):
        st.session_state.pending_prediction = True
        st.session_state.prediction_ready = False
        st.session_state.scroll_to_results = False
        st.rerun()

    st.markdown('<div class="predict-helper">Results will appear below automatically.</div>', unsafe_allow_html=True)

    if st.session_state.pending_prediction:
        with st.spinner("Generating prediction..."):
            prediction_input = pd.DataFrame([current_inputs])
            prediction_percent = float(np.clip(model.predict(prediction_input)[0] * 100, 0, 100))
            status, status_class = release_status(prediction_percent)
            st.session_state.latest_prediction = {
                "prediction_percent": prediction_percent,
                "time": current_inputs[time_column],
                "status": status,
                "status_class": status_class,
                "confidence": confidence_label(prediction_interval),
                "inputs": current_inputs,
            }
        st.session_state.pending_prediction = False
        st.session_state.prediction_ready = True
        st.session_state.scroll_to_results = True
        st.rerun()

    latest = st.session_state.latest_prediction
    st.markdown("")
    render_result(
        latest["prediction_percent"],
        latest["time"],
        latest["confidence"],
        latest["status"],
        latest["status_class"],
        highlight=st.session_state.prediction_ready,
    )
    if st.session_state.scroll_to_results:
        components.html(
            """
            <script>
            const target = window.parent.document.getElementById("results-section");
            if (target) {
              target.scrollIntoView({behavior: "smooth", block: "start"});
              window.parent.location.hash = "results-section";
            }
            </script>
            """,
            height=0,
        )
        st.session_state.scroll_to_results = False

    if st.session_state.prediction_ready:
        st.session_state.prediction_ready = False

    st.markdown("")
    render_insights(analysis, clean_df, model, feature_columns, latest["inputs"])
    st.markdown("")
    render_about(metrics)


if __name__ == "__main__":
    main()
