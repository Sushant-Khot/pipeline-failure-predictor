from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_pipeline_failure
from src.model_insights import (
    build_recommendations,
    compute_health_score,
    compute_kpis,
    get_dataset_predictions,
    get_risk_category,
    get_risk_drivers
)

st.set_page_config(
    page_title="Pipeline Failure Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_STYLES = {
    "LOW": {"color": "#1a7f37", "bg": "#e9f7ee"},
    "MEDIUM": {"color": "#9a6700", "bg": "#fdf3d9"},
    "HIGH": {"color": "#b3391a", "bg": "#fbe9e4"},
    "CRITICAL": {"color": "#ffffff", "bg": "#8f1c1c"},
}

PIPELINE_NAMES = ["customer_daily", "sales_daily", "products_daily", "payments_daily", "orders_daily"]
DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

NAV_SECTIONS = {
    "OVERVIEW": ["Pipeline Monitor", "Risk Analysis"],
    "ANALYTICS": ["Pipeline Health", "Failure Analytics"],
    "TOOLS": ["What-if Simulator"],
}


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] { font-family: "Segoe UI", "Inter", system-ui, sans-serif; }
        .block-container { padding-top: 1.2rem; max-width: 1200px; }

        .app-header {
            display: flex; justify-content: space-between; align-items: center;
            padding-bottom: 0.9rem; margin-bottom: 1.3rem;
            border-bottom: 1px solid #e3e6eb;
        }
        .app-header .brand { font-size: 1.35rem; font-weight: 700; letter-spacing: 0.02em; color: #14161a; }
        .app-header .subtitle { font-size: 0.85rem; color: #6b7280; margin-top: 0.1rem; }
        .status-pill {
            display: inline-flex; align-items: center; gap: 0.4rem;
            font-size: 0.75rem; font-weight: 600; letter-spacing: 0.03em;
            color: #1a7f37; background: #e9f7ee; border: 1px solid #cdeedb;
            padding: 0.25rem 0.65rem; border-radius: 4px;
        }
        .status-pill .dot { width: 7px; height: 7px; border-radius: 50%; background: #1a7f37; display: inline-block; }
        .header-meta { text-align: right; }
        .header-meta .timestamp { font-size: 0.72rem; color: #9ca3af; margin-top: 0.3rem; }

        .section-title { font-size: 1.05rem; font-weight: 700; color: #14161a; margin: 1.4rem 0 0.6rem 0; }
        .section-subtitle { font-size: 0.85rem; color: #6b7280; margin-bottom: 0.9rem; }

        .kpi-card {
            border: 1px solid #e3e6eb; border-radius: 6px; padding: 0.95rem 1.05rem;
            background: #ffffff; box-shadow: 0 1px 2px rgba(16,24,40,0.03);
        }
        .kpi-card .kpi-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em; color: #6b7280; text-transform: uppercase; }
        .kpi-card .kpi-value { font-size: 1.7rem; font-weight: 700; color: #14161a; margin-top: 0.25rem; }
        .kpi-card .kpi-sub { font-size: 0.75rem; color: #9ca3af; margin-top: 0.2rem; }

        .risk-panel {
            border-radius: 6px; padding: 1.1rem 1.3rem; border: 1px solid rgba(0,0,0,0.06);
        }
        .risk-panel .risk-label { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em; opacity: 0.85; }
        .risk-panel .risk-value { font-size: 2.1rem; font-weight: 800; margin: 0.15rem 0; }
        .risk-panel .risk-probability { font-size: 0.95rem; font-weight: 600; }
        .risk-panel .risk-explanation { font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.9; }

        .driver-row { display: flex; align-items: center; margin-bottom: 0.55rem; font-size: 0.82rem; }
        .driver-name { width: 190px; color: #374151; flex-shrink: 0; }
        .driver-bar-track { flex: 1; background: #eef0f3; border-radius: 3px; height: 9px; margin: 0 0.6rem; overflow: hidden; }
        .driver-bar-fill { background: #3b4a66; height: 100%; border-radius: 3px; }

        .rec-row { display: flex; gap: 0.7rem; padding: 0.5rem 0; border-bottom: 1px solid #f0f1f3; font-size: 0.87rem; color: #374151; }
        .rec-num { font-weight: 700; color: #9ca3af; width: 22px; flex-shrink: 0; }

        .caption-note { font-size: 0.75rem; color: #9ca3af; margin-top: 0.4rem; }
        hr.section-divider { border: none; border-top: 1px solid #e3e6eb; margin: 1.6rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            """
            <div class="app-header">
                <div>
                    <div class="brand">PIPELINE AI</div>
                    <div class="subtitle">AI-Powered Data Pipeline Reliability</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="header-meta">
                <span class="status-pill"><span class="dot"></span>SYSTEM OPERATIONAL</span>
                <div class="timestamp">Last analysis: {st.session_state.get('last_analysis', '—')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar() -> str:
    st.sidebar.markdown("### Navigation")
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Pipeline Monitor"

    for section, pages in NAV_SECTIONS.items():
        st.sidebar.markdown(f"**{section}**")
        for page in pages:
            is_active = st.session_state["active_page"] == page
            if st.sidebar.button(page, key=f"nav_{page}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state["active_page"] = page
        st.sidebar.markdown("")

    return st.session_state["active_page"]


def render_kpi_cards(kpis: dict) -> None:
    cols = st.columns(4)
    cards = [
        ("TOTAL PIPELINE RUNS", f"{kpis['total_runs']:,}", "Historical runs on record"),
        ("FAILURE RATE", f"{kpis['failure_rate'] * 100:.1f}%", "Share of historical runs that failed"),
        ("HIGH-RISK RUNS", f"{kpis['high_risk_runs']:,}", "Model-flagged probability ≥ 50%"),
        ("AVG EXECUTION TIME", f"{kpis['avg_execution_time']:.1f} min", "Across all historical runs"),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_pipeline_form() -> dict | None:
    st.markdown('<div class="section-title">Analyze Pipeline Execution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Enter operational telemetry for a run to estimate failure risk.</div>', unsafe_allow_html=True)

    with st.form("pipeline_form"):
        st.markdown("**Pipeline**")
        pipeline_name = st.selectbox("Pipeline name", PIPELINE_NAMES, label_visibility="collapsed")

        st.markdown("**Data Volume**")
        c1, c2 = st.columns(2)
        input_records = c1.number_input("Input records", min_value=0, value=500000, step=10000)
        data_size_mb = c2.number_input("Data size (MB)", min_value=0.0, value=1200.0, step=10.0)
        output_records = st.number_input("Output records", min_value=0, value=480000, step=10000)

        st.markdown("**Execution**")
        c3, c4, c5 = st.columns(3)
        execution_time_min = c3.number_input("Execution time (min)", min_value=0.0, value=30.0, step=1.0)
        expected_time_min = c4.number_input("Expected time (min)", min_value=0, value=25, step=1)
        previous_run_duration = c5.number_input("Previous run duration (min)", min_value=0.0, value=25.0, step=1.0)

        st.markdown("**Resource Utilization**")
        c6, c7, c8 = st.columns(3)
        cpu_utilization = c6.number_input("CPU utilization %", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
        memory_utilization = c7.number_input("Memory utilization %", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        worker_count = c8.number_input("Worker count", min_value=1, value=4, step=1)

        st.markdown("**Reliability**")
        c9, c10, c11 = st.columns(3)
        source_delay_min = c9.number_input("Source delay (min)", min_value=0.0, value=7.0, step=1.0)
        retry_count = c10.number_input("Retry count", min_value=0, value=1, step=1)
        previous_failure_count = c11.number_input("Previous failures", min_value=0, value=0, step=1)

        st.markdown("**Time**")
        c12, c13 = st.columns(2)
        day_of_week = c12.selectbox("Day of week", DAYS_OF_WEEK)
        hour = c13.number_input("Hour of day", min_value=0, max_value=23, value=9, step=1)

        submitted = st.form_submit_button("ANALYZE PIPELINE", type="primary", use_container_width=True)

    if not submitted:
        return None

    return {
        "pipeline_name": pipeline_name,
        "input_records": int(input_records),
        "output_records": int(output_records),
        "data_size_mb": float(data_size_mb),
        "execution_time_min": float(execution_time_min),
        "expected_time_min": int(expected_time_min),
        "cpu_utilization": float(cpu_utilization),
        "memory_utilization": float(memory_utilization),
        "source_delay_min": float(source_delay_min),
        "retry_count": int(retry_count),
        "previous_run_duration": float(previous_run_duration),
        "previous_failure_count": int(previous_failure_count),
        "worker_count": int(worker_count),
        "day_of_week": day_of_week,
        "hour": int(hour),
    }


def render_prediction_result(result: dict) -> str:
    probability = result["probability"]
    risk_category = get_risk_category(probability)
    style = RISK_STYLES[risk_category]

    explanations = {
        "LOW": "Operational metrics fall within normal historical ranges.",
        "MEDIUM": "Some metrics show mild deviation from typical successful runs.",
        "HIGH": "Pipeline execution shows elevated resource pressure and runtime deviation.",
        "CRITICAL": "Multiple metrics show severe deviation; failure risk is substantial.",
    }

    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="risk-panel" style="background:{style['bg']}; color:{style['color']};">
            <div class="risk-label">PIPELINE RISK</div>
            <div class="risk-value">{risk_category}</div>
            <div class="risk-probability">{probability * 100:.1f}% Failure Probability</div>
            <div class="risk-explanation">{explanations[risk_category]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="caption-note">Risk category is a UI-level banding of the model\'s '
        'raw probability output (LOW &lt;25%, MEDIUM &lt;50%, HIGH &lt;75%, CRITICAL ≥75%).</div>',
        unsafe_allow_html=True,
    )
    return risk_category


def render_risk_drivers(risk_drivers: list) -> None:
    st.markdown('<div class="section-title">Risk Drivers</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Contributing factors identified by the model (feature importance).</div>', unsafe_allow_html=True)
    for driver in risk_drivers:
        pct = max(driver["relative"] * 100, 3)
        st.markdown(
            f"""
            <div class="driver-row">
                <div class="driver-name">{driver['feature']}</div>
                <div class="driver-bar-track"><div class="driver-bar-fill" style="width:{pct:.0f}%;"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_recommendations(recommendations: list) -> None:
    st.markdown('<div class="section-title">Recommended Actions</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Recommended operational actions based on the submitted conditions above.</div>', unsafe_allow_html=True)
    for i, rec in enumerate(recommendations, start=1):
        st.markdown(f'<div class="rec-row"><div class="rec-num">{i:02d}</div><div>{rec}</div></div>', unsafe_allow_html=True)


def render_health_score(health: dict) -> None:
    st.markdown('<div class="section-title">Pipeline Health</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">UI-level heuristic score (not a model output) — '
        '90–100 Healthy · 70–89 Monitor · 50–69 Warning · 0–49 Critical.</div>',
        unsafe_allow_html=True,
    )
    st.metric("Health Score", f"{health['score']:.0f} / 100", health["category"])
    st.progress(int(health["score"]))


def render_what_if(base_payload: dict) -> None:
    st.markdown('<div class="section-title">What-if Simulation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Explore how infrastructure and workload changes could affect predicted failure risk. '
        'Every simulated value below is produced by the real trained model.</div>',
        unsafe_allow_html=True,
    )

    current_result = predict_pipeline_failure(base_payload)

    c1, c2 = st.columns(2)
    with c1:
        sim_workers = st.number_input("Workers", min_value=1, value=base_payload["worker_count"], key="sim_workers")
        sim_memory = st.slider("Memory utilization %", 0.0, 100.0, base_payload["memory_utilization"], key="sim_memory")
        sim_cpu = st.slider("CPU utilization %", 0.0, 100.0, base_payload["cpu_utilization"], key="sim_cpu")
    with c2:
        sim_input_records = st.number_input("Input records", min_value=0, value=base_payload["input_records"], step=10000, key="sim_input")
        sim_source_delay = st.number_input("Source delay (min)", min_value=0.0, value=base_payload["source_delay_min"], key="sim_delay")

    simulated_payload = dict(base_payload)
    simulated_payload.update(
        {
            "worker_count": int(sim_workers),
            "memory_utilization": float(sim_memory),
            "cpu_utilization": float(sim_cpu),
            "input_records": int(sim_input_records),
            "source_delay_min": float(sim_source_delay),
        }
    )
    simulated_result = predict_pipeline_failure(simulated_payload)

    col_current, col_arrow, col_sim = st.columns([2, 0.5, 2])
    with col_current:
        st.markdown("**CURRENT CONFIGURATION**")
        st.metric("Current Risk", f"{current_result['probability'] * 100:.0f}%")
    with col_arrow:
        st.markdown("<div style='text-align:center; padding-top:2rem;'>→</div>", unsafe_allow_html=True)
    with col_sim:
        st.markdown("**SIMULATED CONFIGURATION**")
        st.metric(
            "Simulated Risk",
            f"{simulated_result['probability'] * 100:.0f}%",
            delta=f"{(simulated_result['probability'] - current_result['probability']) * 100:.0f} pp",
            delta_color="inverse",
        )


def render_analytics(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Failure Analytics</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        by_pipeline = df.groupby("pipeline_name")["pipeline_failed"].mean().reset_index()
        by_pipeline["pipeline_failed"] *= 100
        fig = px.bar(by_pipeline, x="pipeline_name", y="pipeline_failed", labels={"pipeline_failed": "Failure Rate (%)", "pipeline_name": "Pipeline"})
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=320, title="Failure Rate by Pipeline")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(df, x="execution_time_min", nbins=30, labels={"execution_time_min": "Execution Time (min)"})
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=320, title="Execution Time Distribution")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(df, x="pipeline_failed", y="memory_utilization", labels={"pipeline_failed": "Failed (0/1)", "memory_utilization": "Memory Utilization %"})
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=320, title="Memory Utilization vs Failure")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.box(df, x="pipeline_failed", y="cpu_utilization", labels={"pipeline_failed": "Failed (0/1)", "cpu_utilization": "CPU Utilization %"})
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=320, title="CPU Utilization vs Failure")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df, x="predicted_probability", nbins=30, labels={"predicted_probability": "Predicted Failure Probability"})
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=320, title="Failure Probability Distribution")
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# Page orchestration
# --------------------------------------------------------------------------
def main() -> None:
    inject_css()
    render_header()
    active_page = render_sidebar()

    dataset_predictions = get_dataset_predictions()
    kpis = compute_kpis(dataset_predictions)

    st.markdown("## Pipeline Reliability Monitor")
    st.markdown('<div class="section-subtitle">Predict execution risk before pipeline failure.</div>', unsafe_allow_html=True)
    render_kpi_cards(kpis)
    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

    if active_page in ("Pipeline Monitor", "Risk Analysis"):
        payload = render_pipeline_form()
        if payload is not None:
            st.session_state["last_payload"] = payload
            st.session_state["last_analysis"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        if "last_payload" in st.session_state:
            payload = st.session_state["last_payload"]
            result = predict_pipeline_failure(payload)
            risk_drivers = get_risk_drivers()
            recommendations = build_recommendations(payload, risk_drivers)

            st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
            render_prediction_result(result)
            st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
            render_risk_drivers(risk_drivers)
            st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
            render_recommendations(recommendations)

    elif active_page == "Pipeline Health":
        if "last_payload" in st.session_state:
            health = compute_health_score(st.session_state["last_payload"])
            render_health_score(health)
        else:
            st.info("Run an analysis on Pipeline Monitor first to see a health score for that run.")

    elif active_page == "Failure Analytics":
        render_analytics(dataset_predictions)

    elif active_page == "What-if Simulator":
        if "last_payload" in st.session_state:
            render_what_if(st.session_state["last_payload"])
        else:
            st.info("Run an analysis on Pipeline Monitor first, then return here to simulate changes.")


if __name__ == "__main__":
    main()
