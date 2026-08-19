import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="AutoPricer | Profit Optimiser",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Refined Premium CSS (Minimalist Dark SaaS) ---
st.markdown(
    """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Force Main App Background */
    .stApp {
        background-color: #000000 !important;
        color: #FAFAFA !important;
    }

    /* Hide Streamlit Header & Toolbar */
    header {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}

    /* Top padding reduction */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #FAFAFA !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    p, span, div {
        color: #A1A1AA; /* Muted zinc-400 for secondary text */
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #18181B; /* zinc-900 */
        border: 1px solid #27272A; /* zinc-800 */
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        border-color: #3F3F46; /* zinc-700 */
        transform: translateY(-1px);
    }

    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #FAFAFA !important; /* Crisp white for primary metric numbers */
        letter-spacing: -0.02em;
        margin-top: 8px;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #A1A1AA !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0A0A0A;
        border-right: 1px solid #27272A;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #FAFAFA;
        color: #0A0A0A !important;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    .stButton>button:hover {
        background-color: #E4E4E7; /* zinc-200 */
        color: #0A0A0A !important;
    }
    
    /* Form bounding box */
    div[data-testid="stForm"] {
        background-color: #18181B;
        border: 1px solid #27272A;
        border-radius: 8px;
        padding: 30px;
    }
    
    /* Custom divider */
    hr {
        border-color: #27272A;
        margin-top: 2.5rem;
        margin-bottom: 2.5rem;
    }

    /* Result Box */
    .result-box {
        background-color: #18181B;
        padding: 32px;
        border-radius: 8px;
        border: 1px solid #27272A;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .result-offer {
        color: #FAFAFA;
        font-size: 4rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 10px 0;
    }
    .result-label {
        color: #A1A1AA;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Common Plotly Layout tweaks (Minimalist)
PLOTLY_THEME = {
    "plot_bgcolor": "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#A1A1AA", "family": "Inter, sans-serif"},
    "title_font": {"size": 18, "color": "#FAFAFA", "weight": 600},
    "margin": dict(l=40, r=20, t=60, b=40),
    "xaxis": dict(showgrid=False, zeroline=False, color="#A1A1AA", tickfont=dict(color="#A1A1AA")),
    "yaxis": dict(
        showgrid=True,
        gridcolor="#27272A",
        zeroline=False,
        color="#A1A1AA",
        tickfont=dict(color="#A1A1AA"),
    ),
}


# Helpers
@st.cache_data
def load_reports():
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    drift, perf = {}, {}
    try:
        with open(os.path.join(reports_dir, "drift_report.json")) as f:
            drift = json.load(f)
    except Exception:
        pass
    try:
        with open(os.path.join(reports_dir, "performance_report.json")) as f:
            perf = json.load(f)
    except Exception:
        pass
    return drift, perf


@st.cache_resource
def get_session():
    return requests.Session()


# --- Sidebar ---
st.sidebar.markdown(
    """
<div style="margin-bottom: 30px; padding: 0 10px;">
    <h1 style="font-size: 1.5rem; font-weight: 700; color: #FAFAFA !important; margin-bottom: 4px;">AutoPricer</h1>
    <p style="color: #71717A; font-size: 0.85rem; margin-top: 0px;">Algorithmic Vehicle Pricing</p>
</div>
""",
    unsafe_allow_html=True,
)

page = st.sidebar.radio("", ["Overview", "Drift Diagnostics", "Policy Simulator"])

drift_report, perf_report = load_reports()

if page == "Overview":
    st.markdown("<h1 style='font-size: 2rem;'>Commercial Overview</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#A1A1AA; font-size:1.05rem;'>Expected Value vs Realised Margin & High-Level KPIs</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Profit (Actual)", f"£{perf_report.get('total_profit_actual', 0):,.0f}")
    with col2:
        st.metric("Win Rate", f"{perf_report.get('win_rate', 0) * 100:.1f}%")
    with col3:
        st.metric("Price Model MAE", f"£{perf_report.get('price_mae', 0)}")
    with col4:
        st.metric("Conversion Brier Score", f"{perf_report.get('conversion_brier', 0):.3f}")

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='font-size: 1.3rem; margin-bottom: 20px;'>Profit Attribution Simulator</h3>",
        unsafe_allow_html=True,
    )

    # Mock chart for illustration
    dates = pd.date_range("2026-01-01", periods=30)
    profit_data = pd.DataFrame(
        {
            "Date": dates,
            "AutoPricer EV Policy": np.random.uniform(2000, 5000, 30).cumsum(),
            "Legacy Flat 85%": np.random.uniform(1500, 4000, 30).cumsum(),
        }
    )

    fig = px.line(
        profit_data,
        x="Date",
        y=["AutoPricer EV Policy", "Legacy Flat 85%"],
        color_discrete_sequence=["#FAFAFA", "#52525B"],
        labels={"value": "Profit (£)", "variable": "Policy"},
    )

    fig.update_layout(**PLOTLY_THEME)
    fig.update_layout(
        title="Expected Margin Trajectory",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title=None,
            font=dict(color="#A1A1AA"),
        ),
        hovermode="x unified",
        height=400,
    )
    # Fill area under curve for AutoPricer EV Policy with subtle white gradient
    fig.update_traces(
        fill="tozeroy",
        selector=dict(name="AutoPricer EV Policy"),
        fillcolor="rgba(250, 250, 250, 0.05)",
    )

    st.plotly_chart(fig, use_container_width=True)

elif page == "Drift Diagnostics":
    st.markdown("<h1 style='font-size: 2rem;'>Drift & Performance</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#A1A1AA; font-size:1.05rem;'>Tracking Population Stability and Distribution Shifts in Real-Time</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if drift_report.get("features"):
        features = list(drift_report["features"].keys())
        psis = [drift_report["features"][f]["psi"] for f in features]
        # Minimalist colors: White for stable, muted blue/red for warnings
        colors = ["#EF4444" if p > 0.25 else "#F59E0B" if p > 0.1 else "#FAFAFA" for p in psis]

        fig = go.Figure(
            data=[go.Bar(x=features, y=psis, marker_color=colors, marker_line_width=0, width=0.4)]
        )

        fig.add_hline(
            y=0.25,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text="Critical Threshold",
            annotation_position="top right",
            annotation_font_color="#EF4444",
        )
        fig.add_hline(
            y=0.10,
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text="Warning Threshold",
            annotation_position="top right",
            annotation_font_color="#F59E0B",
        )

        fig.update_layout(**PLOTLY_THEME)
        fig.update_layout(title="", yaxis_title="PSI Score", height=350)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='font-size: 1.2rem; margin-bottom: 15px;'>KS Test Breakdown</h3>",
            unsafe_allow_html=True,
        )
        df_ks = pd.DataFrame(
            {
                "Feature Indicator": features,
                "PSI Score": [f"{p:.3f}" for p in psis],
                "KS p-value": [
                    f"{drift_report['features'][f]['ks_p_value']:.4f}" for f in features
                ],
                "Status": [
                    "Drift Detected" if drift_report["features"][f]["drift_detected"] else "Stable"
                    for f in features
                ],
            }
        )
        st.dataframe(df_ks, use_container_width=True, hide_index=True)
    else:
        st.warning("No drift report found. Run pipelines/monitor/drift.py")

elif page == "Policy Simulator":
    st.markdown("<h1 style='font-size: 2rem;'>Policy Simulator</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#A1A1AA; font-size:1.05rem;'>Test the Expected Value (EV) Pricing Engine</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown(
            "<h3 style='font-size: 1.2rem; margin-bottom: 15px;'>DVLA Vehicle Lookup (UK)</h3>",
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            col_l1, col_l2 = st.columns([3, 1])
            with col_l1:
                reg_input = st.text_input(
                    "Registration Plate", placeholder="e.g. AB12 CDE", label_visibility="collapsed"
                )
            with col_l2:
                btn_lookup = st.button("Lookup", use_container_width=True)

            if btn_lookup and reg_input:
                api_url = os.getenv("AUTOPRICER_API_URL", "http://localhost:8000")
                api_key = os.getenv("API_KEY", "default-dev-key")
                with st.spinner("Fetching from DVLA/DVSA..."):
                    try:
                        session = get_session()
                        headers = {"X-API-Key": api_key}
                        res = session.get(
                            f"{api_url}/lookup", params={"reg": reg_input}, headers=headers
                        )
                        if res.status_code == 200:
                            data = res.json()
                            st.session_state.dvla_data = data
                            source_label = data.get("source", "UNKNOWN").replace("_", " ")
                            st.success(
                                f"**Found: {data['make']} {data['model']}**  \n"
                                f"MOT Status: `{data['mot_status']}` ({data['mot_days_remaining']} days left)  \n"
                                f"Data Source: `{source_label}`"
                            )
                        else:
                            st.error(f"Error: {res.json().get('detail', 'Vehicle not found')}")
                    except Exception as e:
                        st.error(f"API Connection Error: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='font-size: 1.2rem; margin-bottom: 15px;'>Vehicle Parameters</h3>",
            unsafe_allow_html=True,
        )

        # Load Makes/Models
        makes_models = {"Ford": ["Focus"], "BMW": ["3 Series"]}
        try:
            mm_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "seed", "assets", "makes_models.json"
            )
            if os.path.exists(mm_path):
                with open(mm_path, "r") as f:
                    makes_models = json.load(f)
        except Exception:
            pass

        with st.container(border=True):
            # Pre-fill from DVLA if available
            dvla_info = st.session_state.get("dvla_data", {})
            pref_make = dvla_info.get("make")
            pref_model = dvla_info.get("model")
            pref_fuel = dvla_info.get("fuel_type")
            pref_year = dvla_info.get("year", 2019)
            pref_mileage = dvla_info.get("mileage", 45000)

            make_opts = list(makes_models.keys()) + ["Other (type manually)"]

            # Smart index for make
            make_idx = 0
            if pref_make:
                match = [i for i, m in enumerate(make_opts) if pref_make.lower() == m.lower()]
                make_idx = match[0] if match else len(make_opts) - 1

            make = st.selectbox("Make", make_opts, index=make_idx)
            final_make = make
            if make == "Other (type manually)":
                final_make = st.text_input("Enter Make", value=pref_make if pref_make else "")

            model_opts = makes_models.get(make, ["Generic"]) + ["Other (type manually)"]

            # Smart index for model
            model_idx = 0
            if pref_model:
                match = [i for i, m in enumerate(model_opts) if pref_model.lower() == m.lower()]
                model_idx = match[0] if match else len(model_opts) - 1

            model = st.selectbox("Model", model_opts, index=model_idx)
            final_model = model
            if model == "Other (type manually)":
                final_model = st.text_input("Enter Model", value=pref_model if pref_model else "")

            fuel_opts = ["petrol", "diesel", "electric", "hybrid"]
            fuel_idx = (
                fuel_opts.index(pref_fuel.lower())
                if pref_fuel and pref_fuel.lower() in fuel_opts
                else 0
            )
            fuel = st.selectbox("Fuel", fuel_opts, index=fuel_idx)

            years = list(range(2025, 2009, -1))
            year_idx = years.index(pref_year) if pref_year in years else len(years) // 2
            year = st.selectbox("Year", years, index=year_idx)

            mileage = st.number_input("Mileage", 0, 500000, pref_mileage, step=1000)
            channel = st.selectbox("Acquisition Channel", ["dealer", "private", "fleet"])
            damage = st.selectbox(
                "Damage Severity", ["none", "scratches", "dents", "mechanical", "structural"]
            )

            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.button("Generate Offer", use_container_width=True)

    with col2:
        st.markdown(
            "<h3 style='font-size: 1.2rem; margin-bottom: 15px;'>Engine Decision</h3>",
            unsafe_allow_html=True,
        )
        if submit:
            payload = {
                "make": final_make,
                "model": final_model,
                "year": year,
                "mileage": mileage,
                "fuel_type": fuel,
                "channel": channel,
                "damage_flag": damage != "none",
                "damage_type": damage,
            }

            api_url = os.getenv("AUTOPRICER_API_URL", "http://localhost:8000")
            api_key = os.getenv("API_KEY", "default-dev-key")
            with st.spinner("Calculating Expected Value..."):
                try:
                    session = get_session()
                    headers = {"X-API-Key": api_key}
                    res = session.post(f"{api_url}/quote", json=payload, headers=headers)
                    if res.status_code == 200:
                        data = res.json()
                        offer = data["recommended_offer"]
                        ev = data["expected_value"]
                        p_win = data["p_win"]
                        risk = data["risk_band"].upper()

                        with st.container(border=True):
                            # Highly refined minimalist Result Box
                            st.markdown(
                                f"""
                            <div class='result-box'>
                                <div class='result-label'>Recommended Offer</div>
                                <div class='result-offer'>£{offer:,.0f}</div>
                                <div style='color: #71717A; font-size: 1rem;'>Expected Profit: <span style='color: #FAFAFA; font-weight: 500;'>£{ev:,.0f}</span></div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            st.markdown("<br>", unsafe_allow_html=True)

                        # Clean minimalist gauges
                        g1, g2 = st.columns(2)

                        fig_win = go.Figure(
                            go.Indicator(
                                mode="number+gauge",
                                value=p_win * 100,
                                number={
                                    "suffix": "%",
                                    "font": {"color": "#FAFAFA", "size": 32, "family": "Inter"},
                                },
                                title={
                                    "text": "Win Probability",
                                    "font": {"color": "#A1A1AA", "size": 14, "family": "Inter"},
                                },
                                gauge={
                                    "axis": {"range": [0, 100], "visible": False},
                                    "bar": {"color": "#FAFAFA", "thickness": 0.2},
                                    "bgcolor": "#27272A",
                                    "borderwidth": 0,
                                },
                            )
                        )
                        fig_win.update_layout(
                            height=200,
                            margin=dict(l=20, r=20, t=30, b=20),
                            paper_bgcolor="rgba(0,0,0,0)",
                            font={"color": "#FAFAFA"},
                        )
                        with g1:
                            st.plotly_chart(fig_win, use_container_width=True)

                        risk_val = 20 if risk == "LOW" else 50 if risk == "MEDIUM" else 80
                        risk_color = (
                            "#FAFAFA"
                            if risk == "LOW"
                            else "#F59E0B" if risk == "MEDIUM" else "#EF4444"
                        )

                        fig_risk = go.Figure(
                            go.Indicator(
                                mode="gauge",
                                value=risk_val,
                                title={
                                    "text": f"Risk Band: {risk}",
                                    "font": {"color": "#A1A1AA", "size": 14, "family": "Inter"},
                                },
                                gauge={
                                    "axis": {"range": [0, 100], "visible": False},
                                    "bar": {"color": risk_color, "thickness": 0.2},
                                    "bgcolor": "#27272A",
                                    "borderwidth": 0,
                                },
                            )
                        )
                        fig_risk.update_layout(
                            height=200,
                            margin=dict(l=20, r=20, t=30, b=20),
                            paper_bgcolor="rgba(0,0,0,0)",
                            font={"color": "#FAFAFA"},
                        )
                        with g2:
                            st.plotly_chart(fig_risk, use_container_width=True)

                        # Explainability
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(
                            "<div style='background-color: #0A0A0A; border: 1px solid #27272A; padding: 20px; border-radius: 8px;'>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            "<p style='color: #FAFAFA; font-weight: 500; margin-bottom: 10px; margin-top: 0;'>Engine Explainability</p>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"""
                        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'><span style='color: #A1A1AA;'>Retail Estimate (E_sale)</span><span style='color: #FAFAFA;'>£{data['explanation']['e_sale']:,.0f}</span></div>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'><span style='color: #A1A1AA;'>Expected Costs (E_costs)</span><span style='color: #FAFAFA;'>£{data['explanation']['e_costs']:,.0f}</span></div>
                        <div style='display: flex; justify-content: space-between;'><span style='color: #A1A1AA;'>Tail Risk Penalty (q10)</span><span style='color: #FAFAFA;'>£{data['explanation']['tail_penalty']:,.0f}</span></div>
                        """,
                            unsafe_allow_html=True,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                    else:
                        st.error(f"API Error: {res.text}")
                except Exception as e:
                    st.error(f"Could not connect to API at {api_url}: {str(e)}")
                    st.info("Make sure the FastAPI app is running.")
        else:
            st.markdown(
                "<div style='padding: 20px; text-align: center; color: #71717A; border: 1px dashed #27272A; border-radius: 8px; margin-top: 20px;'>Enter vehicle attributes and generate an offer.</div>",
                unsafe_allow_html=True,
            )
