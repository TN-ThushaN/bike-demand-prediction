from pathlib import Path
from datetime import datetime, date, time

import altair as alt
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "gradient_boosting.joblib"
FEATURE_COLUMNS_PATH = PROJECT_DIR / "models" / "feature_columns.joblib"
PREPROCESSED_DATA_PATH = PROJECT_DIR / "results" / "02_preprocessed_data.csv"
MODEL_COMPARISON_PATH = PROJECT_DIR / "results" / "model_comparison.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_DIR / "results" / "feature_importance.csv"


# ============================================================
# DESIGN TOKENS
# ============================================================

INK = "#12303B"
TEAL = "#0F766E"
LANE = "#F2B705"
MIST = "#F1F5F4"
LINE = "#D9E2E0"
MUTED = "#5B6F76"

st.set_page_config(
    page_title="Bike Demand Forecast",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stApp {{ font-family: 'Manrope', 'Segoe UI', system-ui, sans-serif; }}
        .stApp {{ background: {MIST}; color: {INK}; }}
        .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1280px; }}
        footer, #MainMenu {{ visibility: hidden; }}
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none; }}

        /* Top bar */
        .topbar {{
            display: flex; align-items: center; justify-content: space-between;
            background: {INK}; color: #fff; border-radius: 14px;
            padding: 1.1rem 1.5rem; border-bottom: 5px solid {LANE}; margin-bottom: 1rem;
        }}
        .topbar h1 {{ color: #fff; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; padding: 0; }}
        .topbar p {{ color: #B9CCD2; font-size: 0.95rem; margin: 0.15rem 0 0 0; }}
        .topbar .tag {{ background: rgba(255,255,255,0.1); color: #fff; border-radius: 999px;
                        padding: 0.35rem 0.9rem; font-size: 0.85rem; font-weight: 600; white-space: nowrap; }}

        /* KPI strip */
        .kpi {{ background: #fff; border: 1px solid {LINE}; border-radius: 12px; padding: 0.85rem 1.1rem; color: {INK}; }}
        .kpi .k {{ color: {MUTED}; font-size: 0.8rem; font-weight: 600; }}
        .kpi .v {{ font-size: 1.45rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.25; }}

        /* Panels */
        h2, h3 {{ color: {INK}; letter-spacing: -0.01em; font-weight: 700; }}
        h3 {{ font-size: 1.15rem !important; margin-bottom: 0.2rem; }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; border-color: {LINE}; border-radius: 14px; }}
        label {{ color: {INK} !important; font-weight: 600 !important; }}
        .section-note {{ color: {MUTED}; font-size: 0.9rem; margin: 0 0 0.8rem 0; }}
        [data-testid="stCaptionContainer"] {{ color: {MUTED} !important; }}

        /* Button */
        .stButton > button[kind="primary"] {{
            background: {TEAL}; border: none; color: #fff; font-weight: 700;
            font-size: 1.02rem; padding: 0.75rem 1rem; border-radius: 10px;
        }}
        .stButton > button[kind="primary"]:hover {{ background: #0B5E58; color: #fff; }}
        .stButton > button:focus-visible {{ outline: 3px solid {LANE}; outline-offset: 2px; }}

        /* Result */
        .result {{ background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 1.4rem 1.5rem; color: {INK}; }}
        .result .label {{ color: {MUTED}; font-size: 0.9rem; font-weight: 600; }}
        .result .big {{ font-size: 3.1rem; font-weight: 800; line-height: 1.05; letter-spacing: -0.03em; }}
        .result .big small {{ font-size: 1.05rem; font-weight: 600; color: {MUTED}; letter-spacing: 0; }}
        .badge {{ display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.85rem;
                  font-weight: 700; background: {LANE}; color: {INK}; margin-left: 0.6rem; vertical-align: middle; }}
        .lane {{ position: relative; height: 16px; background: {MIST}; border: 1px solid {LINE};
                 border-radius: 999px; margin: 1.2rem 0 0.4rem 0; }}
        .lane .fill {{ height: 100%; background: {TEAL}; border-radius: 999px; }}
        .lane .dash {{ position: absolute; top: 50%; left: 0; right: 0; border-top: 2px dashed rgba(255,255,255,0.75); }}
        .lane .pin {{ position: absolute; top: -7px; width: 4px; height: 28px; background: {LANE};
                      border-radius: 2px; transform: translateX(-2px); }}
        .ticks {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: 0.78rem; font-weight: 600; }}
        .divider {{ border-top: 1px solid {LINE}; margin: 1.1rem 0 0.9rem 0; }}
        .facts {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.7rem 1rem; }}
        .facts .k {{ color: {MUTED}; font-size: 0.78rem; font-weight: 600; }}
        .facts .v {{ font-size: 0.95rem; font-weight: 700; }}

        .empty {{ background: #fff; border: 1px dashed {LINE}; border-radius: 14px; padding: 3.2rem 1.5rem;
                  text-align: center; color: {MUTED}; }}
        .empty b {{ color: {INK}; font-size: 1.05rem; }}

        .metric {{ background: #fff; border: 1px solid {LINE}; border-left: 5px solid {TEAL};
                   border-radius: 12px; padding: 0.9rem 1.2rem; color: {INK}; margin-bottom: 0.7rem; }}
        .metric .k {{ color: {MUTED}; font-size: 0.85rem; font-weight: 600; }}
        .metric .v {{ font-size: 1.8rem; font-weight: 800; letter-spacing: -0.02em; }}
        .metric .h {{ color: {MUTED}; font-size: 0.8rem; }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{ gap: 0.4rem; border-bottom: 1px solid {LINE}; }}
        .stTabs [data-baseweb="tab"] {{ font-weight: 700; color: {MUTED} !important; padding: 0.6rem 1.1rem; }}
        .stTabs [aria-selected="true"] {{ color: {INK} !important; }}
        .stTabs [data-baseweb="tab-highlight"] {{ background: {LANE}; height: 4px; }}

        @media (max-width: 900px) {{ .facts {{ grid-template-columns: 1fr 1fr; }} .topbar {{ flex-direction: column; align-items: flex-start; gap: 0.6rem; }} }}
        @media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD PROJECT FILES
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_feature_columns():
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Feature-column file not found: {FEATURE_COLUMNS_PATH}")
    return joblib.load(FEATURE_COLUMNS_PATH)


@st.cache_data
def load_project_data():
    if not PREPROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found: {PREPROCESSED_DATA_PATH}")
    return pd.read_csv(PREPROCESSED_DATA_PATH)


@st.cache_data
def load_model_comparison():
    if not MODEL_COMPARISON_PATH.exists():
        return None
    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_data
def load_feature_importance():
    if not FEATURE_IMPORTANCE_PATH.exists():
        return None
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


try:
    model = load_model()
    feature_columns = load_feature_columns()
    project_data = load_project_data()
    model_comparison = load_model_comparison()
    feature_importance = load_feature_importance()
except Exception as error:
    st.error("The app could not load its project files. Check that the paths below exist.")
    st.code(str(error))
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_expected_features():
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        return list(model_features)
    return list(feature_columns)


def prepare_input(future_datetime, holiday, workingday,
                  temperature, feels_like, humidity, weather):
    row = pd.DataFrame({
        "year": [int(future_datetime.year)],
        "month": [int(future_datetime.month)],
        "day": [int(future_datetime.day)],
        "hour": [int(future_datetime.hour)],
        "weekday": [int(future_datetime.weekday())],
        "holiday": [int(holiday)],
        "workingday": [int(workingday)],
        "temp": [float(temperature)],
        "feels_like": [float(feels_like)],
        "humidity": [float(humidity)],
        "weather_main": [str(weather)]
    })

    expected_features = [str(c) for c in get_expected_features()]

    if "weather_main" in expected_features:
        return row.reindex(columns=expected_features)

    row = pd.get_dummies(row, columns=["weather_main"], prefix="weather_main", dtype=int)
    row = row.reindex(columns=expected_features, fill_value=0)
    for column in row.columns:
        row[column] = pd.to_numeric(row[column], errors="raise")
    return row.astype(float)


def predict_percentage(dt, holiday, workingday, temperature, feels_like, humidity, weather):
    """One validated prediction, clamped to the 0-100 demand scale."""
    data = prepare_input(dt, holiday, workingday, temperature, feels_like, humidity, weather)
    expected = [str(c) for c in get_expected_features()]
    if list(data.columns) != expected:
        raise ValueError("Prediction input columns do not match the trained model.")
    if data.isnull().any().any():
        raise ValueError("Prediction input contains missing values.")
    return max(0.0, min(100.0, float(model.predict(data)[0])))


def demand_label(pct):
    if pct < 25:
        return "Low"
    if pct < 50:
        return "Moderate"
    if pct < 75:
        return "High"
    return "Very high"


def _flat(html):
    """Strip blank lines and indentation so Markdown never renders HTML as code."""
    return "".join(line.strip() for line in html.splitlines() if line.strip())


MAX_BIKE_COUNT = float(project_data["count"].max())
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEATHER_OPTIONS = ["Clear", "Clouds", "Drizzle", "Fog", "Haze",
                   "Mist", "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"]


def gb_row():
    if model_comparison is None:
        return None
    rows = model_comparison[
        model_comparison["Model"].astype(str).str.contains("Gradient Boosting", case=False, na=False)
    ]
    return None if rows.empty else rows.iloc[0]


# ============================================================
# TOP BAR + KPI STRIP
# ============================================================

st.markdown(
    _flat("""
    <div class="topbar">
        <div>
            <h1>🚲 Bike Demand Forecast</h1>
            <p>Estimate shared-bike demand for any date, time and weather scenario.</p>
        </div>
        <div class="tag">Gradient Boosting model</div>
    </div>
    """),
    unsafe_allow_html=True,
)

g = gb_row()
kpis = [
    ("Model accuracy (R²)", f"{float(g['R2']):.3f}" if g is not None else "n/a"),
    ("Average error (MAE)", f"{float(g['MAE']):.2f}" if g is not None else "n/a"),
    ("Training rows", f"{len(project_data):,}"),
    ("Highest recorded count", f"{int(MAX_BIKE_COUNT):,} bikes"),
]
for col, (k, v) in zip(st.columns(4), kpis):
    with col:
        st.markdown(f'<div class="kpi"><div class="k">{k}</div><div class="v">{v}</div></div>',
                    unsafe_allow_html=True)

st.write("")

tab_predict, tab_insights = st.tabs(["Forecast", "Model insights"])


# ------------------------------------------------------------
# TAB 1: FORECAST
# ------------------------------------------------------------
with tab_predict:
    left, right = st.columns([5, 7], gap="large")

    # ---------------- Inputs ----------------
    with left:
        with st.container(border=True):
            st.subheader("Scenario")
            st.markdown('<div class="section-note">Choose when and under what weather.</div>',
                        unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                future_date = st.date_input("Date", value=date.today())
            with c2:
                future_time = st.time_input("Time", value=time(13, 0), step=900)

            future_datetime = datetime.combine(future_date, future_time)
            weekday = future_datetime.weekday()

            c3, c4 = st.columns(2)
            with c3:
                holiday = st.selectbox("Public holiday", [0, 1],
                                       format_func=lambda x: "No" if x == 0 else "Yes")
            with c4:
                workingday = st.selectbox(
                    "Working day", [0, 1], index=0 if weekday >= 5 else 1,
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    help="Set from the weekday. Change it for special cases.")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            w1, w2 = st.columns(2)
            with w1:
                temperature = st.number_input("Temperature (°C)", -20.0, 50.0, 25.0, 0.5)
            with w2:
                feels_like = st.number_input("Feels like (°C)", -20.0, 50.0, 25.0, 0.5)

            w3, w4 = st.columns(2)
            with w3:
                humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, 1.0)
            with w4:
                weather = st.selectbox("Condition", WEATHER_OPTIONS)

            predict_clicked = st.button("Forecast demand", type="primary", use_container_width=True)

        if predict_clicked:
            try:
                pct = predict_percentage(future_datetime, holiday, workingday,
                                         temperature, feels_like, humidity, weather)
                bikes = round(pct / 100.0 * MAX_BIKE_COUNT)

                # Same day and weather for all 24 hours
                hourly = []
                for h in range(24):
                    dt_h = datetime.combine(future_date, time(h, 0))
                    p = predict_percentage(dt_h, holiday, workingday,
                                           temperature, feels_like, humidity, weather)
                    hourly.append(round(p / 100.0 * MAX_BIKE_COUNT))

                st.session_state["result"] = {
                    "pct": pct, "bikes": bikes, "hour": future_datetime.hour,
                    "when": future_datetime.strftime("%d %b %Y, %H:%M"),
                    "day": WEEKDAYS[weekday],
                    "holiday": "Yes" if holiday else "No",
                    "workingday": "Yes" if workingday else "No",
                    "temp": f"{temperature:.1f} °C", "feels": f"{feels_like:.1f} °C",
                    "humidity": f"{humidity:.0f}%", "weather": weather,
                    "hourly": hourly,
                }
            except Exception as error:
                st.session_state.pop("result", None)
                st.error("The forecast failed. See the details below.")
                st.exception(error)

    # ---------------- Results ----------------
    with right:
        res = st.session_state.get("result")

        if res is None:
            st.markdown(
                _flat("""
                <div class="empty"><b>No forecast yet</b><br>
                Set the scenario on the left, then select <i>Forecast demand</i>.</div>
                """),
                unsafe_allow_html=True,
            )
        else:
            pct = res["pct"]
            st.markdown(
                _flat(f"""
                <div class="result">
                    <div class="label">Estimated demand for {res['when']}</div>
                    <div class="big">{res['bikes']:,} <small>bikes</small>
                        <span class="badge">{demand_label(pct)}</span></div>
                    <div class="lane">
                        <div class="fill" style="width:{pct:.1f}%"></div>
                        <div class="dash"></div>
                        <div class="pin" style="left:{pct:.1f}%"></div>
                    </div>
                    <div class="ticks"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
                    <div class="label" style="margin-top:0.5rem">{pct:.1f}% of the busiest recorded period</div>
                    <div class="divider"></div>
                    <div class="facts">
                        <div><div class="k">Day</div><div class="v">{res['day']}</div></div>
                        <div><div class="k">Weather</div><div class="v">{res['weather']}</div></div>
                        <div><div class="k">Temperature</div><div class="v">{res['temp']}</div></div>
                        <div><div class="k">Humidity</div><div class="v">{res['humidity']}</div></div>
                        <div><div class="k">Holiday</div><div class="v">{res['holiday']}</div></div>
                        <div><div class="k">Working day</div><div class="v">{res['workingday']}</div></div>
                    </div>
                </div>
                """),
                unsafe_allow_html=True,
            )

            st.write("")
            with st.container(border=True):
                st.subheader("Demand across the day")
                hourly = res["hourly"]
                peak_hour = max(range(24), key=lambda i: hourly[i])
                st.markdown(
                    f'<div class="section-note">Same weather all day. Busiest hour: '
                    f'<b>{peak_hour:02d}:00</b> (about {hourly[peak_hour]:,} bikes). '
                    f'The yellow dot marks your selected time.</div>',
                    unsafe_allow_html=True,
                )

                df = pd.DataFrame({"Hour": list(range(24)), "Bikes": hourly})
                base = alt.Chart(df).encode(
                    x=alt.X("Hour:Q", scale=alt.Scale(domain=[0, 23]),
                            axis=alt.Axis(values=list(range(0, 24, 3)), title="Hour of day")),
                    y=alt.Y("Bikes:Q", title="Estimated bikes"),
                    tooltip=["Hour", "Bikes"],
                )
                area = base.mark_area(color=TEAL, opacity=0.15)
                line = base.mark_line(color=TEAL, strokeWidth=3)
                marker = base.transform_filter(alt.datum.Hour == res["hour"]).mark_point(
                    color=LANE, size=190, filled=True, stroke=INK, strokeWidth=2, opacity=1)
                chart = (area + line + marker).properties(height=240).configure_axis(
                    labelColor=INK, titleColor=MUTED, gridColor=LINE).configure_view(strokeWidth=0)
                st.altair_chart(chart, use_container_width=True)

            st.caption("This estimate assumes the weather you entered occurs at the "
                       "selected time. It is not a weather forecast.")


# ------------------------------------------------------------
# TAB 2: MODEL INSIGHTS
# ------------------------------------------------------------
with tab_insights:
    a, b = st.columns([7, 5], gap="large")

    with a:
        with st.container(border=True):
            st.subheader("What drives demand?")
            st.markdown('<div class="section-note">The ten inputs with the most influence on the prediction.</div>',
                        unsafe_allow_html=True)
            if feature_importance is not None and not feature_importance.empty:
                imp = feature_importance.copy()
                imp = imp[imp["Importance"] > 0].sort_values("Importance", ascending=False).head(10)
                imp["Importance (%)"] = (imp["Importance"] * 100).round(2)
                chart = (
                    alt.Chart(imp)
                    .mark_bar(color=TEAL, cornerRadiusEnd=4, size=20)
                    .encode(
                        x=alt.X("Importance (%):Q", title="Importance (%)"),
                        y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
                        tooltip=["Feature", "Importance (%)"],
                    )
                    .properties(height=max(260, 34 * len(imp)))
                    .configure_axis(labelColor=INK, titleColor=MUTED, gridColor=LINE)
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No feature importance file found in the results folder.")

    with b:
        st.subheader("Accuracy on test data")
        st.markdown('<div class="section-note">Scores for the selected model.</div>',
                    unsafe_allow_html=True)
        if g is not None:
            for name, val, hint in [
                ("MAE", g["MAE"], "Average error. Lower is better."),
                ("RMSE", g["RMSE"], "Penalises large misses. Lower is better."),
                ("R²", g["R2"], "Share of variation explained. Higher is better."),
            ]:
                st.markdown(
                    f'<div class="metric"><div class="k">{name}</div>'
                    f'<div class="v">{float(val):.4f}</div><div class="h">{hint}</div></div>',
                    unsafe_allow_html=True)
        else:
            st.info("No model comparison file found in the results folder.")

    st.write("")
    with st.container(border=True):
        st.subheader("Models compared")
        st.markdown('<div class="section-note">The selected model is highlighted.</div>',
                    unsafe_allow_html=True)
        if model_comparison is not None:
            table = model_comparison.copy()
            num_cols = [c for c in ["MAE", "RMSE", "R2"] if c in table.columns]
            for c in num_cols:
                table[c] = table[c].astype(float).round(4)

            def highlight(row):
                is_gb = "gradient boosting" in str(row.get("Model", "")).lower()
                style = f"background-color: #FFF4CC; color: {INK}; font-weight: 700" if is_gb else ""
                return [style] * len(row)

            st.dataframe(
                table.style.apply(highlight, axis=1).format({c: "{:.4f}" for c in num_cols}),
                use_container_width=True, hide_index=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f'<div style="text-align:center; color:{MUTED}; font-size:0.85rem; margin-top:2rem;">'
    f'Bike Demand Forecast · Machine Learning Capstone Project</div>',
    unsafe_allow_html=True,
)