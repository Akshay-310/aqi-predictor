"""
Karachi AQI — Field Report
A 3-day AQI forecast dashboard for Karachi, Pakistan.

Run with: streamlit run app.py
Requires: HOPSWORKS_HOST, HOPSWORKS_PROJECT, HOPSWORKS_API_KEY in .env
"""

import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
import hopsworks
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Karachi AQI — Field Report",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Category scale — dusty/smog tones instead of stock traffic-light colors
CATEGORY_SCALE = [
    (0, 50, "Good", "#5E8F5C"),
    (51, 100, "Moderate", "#BC8A2E"),
    (101, 150, "Unhealthy (Sensitive)", "#B8623A"),
    (151, 200, "Unhealthy", "#9C3F33"),
    (201, 300, "Very Unhealthy", "#733650"),
    (301, 500, "Hazardous", "#452133"),
]

ADVISORY = {
    "Good": "Air quality is satisfactory. Normal outdoor activity is fine for everyone.",
    "Moderate": "Acceptable air quality. Unusually sensitive individuals should consider limiting prolonged outdoor exertion.",
    "Unhealthy (Sensitive)": "Children, older adults, and people with asthma or heart conditions should reduce prolonged outdoor exertion — a mask outdoors is a reasonable precaution.",
    "Unhealthy": "Everyone may begin to notice effects. Sensitive groups should avoid outdoor exertion; the general public should limit prolonged time outside.",
    "Very Unhealthy": "Health alert. Avoid outdoor exertion — everyone. Keep windows closed and run an air purifier indoors if you have one.",
    "Hazardous": "Emergency conditions. Stay indoors, seal windows and doors, and avoid all outdoor exertion.",
}

# Which model won each horizon — from your train_models.py results.
# Update this if you retrain and a different algo wins a horizon.
HORIZON_MODELS = {1: "Ridge Regression", 2: "Ridge Regression", 3: "LightGBM"}

# Mirrors train_models.py exactly — feature order and naming must match
# what each registered model was trained on.
BASE_FEATURES = [
    "us_aqi", "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "wind_direction_10m", "surface_pressure",
    "precipitation", "shortwave_radiation", "day_of_week", "month",
    "day_of_year", "aqi_change_rate", "aqi_roll3", "aqi_roll7", "pm25_roll3",
]
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "surface_pressure", "precipitation",
    "shortwave_radiation",
]
HORIZON_NAMES = {1: "day1", 2: "day2", 3: "day3"}

HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")


def categorize(aqi: float):
    for lo, hi, name, color in CATEGORY_SCALE:
        if lo <= aqi <= hi:
            return name, color
    return "Hazardous", CATEGORY_SCALE[-1][3]


def severity_index(category_name: str) -> int:
    names = [c[2] for c in CATEGORY_SCALE]
    return names.index(category_name)


# ══════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; }
        .stApp { background-color: #F6F1E7; color: #2B2A26; }
        section[data-testid="stSidebar"] { background-color: #EFE8D8; border-right: 1px solid #DED4BE; }

        h1, h2, h3 { font-family: 'Fraunces', serif !important; color: #26241F !important; font-weight: 600 !important; }
        p, span, label, div { color: #3B392F; }
        .muted { color: #857F6E !important; font-size: 0.85rem; }

        .aqi-number { font-family: 'IBM Plex Mono', monospace; font-weight: 600; line-height: 1; color: #26241F; }

        .card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 1.4rem 1.6rem;
            border: 1px solid #E6DFCC;
            box-shadow: 0 1px 3px rgba(43, 42, 38, 0.04);
            height: 100%;
        }
        .badge {
            display: inline-block; padding: 3px 12px; border-radius: 20px;
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
            text-transform: uppercase; color: #FFFFFF;
        }
        .model-tag {
            display: inline-block; margin-top: 0.6rem; padding: 2px 10px;
            border-radius: 6px; font-size: 0.7rem; color: #857F6E;
            border: 1px solid #E6DFCC; font-family: 'IBM Plex Mono', monospace;
        }
        hr { border-color: #E6DFCC; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def horizon_skyline(aqi: float, height: int = 110) -> str:
    """
    Signature visual: a skyline where distant buildings fade out as AQI
    worsens — visibility loss standing in for the AQI number itself.
    """
    visibility = max(0.08, 1 - (aqi / 300))  # 1 = clear, ~0 = hazardous
    _, color = categorize(aqi)

    n_layers = 5
    rows = []
    for i in range(n_layers):
        depth = i / (n_layers - 1)  # 0 = farthest, 1 = nearest
        base_opacity = 0.30 + depth * 0.35  # nearer layers read more solid even in clear air
        layer_opacity = min(1.0, max(0.06, base_opacity * (0.35 + 0.65 * visibility)))
        y_base = 24 + i * 15
        h_min, h_max = 14 + i * 9, 28 + i * 13
        rng = np.random.RandomState(i * 7 + 3)

        buildings = []
        x = -10
        while x < 630:
            w = int(rng.randint(20, 40))
            h = int(rng.randint(h_min, h_max))
            buildings.append(
                f'<rect x="{x}" y="{height - y_base - h}" width="{w}" height="{h}" '
                f'fill="{color}" opacity="{layer_opacity:.2f}" rx="1.5" />'
            )
            x += w + int(rng.randint(10, 24))  # gap sized independently of width — no overlap
        rows.append("".join(buildings))

    svg = f"""
    <svg viewBox="0 0 620 {height}" width="100%" height="{height}" preserveAspectRatio="none"
         style="border-radius:10px; background:linear-gradient(180deg, #FBF7EC 0%, #F1E9D4 100%);">
        {''.join(rows)}
    </svg>
    """
    return svg


# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Connecting to Hopsworks...")
def get_project():
    return hopsworks.login(
        host=HOPSWORKS_HOST,
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
        engine="python",
    )


@st.cache_data(ttl=900, show_spinner="Loading historical data...")
def load_historical() -> pd.DataFrame:
    project = get_project()
    fs = project.get_feature_store()
    daily_fg = fs.get_feature_group(name="aqi_daily_features", version=2)
    df = daily_fg.read()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner="Loading latest reading...")
def load_current() -> dict:
    project = get_project()
    fs = project.get_feature_store()
    raw_fg = fs.get_feature_group(name="aqi_raw_hourly", version=1)
    df = raw_fg.read()
    df["time"] = pd.to_datetime(df["time"])
    latest = df.sort_values("time").iloc[-1]

    # fetch_current_data.py requests Open-Meteo data with timezone=Asia/Karachi,
    # so the stored value is ALREADY Karachi local clock time — Hopsworks just
    # tags it "+00:00" by default since no real tz info was attached. Strip
    # that incorrect tag rather than converting (converting would double-shift
    # it 5 hours forward, which is what the previous version of this function
    # did — worth knowing if this code gets copied elsewhere).
    latest_time = latest["time"]
    if latest_time.tzinfo is not None:
        latest_time = latest_time.tz_localize(None)

    return {"aqi": float(latest["us_aqi"]), "time": latest_time}


@st.cache_resource(show_spinner=False)
def load_model(horizon_name: str):
    """Downloads and loads the best registered model for a horizon
    (matches how train_models.py's register_model() saved it: joblib
    dump to model.pkl inside the model's Hopsworks artifact directory)."""
    project = get_project()
    mr = project.get_model_registry()
    model_meta = mr.get_best_model(
        name=f"aqi_predictor_{horizon_name}", metric="rmse", direction="min"
    )
    model_dir = model_meta.download()
    return joblib.load(os.path.join(model_dir, "model.pkl"))


KARACHI_LAT, KARACHI_LON = 24.8607, 67.0011
KARACHI_TZ = "Asia/Karachi"

DAILY_AGG = {
    "us_aqi": "mean", "pm2_5": "mean", "pm10": "mean",
    "carbon_monoxide": "mean", "nitrogen_dioxide": "mean",
    "sulphur_dioxide": "mean", "ozone": "mean",
    "temperature_2m": "mean", "relative_humidity_2m": "mean",
    "wind_speed_10m": "mean", "wind_direction_10m": "mean",
    "surface_pressure": "mean", "precipitation": "sum",
    "shortwave_radiation": "mean",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_forecast_daily() -> pd.DataFrame:
    """Pulls Open-Meteo's forward-looking weather forecast and aggregates
    it to daily values the same way feature_engineering.py's
    aggregate_daily() does (mean for weather vars, sum for precipitation) —
    this fills in the target-day weather columns that, at training time,
    came from actual historical weather instead."""
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": KARACHI_LAT, "longitude": KARACHI_LON,
            "hourly": ",".join(WEATHER_VARS),
            "forecast_days": 5,
            "timezone": KARACHI_TZ,
        },
        timeout=30,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date

    agg = {var: ("sum" if var == "precipitation" else "mean") for var in WEATHER_VARS}
    daily = df.groupby("date").agg(agg).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def build_live_base_row() -> pd.Series | None:
    """
    Builds the base feature row live, from raw hourly data, instead of
    reading aqi_daily_features (which structurally excludes recent days —
    every row there needs 3 days of FUTURE actual AQI as training targets,
    so its latest row is always ~3-4 days stale by design; correct for
    training, wrong for live inference).

    Uses YESTERDAY as the base day: the most recent FULLY completed
    24-hour day, matching the complete-day averages the models were
    trained on. Today's partial-day average would be an input distribution
    the model never saw during training.
    """
    project = get_project()
    fs = project.get_feature_store()
    raw = fs.get_feature_group(name="aqi_raw_hourly", version=1).read()
    raw["time"] = pd.to_datetime(raw["time"])
    if raw["time"].dt.tz is not None:
        raw["time"] = raw["time"].dt.tz_localize(None)  # already Karachi-local, see load_current()

    raw["date"] = raw["time"].dt.date
    daily = raw.groupby("date").agg(DAILY_AGG).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    today_local = pd.Timestamp.now(tz=KARACHI_TZ).tz_localize(None).normalize()
    daily = daily[daily["date"] < today_local].reset_index(drop=True)

    if len(daily) < 8:  # need 7 prior days for aqi_roll7
        return None

    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["aqi_change_rate"] = daily["us_aqi"].diff()
    daily["aqi_roll3"] = daily["us_aqi"].shift(1).rolling(window=3).mean()
    daily["aqi_roll7"] = daily["us_aqi"].shift(1).rolling(window=7).mean()
    daily["pm25_roll3"] = daily["pm2_5"].shift(1).rolling(window=3).mean()

    return daily.iloc[-1]  # yesterday — fully complete


def build_feature_row_live(base_row: pd.Series, forecast_weather: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target_date = base_row["date"] + pd.Timedelta(days=horizon)
    weather_row = forecast_weather[forecast_weather["date"] == target_date]
    if weather_row.empty:
        raise ValueError(f"No forecast weather available for {target_date.date()}")
    weather_row = weather_row.iloc[0]

    row = {col: base_row[col] for col in BASE_FEATURES}
    for var in WEATHER_VARS:
        row[f"{var}_h{horizon}"] = weather_row[var]

    feature_cols = BASE_FEATURES + [f"{var}_h{horizon}" for var in WEATHER_VARS]
    return pd.DataFrame([row])[feature_cols]


@st.cache_data(ttl=3600, show_spinner="Running 3-day forecast...")
def load_forecast() -> list[dict]:
    """Predicts day1/2/3 AQI from yesterday's completed base features plus
    real Open-Meteo weather forecasts for each target day. Since the base
    day is yesterday, day1 = today, day2 = tomorrow, day3 = the day after."""
    base_row = build_live_base_row()
    if base_row is None:
        st.warning("Not enough recent history yet to generate a forecast (need 7+ days).")
        return [{"day": h, "date": None, "aqi": None, "model": HORIZON_MODELS[h]} for h in (1, 2, 3)]

    forecast_weather = fetch_weather_forecast_daily()

    results = []
    for h, horizon_name in HORIZON_NAMES.items():
        target_date = base_row["date"] + pd.Timedelta(days=h)
        try:
            model = load_model(horizon_name)
            X = build_feature_row_live(base_row, forecast_weather, h)
            predicted_aqi = float(model.predict(X)[0])
        except Exception as e:
            st.warning(f"Could not generate {horizon_name} forecast: {e}")
            predicted_aqi = None
        results.append(
            {
                "day": h,
                "date": target_date,
                "aqi": predicted_aqi,
                "model": HORIZON_MODELS[h],
            }
        )
    return results


def same_week_last_year(hist: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    last_year_start = today - pd.Timedelta(days=365 + 3)
    last_year_end = today - pd.Timedelta(days=365 - 3)
    mask = (hist["date"] >= last_year_start) & (hist["date"] <= last_year_end)
    return hist.loc[mask, ["date", "us_aqi"]]


# ══════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("### System")
        try:
            get_project()
            st.markdown("🟢 Hopsworks connected")
        except Exception:
            st.markdown("🔴 Hopsworks unreachable")
        st.markdown(f"<span class='model-tag'>day1 · {HORIZON_MODELS[1]}</span>", unsafe_allow_html=True)
        st.markdown(f"<span class='model-tag'>day2 · {HORIZON_MODELS[2]}</span>", unsafe_allow_html=True)
        st.markdown(f"<span class='model-tag'>day3 · {HORIZON_MODELS[3]}</span>", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.markdown(
            "<p class='muted'>Karachi AQI Predictor<br>Data Science Internship Project</p>",
            unsafe_allow_html=True,
        )


def render_header(current: dict):
    cat_name, cat_color = categorize(current["aqi"])
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("# Karachi AQI — Field Report")
        st.markdown(
            "<p class='muted'>What today's sky looks like, and what the next three days hold.</p>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<p class='muted' style='text-align:right; margin-top:1.8rem;'>"
            f"Last reading: {current['time'].strftime('%b %d, %H:%M')} PKT</p>",
            unsafe_allow_html=True,
        )

    st.markdown(horizon_skyline(current["aqi"]), unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            f"""
            <div class="card">
                <p class="muted">CURRENT AQI</p>
                <span class="aqi-number" style="font-size:3.2rem;">{current['aqi']:.0f}</span>
                <div><span class="badge" style="background-color:{cat_color};">{cat_name}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="card">
                <p class="muted">TODAY'S READING</p>
                <p style="margin-top:0.4rem;">{ADVISORY[cat_name]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Strongest, most explanatory pollutants per the EDA correlation matrix —
# PM2.5 (r=0.92, dominates the AQI score), CO (r=0.65) and NO2 (r=0.59) as
# combustion markers, PM10 as the coarse-particle counterpart to PM2.5.
POLLUTANT_META = [
    ("pm2_5", "PM2.5", "µg/m³"),
    ("pm10", "PM10", "µg/m³"),
    ("nitrogen_dioxide", "NO\u2082", "µg/m³"),
    ("carbon_monoxide", "CO", "µg/m³"),
]


def render_pollutants(daily_df: pd.DataFrame):
    latest = daily_df.iloc[-1]
    st.markdown("### Key pollutants")
    cols = st.columns(4)
    for col, (key, label, unit) in zip(cols, POLLUTANT_META):
        val = latest.get(key)
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <p class="muted">{label}</p>
                    <span class="aqi-number" style="font-size:1.7rem;">{val:.1f}</span>
                    <p class="muted" style="font-size:0.72rem; margin-top:0.2rem;">{unit}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Wind speed shown separately from the pollutant tiles — it's not a
    # pollutant, it's the dispersion mechanism (r=-0.51 with AQI in the
    # EDA), and it's the physical reason the skyline above clears or fades.
    wind = latest.get("wind_speed_10m")
    if wind is not None:
        note = "typically disperses pollution faster" if wind >= 12 else "limited dispersion — pollutants linger"
        st.markdown(
            f"""
            <p class="muted" style="margin-top:0.6rem;">
                Wind speed: <span style="color:#26241F; font-weight:600;">{wind:.1f} km/h</span> — {note}.
            </p>
            """,
            unsafe_allow_html=True,
        )


def render_forecast(forecast: list[dict]):
    st.markdown("### Next 3 days")
    cols = st.columns(3)
    for col, day in zip(cols, forecast):
        aqi_val = day["aqi"]
        if aqi_val is None:
            with col:
                st.markdown(
                    """
                    <div class="card">
                        <p class="muted">FORECAST PENDING</p>
                        <p class="muted" style="font-size:0.8rem;">Wire in the trained model to populate this card.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            continue
        cat_name, cat_color = categorize(aqi_val)
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <p class="muted">{day['date'].strftime('%A, %b %d').upper()}</p>
                    <span class="aqi-number" style="font-size:2.4rem;">{aqi_val:.0f}</span>
                    <div><span class="badge" style="background-color:{cat_color};">{cat_name}</span></div>
                    <div class="model-tag">predicted by {day['model']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_trend(hist: pd.DataFrame):
    st.markdown("### Historical trend")
    fig = go.Figure()

    for lo, hi, name, color in CATEGORY_SCALE:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.06, line_width=0)

    fig.add_trace(
        go.Scatter(
            x=hist["date"], y=hist["us_aqi"], mode="lines",
            line=dict(color="#BC8A2E", width=1.4), name="Daily AQI",
        )
    )
    fig.add_hline(y=100, line_dash="dash", line_color="#9C3F33", opacity=0.6,
                   annotation_text="Unhealthy threshold", annotation_font_color="#857F6E")

    fig.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#3B392F", family="IBM Plex Sans"),
        margin=dict(l=10, r=10, t=10, b=10), height=380,
        xaxis=dict(gridcolor="#EDE6D3", rangeslider=dict(visible=True), rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#FFFFFF", font=dict(color="#3B392F"),
        )),
        yaxis=dict(gridcolor="#EDE6D3", title="US AQI"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_year_comparison(hist: pd.DataFrame, current: dict):
    st.markdown("### This week, last year")
    today = pd.Timestamp(current["time"].date())
    window = same_week_last_year(hist, today)
    if window.empty:
        st.markdown("<p class='muted'>Not enough history yet for this comparison.</p>", unsafe_allow_html=True)
        return

    last_year_avg = window["us_aqi"].mean()
    delta = current["aqi"] - last_year_avg
    direction = "higher" if delta > 0 else "lower"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"""
            <div class="card">
                <p class="muted">SAME WEEK LAST YEAR (AVG)</p>
                <span class="aqi-number" style="font-size:2.2rem;">{last_year_avg:.0f}</span>
                <p class="muted" style="margin-top:0.4rem;">
                    Today is {abs(delta):.0f} points {direction} than this week's average last year.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=window["date"], y=window["us_aqi"], mode="lines+markers",
                                  line=dict(color="#733650", width=1.6), marker=dict(size=4)))
        fig.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#857F6E", size=10), height=140,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#EDE6D3", showticklabels=True),
            yaxis=dict(gridcolor="#EDE6D3"),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_alerts(current: dict, forecast: list[dict]):
    st.markdown("### Health advisory")
    worst_cat, worst_color = categorize(current["aqi"])
    for day in forecast:
        if day["aqi"] is not None:
            cat, color = categorize(day["aqi"])
            if severity_index(cat) > severity_index(worst_cat):
                worst_cat, worst_color = cat, color

    st.markdown(
        f"""
        <div class="card" style="border-left: 4px solid {worst_color};">
            <span class="badge" style="background-color:{worst_color};">Worst expected: {worst_cat}</span>
            <p style="margin-top:0.8rem;">{ADVISORY[worst_cat]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    inject_css()
    render_sidebar()

    try:
        current = load_current()
        hist = load_historical()
        forecast = load_forecast()
    except Exception as e:
        st.error(f"Could not load data from Hopsworks: {e}")
        st.stop()

    render_header(current)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    render_pollutants(hist)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    render_forecast(forecast)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    render_trend(hist)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    render_year_comparison(hist, current)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    render_alerts(current, forecast)


if __name__ == "__main__":
    main()