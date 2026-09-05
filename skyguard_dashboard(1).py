import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import plotly.express as px
import numpy as np
import requests
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from datetime import datetime

# ============================================================
# SKYGUARD AI - INTELLIGENT AWS ANOMALY DETECTION DASHBOARD
# ============================================================

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🌦️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

# ============================================================
# AUTOMATIC LIVE MONITORING REFRESH
# ============================================================

REFRESH_INTERVAL_MS = 60 * 1000

refresh_count = st_autorefresh(
    interval=REFRESH_INTERVAL_MS,
    key="skyguard_live_refresh"
)

# ============================================================
# REAL-TIME MODEL CONFIGURATION
# ============================================================

# NOTE:
# The currently trained production artifact is the existing 4-feature
# model. Precipitation is kept internally so the saved model continues
# to work. The dashboard intentionally focuses on the three parameters
# named in the SIH problem statement: Temperature, Pressure, Humidity.

MODEL_FILE = BASE_DIR / "skyguard_realtime_model.pth"
SCALER_FILE = BASE_DIR / "skyguard_realtime_scaler.pkl"
THRESHOLD_FILE = BASE_DIR / "skyguard_realtime_threshold.txt"


class SkyGuardAutoencoder(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


@st.cache_resource
def load_realtime_ai():
    model = SkyGuardAutoencoder(input_dim=4)

    state = torch.load(
        MODEL_FILE,
        map_location="cpu"
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()

    scaler = joblib.load(SCALER_FILE)
    threshold = float(THRESHOLD_FILE.read_text().strip())

    return model, scaler, threshold


try:
    realtime_model, realtime_scaler, realtime_threshold = load_realtime_ai()
    model_status_ok = True
except Exception as e:
    realtime_model = None
    realtime_scaler = None
    realtime_threshold = None
    model_status_ok = False
    model_load_error = e

# ============================================================
# LIVE LOCATION DATA
# ============================================================

@st.cache_data

def load_live_locations():
    locations = pd.read_csv(
        BASE_DIR / "skyguard_city_coordinates.csv"
    )

    locations = locations[
        ["state", "city", "lat", "lon"]
    ].copy()

    locations = locations.drop_duplicates(
        subset=["state", "city"]
    )

    locations = locations.dropna(
        subset=["lat", "lon"]
    )

    return locations


# ============================================================
# LIVE WEATHER COLLECTION
# ============================================================

@st.cache_data(ttl=50)
def fetch_all_live_weather(locations):
    try:
        params = {
            "latitude": ",".join(locations["lat"].astype(str)),
            "longitude": ",".join(locations["lon"].astype(str)),
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "surface_pressure,"
                "precipitation"
            ),
            "temperature_unit": "celsius",
            "timezone": "auto"
        }

        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params,
            timeout=30
        )
        response.raise_for_status()

        weather_data = response.json()

        if isinstance(weather_data, dict):
            weather_data = [weather_data]

        results = []

        for i, (_, location) in enumerate(locations.iterrows()):
            if i >= len(weather_data):
                continue

            current = weather_data[i]["current"]

            results.append({
                "state": location["state"],
                "city": location["city"],
                "lat": float(location["lat"]),
                "lon": float(location["lon"]),
                "temperature_C": float(current["temperature_2m"]),
                "humidity_pct": float(current["relative_humidity_2m"]),
                "pressure_hPa": float(current["surface_pressure"]),
                "precip_mm": float(current["precipitation"]),
                "observation_time": current.get(
                    "time",
                    datetime.now().isoformat()
                )
            })

        return pd.DataFrame(results)

    except Exception as e:
        st.error(f"Live weather API request failed: {e}")
        return pd.DataFrame()


# ============================================================
# AI ANALYSIS
# ============================================================

def analyze_live_network(live_weather):
    if not model_status_ok or live_weather.empty:
        return pd.DataFrame()

    results = []

    for _, row in live_weather.iterrows():
        values = [
            row["temperature_C"],
            row["humidity_pct"],
            row["pressure_hPa"],
            row["precip_mm"]
        ]

        if any(pd.isna(v) for v in values):
            results.append({
                "state": row["state"],
                "city": row["city"],
                "lat": row["lat"],
                "lon": row["lon"],
                "temperature_C": row["temperature_C"],
                "humidity_pct": row["humidity_pct"],
                "pressure_hPa": row["pressure_hPa"],
                "precip_mm": row["precip_mm"],
                "anomaly_score": np.nan,
                "threshold": realtime_threshold,
                "is_anomaly": False,
                "risk_level": "DATA_ERROR",
                "dominant_sensor": "Unknown",
                "explanation": "Live observation unavailable."
            })
            continue

        input_values = np.array([[
            float(row["temperature_C"]),
            float(row["humidity_pct"]),
            float(row["pressure_hPa"]),
            float(row["precip_mm"])
        ]], dtype=np.float32)

        scaled_values = realtime_scaler.transform(input_values)

        input_tensor = torch.tensor(
            scaled_values,
            dtype=torch.float32
        )

        with torch.no_grad():
            reconstructed = realtime_model(input_tensor)

        squared_errors = (
            input_tensor - reconstructed
        ) ** 2

        anomaly_score = torch.mean(
            squared_errors,
            dim=1
        ).item()

        is_anomaly = anomaly_score > realtime_threshold

        if anomaly_score <= realtime_threshold:
            risk_level = "NORMAL"
        elif anomaly_score <= realtime_threshold * 1.5:
            risk_level = "WARNING"
        elif anomaly_score <= realtime_threshold * 2.5:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        feature_errors = squared_errors[0].cpu().numpy()

        # The model's four internal features are retained for inference,
        # but only the three SIH parameters are presented as core sensors.
        feature_names = [
            "Temperature",
            "Humidity",
            "Pressure",
            "Precipitation"
        ]

        dominant_index = int(np.argmax(feature_errors))
        dominant_sensor = feature_names[dominant_index]

        if risk_level == "NORMAL":
            explanation = (
                "Observation is consistent with the learned normal "
                "multivariate pattern."
            )
        else:
            explanation = (
                "Multivariate inconsistency detected. "
                f"{dominant_sensor} has the highest reconstruction error."
            )

        results.append({
            "state": row["state"],
            "city": row["city"],
            "lat": row["lat"],
            "lon": row["lon"],
            "temperature_C": row["temperature_C"],
            "humidity_pct": row["humidity_pct"],
            "pressure_hPa": row["pressure_hPa"],
            "precip_mm": row["precip_mm"],
            "anomaly_score": float(anomaly_score),
            "threshold": float(realtime_threshold),
            "is_anomaly": bool(is_anomaly),
            "risk_level": risk_level,
            "dominant_sensor": dominant_sensor,
            "explanation": explanation,
            "observation_time": row["observation_time"]
        })

    return pd.DataFrame(results)


# ============================================================
# HEADER
# ============================================================

st.title("🌦️ SkyGuard AI")
st.subheader(
    "Intelligent Real-Time Anomaly Detection for Automatic Weather Stations"
)
st.markdown(
    "Detect abnormal and inconsistent AWS observations using AI, "
    "with a focus on **Temperature, Atmospheric Pressure, and Relative Humidity**."
)

st.caption(
    f"🟢 Live Monitoring • Automatic refresh every 60 seconds • "
    f"Refresh cycle: {refresh_count}"
)

# ============================================================
# MODEL STATUS
# ============================================================

if model_status_ok:
    st.success(
        f"🤖 SkyGuard AI model loaded • Detection threshold: "
        f"{realtime_threshold:.6f}"
    )
else:
    st.error(f"AI model could not be loaded: {model_load_error}")
    st.stop()

# ============================================================
# LOAD + FETCH LIVE DATA
# ============================================================

live_locations = load_live_locations()

with st.spinner(
    f"Fetching live weather data for {len(live_locations)} monitored locations..."
):
    live_weather = fetch_all_live_weather(live_locations)

if live_weather.empty:
    st.warning("No live weather data was received.")
    st.stop()

# ============================================================
# LIVE UPDATE STATUS
# ============================================================

latest_update = datetime.now().strftime(
    "%d %b %Y • %I:%M:%S %p"
)

st.success(
    f"🟢 LIVE DATA UPDATED • {latest_update} • "
    f"{len(live_weather)}/{len(live_locations)} locations received data"
)

# ============================================================
# RUN AI
# ============================================================

with st.spinner("Running SkyGuard AI anomaly detection..."):
    live_ai_results = analyze_live_network(live_weather)

if live_ai_results.empty:
    st.warning("AI analysis did not produce any results.")
    st.stop()

valid_results = live_ai_results[
    live_ai_results["anomaly_score"].notna()
].copy()

# ============================================================
# NETWORK STATUS
# ============================================================

st.header("📡 Live AWS Network Status")

# Only three SIH parameters are highlighted in the dashboard.

station_count = len(valid_results)
anomaly_count = int(valid_results["is_anomaly"].sum())
normal_count = station_count - anomaly_count
warning_count = int((valid_results["risk_level"] == "WARNING").sum())
high_count = int((valid_results["risk_level"] == "HIGH").sum())
critical_count = int((valid_results["risk_level"] == "CRITICAL").sum())

live_anomaly_rate = (
    anomaly_count / station_count * 100
    if station_count > 0 else 0.0
)

network_health = max(
    0,
    min(100, 100 - live_anomaly_rate * 5)
)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Stations Monitored", station_count)

with c2:
    st.metric("Normal", normal_count)

with c3:
    st.metric("AI Anomalies", anomaly_count)

with c4:
    st.metric("Critical", critical_count)

with c5:
    st.metric("Network Health", f"{network_health:.1f}%")

# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader("🚦 Anomaly Severity")

risk_counts = (
    valid_results["risk_level"]
    .value_counts()
    .reindex(
        ["NORMAL", "WARNING", "HIGH", "CRITICAL"],
        fill_value=0
    )
    .reset_index()
)

risk_counts.columns = ["Risk Level", "Stations"]

fig_risk = px.bar(
    risk_counts,
    x="Risk Level",
    y="Stations",
    title="Live AI Risk Distribution"
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)

# ============================================================
# AI ALERTS
# ============================================================

st.header("🚨 Real-Time AI Alerts")

anomalies = valid_results[
    valid_results["is_anomaly"] == True
].copy()

if anomalies.empty:
    st.success(
        "🟢 No AI anomalies detected in the current observation cycle."
    )
else:
    anomalies = anomalies.sort_values(
        "anomaly_score",
        ascending=False
    )

    top_alert = anomalies.iloc[0]

    if top_alert["risk_level"] == "CRITICAL":
        st.error(
            f"🔴 CRITICAL • {top_alert['city']}, {top_alert['state']} • "
            f"Anomaly score: {top_alert['anomaly_score']:.5f}"
        )
    elif top_alert["risk_level"] == "HIGH":
        st.warning(
            f"🟠 HIGH • {top_alert['city']}, {top_alert['state']} • "
            f"Anomaly score: {top_alert['anomaly_score']:.5f}"
        )
    else:
        st.info(
            f"🟡 WARNING • {top_alert['city']}, {top_alert['state']} • "
            f"Anomaly score: {top_alert['anomaly_score']:.5f}"
        )

    st.caption(
        "The AI score indicates how inconsistent the observation is with "
        "the learned multivariate pattern. A high score is not by itself proof "
        "of physical sensor failure."
    )

    alert_columns = [
        "state",
        "city",
        "temperature_C",
        "humidity_pct",
        "pressure_hPa",
        "anomaly_score",
        "risk_level",
        "dominant_sensor",
        "explanation"
    ]

    st.dataframe(
        anomalies[alert_columns],
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# NETWORK MAP
# ============================================================

st.header("🗺️ Live AWS Network Map")
st.caption(
    "Spatial view of the current AI assessment across the monitored network."
)

map_data = valid_results.dropna(
    subset=["lat", "lon"]
).copy()

if not map_data.empty:
    fig_map = px.scatter_map(
        map_data,
        lat="lat",
        lon="lon",
        color="risk_level",
        hover_name="city",
        hover_data={
            "state": True,
            "temperature_C": ":.1f",
            "humidity_pct": ":.1f",
            "pressure_hPa": ":.1f",
            "anomaly_score": ":.5f",
            "risk_level": True,
            "dominant_sensor": True,
            "lat": False,
            "lon": False
        },
        zoom=4,
        height=620,
        title="SkyGuard AI - Live Observation Trust Map"
    )

    fig_map.update_layout(map_style="open-street-map")

    st.plotly_chart(
        fig_map,
        use_container_width=True
    )

# ============================================================
# STATION INVESTIGATION
# ============================================================

st.header("🔎 Station Investigation")

location_options = (
    valid_results["state"].astype(str)
    + " — "
    + valid_results["city"].astype(str)
).tolist()

selected_live_location = st.selectbox(
    "Select a monitored location",
    location_options,
    key="live_station_selector"
)

selected_mask = (
    valid_results["state"].astype(str)
    + " — "
    + valid_results["city"].astype(str)
) == selected_live_location

selected_station = valid_results[
    selected_mask
].iloc[0]

st.subheader(
    f"📍 {selected_station['city']}, {selected_station['state']}"
)

# Only SIH-defined parameters are shown as the station's core readings.
s1, s2, s3 = st.columns(3)

with s1:
    st.metric(
        "Temperature",
        f"{selected_station['temperature_C']:.1f} °C"
    )

with s2:
    st.metric(
        "Relative Humidity",
        f"{selected_station['humidity_pct']:.1f} %"
    )

with s3:
    st.metric(
        "Atmospheric Pressure",
        f"{selected_station['pressure_hPa']:.1f} hPa"
    )

# ============================================================
# AI ASSESSMENT + EXPLAINABILITY
# ============================================================

st.subheader("🧠 AI Assessment")

a1, a2, a3 = st.columns(3)

with a1:
    st.metric(
        "Anomaly Score",
        f"{selected_station['anomaly_score']:.5f}"
    )

with a2:
    st.metric(
        "Detection Threshold",
        f"{selected_station['threshold']:.5f}"
    )

with a3:
    st.metric(
        "Risk Level",
        selected_station["risk_level"]
    )

risk = selected_station["risk_level"]
explanation = str(selected_station["explanation"])

if risk == "NORMAL":
    st.success("🟢 " + explanation)
elif risk == "WARNING":
    st.warning("🟡 " + explanation)
elif risk == "HIGH":
    st.warning("🟠 " + explanation)
elif risk == "CRITICAL":
    st.error("🔴 " + explanation)
else:
    st.info(explanation)

if risk != "NORMAL":
    st.info(
        "🎯 Dominant parameter contribution: "
        + str(selected_station["dominant_sensor"])
    )

st.caption(
    "Dominant parameter identifies the feature contributing most to the "
    "autoencoder reconstruction error; it is a diagnostic clue, not a confirmed fault."
)

st.caption(
    "🕐 Observation time: "
    + str(selected_station.get("observation_time", "Unavailable"))
)

# ============================================================
# DATA QUALITY / COMMUNICATION STATUS
# ============================================================

st.header("🔌 Data Quality Status")

missing_mask = live_weather[
    [
        "temperature_C",
        "humidity_pct",
        "pressure_hPa"
    ]
].isna().any(axis=1)

missing_count = int(missing_mask.sum())
received_count = len(live_weather)

q1, q2, q3 = st.columns(3)

with q1:
    st.metric("Locations Responding", received_count)

with q2:
    st.metric("Incomplete Core Readings", missing_count)

with q3:
    st.metric(
        "Current Anomaly Rate",
        f"{live_anomaly_rate:.2f}%"
    )

if missing_count == 0:
    st.success(
        "🟢 Temperature, humidity and pressure values are available "
        "for all locations in the current response."
    )
else:
    st.warning(
        f"⚠️ {missing_count} locations have incomplete core observations. "
        "Communication/data-quality investigation is recommended."
    )

# ============================================================
# SCOPE NOTE
# ============================================================

st.divider()
st.caption(
    "SkyGuard AI is an AWS observation-quality and anomaly-detection layer. "
    "The live feed shown here is a reference/simulated weather feed; it is not "
    "being represented as a physical AWS sensor feed."
)

st.caption(
    "Core SIH parameters: Temperature (°C) • Atmospheric Pressure (hPa) • "
    "Relative Humidity (%)"
)
