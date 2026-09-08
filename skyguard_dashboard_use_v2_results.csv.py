
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
# SKYGUARD V2 - AI WEATHER STATION ANOMALY DASHBOARD
# ============================================================

st.set_page_config(
    page_title="SkyGuard V2",
    page_icon="🌦️",
    layout="wide"
)

# ============================================================
# STEP 18A - AUTOMATIC LIVE MONITORING REFRESH
# ============================================================

REFRESH_INTERVAL_MS = 300 * 1000  # 5 minutes

refresh_count = st_autorefresh(
    interval=REFRESH_INTERVAL_MS,
    key="skyguard_live_refresh"
)

st.caption(
    f"🟢 Live Monitoring • "
    f"Automatic refresh: every  5 minutes• "
    f"Refresh cycle: {refresh_count}"
)


BASE_DIR = Path(__file__).parent

# ============================================================
# STEP 15 - REAL-TIME SKYGUARD MODEL CONFIGURATION
# ============================================================

LIVE_FEATURES = [
    "temperature_C",
    "humidity_pct",
    "pressure_hPa"
]

MODEL_FILE = BASE_DIR / "skyguard_realtime_model.pth"
SCALER_FILE = BASE_DIR / "skyguard_realtime_scaler.pkl"
THRESHOLD_FILE = BASE_DIR / "skyguard_realtime_threshold.txt"


# ============================================================
# SKYGUARD 3-FEATURE AUTOENCODER
# ============================================================

class SkyGuardAutoencoder(nn.Module):

    def __init__(self, input_dim=3):

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


# ============================================================
# LOAD REAL-TIME AI MODEL
# ============================================================

@st.cache_resource
def load_realtime_ai():

    model = SkyGuardAutoencoder(input_dim=3)

    state = torch.load(
        MODEL_FILE,
        map_location="cpu"
    )

    # Support both raw state_dict and checkpoint format
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)

    model.eval()

    scaler = joblib.load(
        SCALER_FILE
    )

    threshold = float(
        THRESHOLD_FILE.read_text().strip()
    )

    return model, scaler, threshold


# ============================================================
# INITIALIZE REAL-TIME AI
# ============================================================

realtime_model, realtime_scaler, realtime_threshold = load_realtime_ai()

# ============================================================
# STEP 16 - REAL-TIME 3-FEATURE AI INFERENCE
# ============================================================

def analyze_live_reading(
    temperature,
    humidity,
    pressure
):

    values = np.array([[
        float(temperature),
        float(humidity),
        float(pressure)
    ]], dtype=np.float32)

    # Apply the SAME scaler used during model training
    scaled_values = realtime_scaler.transform(values)

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(
        scaled_values,
        dtype=torch.float32
    )

    # AI inference
    with torch.no_grad():

        reconstructed = realtime_model(
            input_tensor
        )

    # Reconstruction error
    error = torch.mean(
        (input_tensor - reconstructed) ** 2,
        dim=1
    ).item()

    # Anomaly decision
    is_anomaly = error > realtime_threshold

    # Risk classification
    if error <= realtime_threshold:

        risk_level = "NORMAL"

    elif error <= realtime_threshold * 1.5:

        risk_level = "WARNING"

    elif error <= realtime_threshold * 2.5:

        risk_level = "HIGH"

    else:

        risk_level = "CRITICAL"

    return {
        "temperature_C": float(temperature),
        "humidity_pct": float(humidity),
        "pressure_hPa": float(pressure),
        "anomaly_score": float(error),
        "threshold": float(realtime_threshold),
        "is_anomaly": bool(is_anomaly),
        "risk_level": risk_level
    }

# ============================================================
# STEP 17A - LOAD ALL CITY LOCATIONS FOR LIVE MONITORING
# ============================================================

@st.cache_data
def load_live_locations():

    locations_file = BASE_DIR / "skyguard_city_coordinates.csv"

    locations = pd.read_csv(locations_file)

    # Keep only required columns
    locations = locations[
        ["state", "city", "lat", "lon"]
    ].copy()

    # Remove duplicate locations
    locations = locations.drop_duplicates(
        subset=["state", "city"]
    )

    # Remove invalid coordinates
    locations = locations.dropna(
        subset=["lat", "lon"]
    )

    return locations


live_locations = load_live_locations()

# ============================================================
# GLOBAL SKYGUARD LOCATION SEARCH
# ============================================================

st.sidebar.title("🔎 SkyGuard Search")

# Google-style searchable location selector.
# Start typing a city or state and Streamlit automatically narrows
# the available suggestions.
location_options = ["🌐 All Locations"] + sorted(
    live_locations.apply(
        lambda row: f"{row['city']}, {row['state']}",
        axis=1
    ).drop_duplicates().tolist()
)

selected_location = st.sidebar.selectbox(
    "Search Location",
    location_options,
    index=0,
    key="global_location_search",
    help="Start typing a city or state to see matching suggestions."
)

# Filter the entire dashboard to the selected suggestion.
live_filter = live_locations.copy()
search_text = ""
selected_city = None
selected_state = None

if selected_location != "🌐 All Locations":
    selected_city, selected_state = [
        part.strip()
        for part in selected_location.split(",", 1)
    ]

    search_text = selected_location

    live_filter = live_filter[
        (
            live_filter["city"].astype(str).str.casefold()
            == selected_city.casefold()
        )
        & (
            live_filter["state"].astype(str).str.casefold()
            == selected_state.casefold()
        )
    ]

st.write(
    f"Live monitoring locations: {len(live_filter)} / {len(live_locations)}"
)


# ============================================================
# STEP 17B - SINGLE-BATCH LIVE WEATHER COLLECTION
# ============================================================

def fetch_all_live_weather(locations):

    try:

        params = {
            "latitude": ",".join(
                locations["lat"].astype(str)
            ),

            "longitude": ",".join(
                locations["lon"].astype(str)
            ),

            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "surface_pressure,"
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

        for i, (_, location) in enumerate(
            locations.iterrows()
        ):

            if i >= len(weather_data):
                continue

            current = weather_data[i]["current"]

            results.append({

                "state": location["state"],
                "city": location["city"],

                "lat": float(location["lat"]),
                "lon": float(location["lon"]),

                "temperature_C": float(
                    current["temperature_2m"]
                ),

                "humidity_pct": float(
                    current["relative_humidity_2m"]
                ),

                "pressure_hPa": float(
                    current["surface_pressure"]
                ),

                "observation_time": current.get(
                    "time",
                    datetime.now().isoformat()
                )
            })

        return pd.DataFrame(results)

    except Exception as e:

        st.error(
            f"Live weather API request failed: {e}"
        )

        return pd.DataFrame()


# ============================================================
# TEST
# ============================================================

with st.spinner(
    f"Fetching live weather data for {len(live_filter)} selected locations..."
):

    live_weather = fetch_all_live_weather(
        live_filter
    )

# ============================================================
# STEP 18B - LIVE UPDATE STATUS
# ============================================================

if not live_weather.empty:

    current_update_time = datetime.now().strftime(
        "%d %b %Y • %I:%M:%S %p"
    )

    st.success(
        f"🟢 LIVE DATA UPDATED • {current_update_time}"
    )

    st.dataframe(
        live_weather[
            [
                "state",
                "city",
                "temperature_C",
                "humidity_pct",
                "pressure_hPa"
            ]
        ],
        use_container_width=True
    )

else:

    st.warning(
        "No live weather data was received."
    )

# ============================================================
# STEP 17C - REAL-TIME SKYGUARD AI ANALYSIS
# ============================================================

# ============================================================
# FROZEN-VALUE DETECTION MEMORY
# ============================================================

if "frozen_history" not in st.session_state:
    st.session_state.frozen_history = {}

FROZEN_REPEAT_LIMIT = 4  # Require 4 distinct API observations before declaring a frozen sensor

# ============================================================
# STEP 17A - TEMPORAL CONSISTENCY MEMORY
# ============================================================

if "temporal_history" not in st.session_state:
    st.session_state.temporal_history = {}

TEMPORAL_CHANGE_LIMITS = {
    "temperature_C": 3.0,
    "humidity_pct": 15.0,
    "pressure_hPa": 5.0
}

TEMPORAL_HISTORY_LIMIT = 8

DRIFT_CHANGE_LIMITS = {
    "temperature_C": 0.75,
    "humidity_pct": 4.0,
    "pressure_hPa": 1.5
}

MIN_TREND_HISTORY = 4

PHYSICAL_LIMITS = {
    "temperature_C": (-90.0, 60.0),
    "humidity_pct": (0.0, 100.0),
    "pressure_hPa": (850.0, 1100.0)
}

SENSOR_NAMES = {
    "temperature_C": "Temperature",
    "humidity_pct": "Humidity",
    "pressure_hPa": "Pressure"
}

if "maintenance_history" not in st.session_state:
    st.session_state.maintenance_history = {}

MAINTENANCE_HISTORY_LIMIT = 20

# Small tolerance to handle tiny floating-point changes
FROZEN_TOLERANCES = {
    "temperature_C": 0.01,
    "humidity_pct": 0.01,
    "pressure_hPa": 0.01
}

# ============================================================
# STEP 17C - SPATIAL CONSISTENCY INTELLIGENCE
# ============================================================

def calculate_spatial_context(live_weather, current_row):
    """
    Compare the current station with nearby live stations.

    This is contextual evidence only.
    It does NOT modify the AI anomaly score or risk level.
    """

    current_lat = current_row["lat"]
    current_lon = current_row["lon"]

    current_temperature = current_row["temperature_C"]
    current_humidity = current_row["humidity_pct"]
    current_pressure = current_row["pressure_hPa"]

    nearby_stations = []

    for _, other_row in live_weather.iterrows():

        if (
            str(other_row["state"]) == str(current_row["state"])
            and str(other_row["city"]) == str(current_row["city"])
        ):
            continue

        if (
            other_row["temperature_C"] is None
            or other_row["humidity_pct"] is None
            or other_row["pressure_hPa"] is None
        ):
            continue

        # Approximate geographic distance
        lat_diff = float(other_row["lat"]) - float(current_lat)
        lon_diff = float(other_row["lon"]) - float(current_lon)

        distance_score = (
            lat_diff ** 2
            + lon_diff ** 2
        ) ** 0.5

        nearby_stations.append({
            "state": other_row["state"],
            "city": other_row["city"],
            "temperature_C": other_row["temperature_C"],
            "humidity_pct": other_row["humidity_pct"],
            "pressure_hPa": other_row["pressure_hPa"],
            "distance": distance_score
        })

    # Keep the closest stations
    nearby_stations = sorted(
        nearby_stations,
        key=lambda x: x["distance"]
    )[:5]

    if not nearby_stations:
        return {
            "status": "INSUFFICIENT_NEIGHBORS",
            "neighbor_count": 0,
            "similar_temperature": None,
            "similar_humidity": None,
            "similar_pressure": None
        }

    temperature_matches = 0
    humidity_matches = 0
    pressure_matches = 0

    for neighbor in nearby_stations:

        if abs(
            float(current_temperature)
            - float(neighbor["temperature_C"])
        ) <= 3.0:
            temperature_matches += 1

        if abs(
            float(current_humidity)
            - float(neighbor["humidity_pct"])
        ) <= 15.0:
            humidity_matches += 1

        if abs(
            float(current_pressure)
            - float(neighbor["pressure_hPa"])
        ) <= 5.0:
            pressure_matches += 1

    neighbor_count = len(nearby_stations)

    similar_temperature = (
        temperature_matches / neighbor_count
    )

    similar_humidity = (
        humidity_matches / neighbor_count
    )

    similar_pressure = (
        pressure_matches / neighbor_count
    )

    overall_similarity = (
        similar_temperature
        + similar_humidity
        + similar_pressure
    ) / 3.0

    if overall_similarity >= 0.67:
        status = "CONSISTENT_WITH_NEIGHBORS"

    elif overall_similarity >= 0.34:
        status = "PARTIALLY_CONSISTENT"

    else:
        status = "ISOLATED_CHANGE"

    return {
        "status": status,
        "neighbor_count": neighbor_count,
        "similar_temperature": similar_temperature,
        "similar_humidity": similar_humidity,
        "similar_pressure": similar_pressure
    }

def calculate_temporal_intelligence(current_values, history):
    """Priority 4: multi-observation temporal intelligence."""
    result = {
        "temporal_score": 0.0,
        "unusual_sensors": [],
        "spike_sensors": [],
        "drop_sensors": [],
        "drift_sensors": [],
        "changes": {},
        "trend_direction": {},
        "trend_strength": {}
    }

    valid_history = [
        h for h in history
        if all(h.get(sensor) is not None for sensor in current_values)
    ]
    if not valid_history:
        return result

    previous = valid_history[-1]
    scores = []

    for sensor, current in current_values.items():
        previous_value = previous.get(sensor)
        if current is None or previous_value is None:
            continue

        change = float(current - previous_value)
        result["changes"][sensor] = change
        limit = TEMPORAL_CHANGE_LIMITS[sensor]
        magnitude = abs(change) / max(limit, 1e-9)

        if change > limit:
            result["spike_sensors"].append(sensor)
            result["unusual_sensors"].append(sensor)
            scores.append(min(1.0, magnitude))
        elif change < -limit:
            result["drop_sensors"].append(sensor)
            result["unusual_sensors"].append(sensor)
            scores.append(min(1.0, magnitude))
        else:
            scores.append(min(1.0, magnitude * 0.35))

    if len(valid_history) >= MIN_TREND_HISTORY:
        recent = valid_history[-(MIN_TREND_HISTORY - 1):]
        for sensor in current_values:
            values = [h[sensor] for h in recent if h.get(sensor) is not None]
            values.append(current_values[sensor])
            if len(values) < 3:
                continue
            diffs = np.diff(np.asarray(values, dtype=float))
            avg_change = float(np.mean(diffs))
            same_direction = bool(
                np.all(diffs >= 0) or np.all(diffs <= 0)
            )
            direction = (
                "INCREASING" if avg_change > 0 else
                "DECREASING" if avg_change < 0 else "STABLE"
            )
            strength = min(
                1.0,
                abs(avg_change) / max(DRIFT_CHANGE_LIMITS[sensor], 1e-9)
            )
            result["trend_direction"][sensor] = direction
            result["trend_strength"][sensor] = float(strength)

            # Drift requires persistence, not just one moderate jump.
            if same_direction and abs(avg_change) >= DRIFT_CHANGE_LIMITS[sensor]:
                result["drift_sensors"].append(sensor)
                if sensor not in result["unusual_sensors"]:
                    result["unusual_sensors"].append(sensor)
                scores.append(strength)

    if scores:
        result["temporal_score"] = float(min(1.0, np.mean(scores)))

    return result


def calculate_multivariate_intelligence(current_values, previous_values, temporal_info):
    """Priority 4: determine whether sensor changes are coordinated."""
    result = {
        "multivariate_score": 0.0,
        "status": "NORMAL",
        "inconsistent_sensors": [],
        "coordinated_change": False,
        "reason": ""
    }

    if not previous_values:
        result["reason"] = "No previous observation available."
        return result

    changes = {}
    for sensor, current in current_values.items():
        previous = previous_values.get(sensor)
        if current is not None and previous is not None:
            changes[sensor] = float(current - previous)

    significant = [
        sensor for sensor, change in changes.items()
        if abs(change) > TEMPORAL_CHANGE_LIMITS[sensor]
    ]

    if not significant:
        result["reason"] = (
            "Sensor variables remain within expected temporal relationships."
        )
        return result

    if len(significant) == 1:
        sensor = significant[0]
        result.update({
            "status": "SINGLE_SENSOR_INCONSISTENCY",
            "inconsistent_sensors": [sensor],
            "multivariate_score": 0.85,
            "reason": (
                f"{SENSOR_NAMES[sensor]} changed significantly while the "
                "other sensor variables remained comparatively stable."
            )
        })
        return result

    result["coordinated_change"] = True
    directions = ["UP" if changes[s] > 0 else "DOWN" for s in significant]

    if len(set(directions)) == 1:
        result.update({
            "status": "COORDINATED_MULTIVARIATE_CHANGE",
            "multivariate_score": 0.35,
            "reason": (
                "Multiple sensor variables changed in a coordinated direction, "
                "which is more consistent with an environmental event."
            )
        })
    else:
        result.update({
            "status": "MULTIVARIATE_INCONSISTENCY",
            "multivariate_score": 0.75,
            "reason": (
                "Multiple sensor variables changed in inconsistent directions, "
                "suggesting a possible multivariate sensor or data anomaly."
            )
        })

    result["inconsistent_sensors"] = significant
    return result


def classify_anomaly_type(current_values, temporal_info, multivariate_info,
                          frozen_sensors, ai_anomaly):
    """Priority 3: explicit anomaly type classification."""
    missing = [s for s, v in current_values.items() if v is None]
    if missing:
        return (
            "MISSING_DATA",
            "Missing sensor data: " + ", ".join(SENSOR_NAMES[s] for s in missing) + "."
        )

    impossible = []
    for sensor, value in current_values.items():
        low, high = PHYSICAL_LIMITS[sensor]
        if value < low or value > high:
            impossible.append(sensor)
    if impossible:
        return (
            "PHYSICALLY_IMPOSSIBLE_VALUE",
            "Value outside physical operating range: " +
            ", ".join(SENSOR_NAMES[s] for s in impossible) + "."
        )

    if frozen_sensors:
        return (
            "FROZEN_SENSOR",
            "Possible frozen sensor value detected for " +
            ", ".join(SENSOR_NAMES[s] for s in frozen_sensors) + "."
        )

    if temporal_info["spike_sensors"]:
        return (
            "SUDDEN_SPIKE",
            "Rapid upward change detected in " +
            ", ".join(SENSOR_NAMES[s] for s in temporal_info["spike_sensors"]) + "."
        )

    if temporal_info["drop_sensors"]:
        return (
            "SUDDEN_DROP",
            "Rapid downward change detected in " +
            ", ".join(SENSOR_NAMES[s] for s in temporal_info["drop_sensors"]) + "."
        )

    if temporal_info["drift_sensors"]:
        return (
            "SENSOR_DRIFT",
            "Persistent gradual change detected in " +
            ", ".join(SENSOR_NAMES[s] for s in temporal_info["drift_sensors"]) + "."
        )

    if multivariate_info["status"] == "MULTIVARIATE_INCONSISTENCY":
        return "MULTIVARIATE_INCONSISTENCY", multivariate_info["reason"]

    if multivariate_info["status"] == "SINGLE_SENSOR_INCONSISTENCY":
        return "SINGLE_SENSOR_INCONSISTENCY", multivariate_info["reason"]

    if ai_anomaly:
        return (
            "AI_PATTERN_ANOMALY",
            "The autoencoder identified a multivariate pattern that differs "
            "from learned normal behavior."
        )

    return (
        "NORMAL_OPERATION",
        "Sensor behavior is consistent with expected temporal and multivariate patterns."
    )


def classify_weather_vs_sensor(anomaly_type, temporal_info, multivariate_info,
                               spatial_context, ai_anomaly):
    """Priority 5: distinguish a likely weather event from a sensor fault."""
    if anomaly_type == "NORMAL_OPERATION":
        return "NORMAL", "No anomaly evidence requiring weather/fault classification."

    if anomaly_type in {"MISSING_DATA", "PHYSICALLY_IMPOSSIBLE_VALUE", "FROZEN_SENSOR"}:
        return "LIKELY_SENSOR_FAULT", "The anomaly is directly associated with sensor/data integrity."

    spatial_status = spatial_context.get("status", "INSUFFICIENT_NEIGHBORS")
    neighbor_count = spatial_context.get("neighbor_count", 0)
    similarity = np.mean([
        x for x in [
            spatial_context.get("similar_temperature"),
            spatial_context.get("similar_humidity"),
            spatial_context.get("similar_pressure")
        ] if x is not None
    ]) if neighbor_count else None

    if multivariate_info["status"] == "SINGLE_SENSOR_INCONSISTENCY":
        return "LIKELY_SENSOR_FAULT", "Only one variable changed strongly relative to the previous observation."

    if multivariate_info["status"] == "MULTIVARIATE_INCONSISTENCY":
        return "LIKELY_SENSOR_FAULT", "Multiple variables changed in inconsistent directions."

    if multivariate_info["status"] == "COORDINATED_MULTIVARIATE_CHANGE":
        if spatial_status == "CONSISTENT_WITH_NEIGHBORS" or (similarity is not None and similarity >= 0.67):
            return "LIKELY_WEATHER_EVENT", "Multiple sensors changed together and neighboring stations show similar conditions."
        if spatial_status == "PARTIALLY_CONSISTENT" or (similarity is not None and similarity >= 0.34):
            return "POSSIBLE_WEATHER_EVENT", "Multiple sensors changed together with partial spatial support."
        return "POSSIBLE_SENSOR_FAULT", "A coordinated local change lacks sufficient neighboring-station support."

    if anomaly_type in {"SUDDEN_SPIKE", "SUDDEN_DROP", "SENSOR_DRIFT"}:
        if spatial_status == "ISOLATED_CHANGE" or (similarity is not None and similarity < 0.34):
            return "LIKELY_SENSOR_FAULT", "The temporal anomaly is isolated from nearby stations."
        if anomaly_type in {"SUDDEN_SPIKE", "SUDDEN_DROP"} and multivariate_info["coordinated_change"]:
            return "POSSIBLE_WEATHER_EVENT", "The rapid change affects multiple variables and has spatial support."
        return "POSSIBLE_SENSOR_FAULT", "Temporal anomaly detected; additional observations are recommended."

    if ai_anomaly and spatial_status == "CONSISTENT_WITH_NEIGHBORS":
        return "POSSIBLE_WEATHER_EVENT", "AI anomaly is supported by neighboring station conditions."

    return "POSSIBLE_SENSOR_FAULT", "Anomaly evidence is present but insufficient to confirm a weather event."


def classify_root_cause(anomaly_type, weather_classification, temporal_info,
                        multivariate_info, spatial_context, frozen_sensors):
    """Priority 5: human-readable root-cause category."""
    if anomaly_type == "NORMAL_OPERATION":
        return "NORMAL_OPERATION"
    if anomaly_type == "MISSING_DATA":
        return "COMMUNICATION_OR_DATA_INTERRUPTION"
    if anomaly_type == "PHYSICALLY_IMPOSSIBLE_VALUE":
        return "INVALID_SENSOR_READING"
    if anomaly_type == "FROZEN_SENSOR":
        return "FROZEN_SENSOR"
    if weather_classification == "LIKELY_WEATHER_EVENT":
        return "GENUINE_WEATHER_EVENT"
    if anomaly_type == "SENSOR_DRIFT":
        return "SENSOR_DEGRADATION_OR_CALIBRATION_DRIFT"
    if multivariate_info["status"] == "SINGLE_SENSOR_INCONSISTENCY":
        return SENSOR_NAMES[multivariate_info["inconsistent_sensors"][0]] + "_SENSOR_FAULT"
    if spatial_context.get("status") == "ISOLATED_CHANGE":
        return "LOCAL_SENSOR_OR_DATA_FAULT"
    if anomaly_type == "MULTIVARIATE_INCONSISTENCY":
        return "MULTIVARIATE_SENSOR_ANOMALY"
    return "UNCONFIRMED_ANOMALY"


def update_maintenance_health(station_key, anomaly_type, is_anomaly,
                              temporal_score, multivariate_score,
                              frozen_value_detected, decision_confidence):
    """Priority 10: rolling degradation forecasting and maintenance risk."""
    history = st.session_state.maintenance_history.setdefault(station_key, [])
    history.append({
        "anomaly": int(bool(is_anomaly)),
        "type": anomaly_type,
        "temporal": float(temporal_score),
        "multivariate": float(multivariate_score),
        "frozen": int(bool(frozen_value_detected)),
        "confidence": float(decision_confidence)
    })
    history = history[-MAINTENANCE_HISTORY_LIMIT:]
    st.session_state.maintenance_history[station_key] = history

    fault_types = {
        "SUDDEN_SPIKE", "SUDDEN_DROP", "FROZEN_SENSOR",
        "SENSOR_DRIFT", "PHYSICALLY_IMPOSSIBLE_VALUE",
        "SINGLE_SENSOR_INCONSISTENCY", "MULTIVARIATE_INCONSISTENCY",
        "MISSING_DATA"
    }

    # Reconstruct a health trajectory for every observation.
    health_series = []
    for item in history:
        anomaly_rate = float(item["anomaly"])
        fault_rate = 1.0 if item["type"] in fault_types else 0.0
        frozen_rate = float(item["frozen"])
        degradation = (
            0.40 * anomaly_rate +
            0.20 * fault_rate +
            0.15 * frozen_rate +
            0.15 * float(item["temporal"]) +
            0.10 * float(item["multivariate"])
        ) * 100.0
        health_series.append(max(0.0, min(100.0, 100.0 - degradation)))

    n = len(history)
    health = float(health_series[-1]) if health_series else 100.0

    # Forecast degradation from the recent health trajectory.
    trend_window = min(8, n)
    recent_health = np.asarray(health_series[-trend_window:], dtype=float)
    if trend_window >= 3:
        x = np.arange(trend_window, dtype=float)
        slope = float(np.polyfit(x, recent_health, 1)[0])
    else:
        slope = 0.0

    forecast_horizon = 5
    forecast_health = float(np.clip(health + slope * forecast_horizon, 0.0, 100.0))

    # Probability that maintenance will be needed within the next five
    # observations. This is a calibrated risk heuristic until labeled
    # maintenance/failure history is available for supervised training.
    risk_signal = (60.0 - forecast_health) / 8.0 + max(0.0, -slope) * 1.5
    maintenance_risk = float(1.0 / (1.0 + np.exp(-risk_signal)))

    if n < 4:
        status = "MONITORING"
    elif maintenance_risk >= 0.80 or forecast_health < 40:
        status = "MAINTENANCE_URGENT"
    elif maintenance_risk >= 0.55 or forecast_health < 60:
        status = "MAINTENANCE_RECOMMENDED"
    elif maintenance_risk >= 0.30 or forecast_health < 80:
        status = "WATCH"
    else:
        status = "HEALTHY"

    drift_count = sum(x["type"] == "SENSOR_DRIFT" for x in history)
    frozen_rate = sum(x["frozen"] for x in history) / max(n, 1)
    fault_rate = sum(x["type"] in fault_types for x in history) / max(n, 1)

    if slope <= -3.0:
        reason = (
            f"Health is degrading at {abs(slope):.2f} points per observation; "
            f"the model forecasts {forecast_health:.1f}/100 within the next "
            f"{forecast_horizon} observations."
        )
    elif drift_count >= 3:
        reason = "Repeated sensor-drift events are driving the predicted maintenance risk."
    elif frozen_rate >= 0.25:
        reason = "Repeated frozen-value behavior is increasing predicted maintenance risk."
    elif fault_rate >= 0.35:
        reason = "Repeated sensor/data faults are increasing predicted maintenance risk."
    else:
        reason = (
            f"Recent degradation trend is stable; projected health is "
            f"{forecast_health:.1f}/100 over the next {forecast_horizon} observations."
        )

    return (
        float(health),
        status,
        reason,
        maintenance_risk * 100.0,
        forecast_health,
        slope
    )


def analyze_live_network(live_weather):
    results = []

    for _, row in live_weather.iterrows():
        station_key = (str(row["state"]), str(row["city"]))

        current_values = {
            "temperature_C": float(row["temperature_C"]) if not pd.isna(row["temperature_C"]) else None,
            "humidity_pct": float(row["humidity_pct"]) if not pd.isna(row["humidity_pct"]) else None,
            "pressure_hPa": float(row["pressure_hPa"]) if not pd.isna(row["pressure_hPa"]) else None
        }

        previous_history = st.session_state.temporal_history.get(station_key, [])
        previous_values = previous_history[-1].copy() if previous_history else None

        temporal_info = calculate_temporal_intelligence(current_values, previous_history)
        multivariate_info = calculate_multivariate_intelligence(
            current_values, previous_values, temporal_info
        )

        # Store only complete observations so missing values do not poison the trend history.
        if all(v is not None for v in current_values.values()):
            st.session_state.temporal_history.setdefault(station_key, []).append(current_values.copy())
            st.session_state.temporal_history[station_key] = st.session_state.temporal_history[station_key][-TEMPORAL_HISTORY_LIMIT:]

        spatial_context = calculate_spatial_context(live_weather, row)

        # Frozen-value detection must use DISTINCT WEATHER API OBSERVATIONS,
        # not Streamlit reruns. Otherwise a manual click or dashboard rerun
        # can incorrectly count the same reading as another frozen observation.
        previous_frozen = st.session_state.frozen_history.get(station_key)
        current_observation_time = str(row.get("observation_time", ""))
        frozen_sensors = []
        repeat_counts = {}

        if previous_frozen is None:
            repeat_counts = {sensor: 1 if value is not None else 0 for sensor, value in current_values.items()}
        elif current_observation_time == previous_frozen.get("observation_time"):
            # Same API observation: do NOT increment the frozen counter.
            repeat_counts = previous_frozen.get("counts", {}).copy()
            for sensor, value in current_values.items():
                if value is None:
                    repeat_counts[sensor] = 0
        else:
            # New API observation: compare against the previous observation.
            for sensor, current_value in current_values.items():
                if current_value is None:
                    repeat_counts[sensor] = 0
                    continue

                previous_value = previous_frozen["values"].get(sensor)
                previous_count = previous_frozen["counts"].get(sensor, 1)

                if (
                    previous_value is not None
                    and abs(current_value - previous_value) <= FROZEN_TOLERANCES[sensor]
                ):
                    repeat_counts[sensor] = previous_count + 1
                else:
                    repeat_counts[sensor] = 1

        for sensor, count in repeat_counts.items():
            if count >= FROZEN_REPEAT_LIMIT and current_values.get(sensor) is not None:
                frozen_sensors.append(sensor)

        st.session_state.frozen_history[station_key] = {
            "values": current_values.copy(),
            "counts": repeat_counts.copy(),
            "observation_time": current_observation_time
        }

        frozen_value_detected = bool(frozen_sensors)
        frozen_sensor_text = ", ".join(SENSOR_NAMES[s] for s in frozen_sensors)

        anomaly_type, anomaly_type_reason = classify_anomaly_type(
            current_values, temporal_info, multivariate_info,
            frozen_sensors, False
        )

        if any(v is None for v in current_values.values()):
            maintenance_health, maintenance_status, maintenance_reason, maintenance_risk, forecast_health, degradation_slope = update_maintenance_health(
                station_key, anomaly_type, True,
                temporal_info["temporal_score"],
                multivariate_info["multivariate_score"],
                frozen_value_detected, 100.0
            )
            results.append({
                "state": row["state"], "city": row["city"],
                "lat": row["lat"], "lon": row["lon"],
                "temperature_C": row["temperature_C"],
                "humidity_pct": row["humidity_pct"],
                "pressure_hPa": row["pressure_hPa"],
                "anomaly_score": np.nan, "threshold": float(realtime_threshold),
                "is_anomaly": True, "risk_level": "DATA_ERROR",
                "decision_confidence": 100.0, "dominant_sensor": "Unknown",
                "anomaly_type": anomaly_type,
                "anomaly_type_reason": anomaly_type_reason,
                "weather_classification": "LIKELY_SENSOR_FAULT",
                "weather_classification_reason": "Live observation is incomplete.",
                "root_cause": "COMMUNICATION_OR_DATA_INTERRUPTION",
                "temporal_score": float(temporal_info["temporal_score"]),
                "multivariate_score": float(multivariate_info["multivariate_score"]),
                "temporal_status": "MISSING_DATA",
                "multivariate_status": multivariate_info["status"],
                "fused_anomaly_score": 1.0,
                "maintenance_health": maintenance_health,
                "maintenance_status": maintenance_status,
                "maintenance_reason": maintenance_reason,
                "maintenance_risk_percent": maintenance_risk,
                "forecast_health": forecast_health,
                "degradation_slope": degradation_slope,
                "explanation": anomaly_type_reason,
                "frozen_value_detected": frozen_value_detected,
                "frozen_sensors": frozen_sensor_text,
                "frozen_repeat_count": max(repeat_counts.values(), default=0),
                "observation_time": row["observation_time"]
            })
            continue

        input_values = np.array([[
            current_values["temperature_C"],
            current_values["humidity_pct"],
            current_values["pressure_hPa"]
        ]], dtype=np.float32)
        scaled_values = realtime_scaler.transform(input_values)
        input_tensor = torch.tensor(scaled_values, dtype=torch.float32)

        with torch.no_grad():
            reconstructed = realtime_model(input_tensor)

        squared_errors = (input_tensor - reconstructed) ** 2
        anomaly_score = float(torch.mean(squared_errors, dim=1).item())
        ai_anomaly = anomaly_score > realtime_threshold

        # Combined evidence score used for display and maintenance decisions.
        ai_signal = 1.0 if ai_anomaly else 0.0
        temporal_score = float(temporal_info["temporal_score"])
        multivariate_score = float(multivariate_info["multivariate_score"])
        fused_anomaly_score = float(
            min(1.0, 0.50 * ai_signal + 0.25 * temporal_score + 0.25 * multivariate_score)
        )

        # IMPORTANT: a frozen value is a SENSOR-HEALTH condition, not by itself
        # an AI weather/data anomaly. It must not inflate the AI Anomalies card.
        is_anomaly = bool(
            ai_anomaly
            or temporal_score >= 0.75
            or multivariate_score >= 0.75
        )

        if fused_anomaly_score < 0.25:
            risk_level = "NORMAL"
        elif fused_anomaly_score < 0.50:
            risk_level = "WARNING"
        elif fused_anomaly_score < 0.75:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        feature_errors = squared_errors[0].detach().cpu().numpy()
        dominant_sensor = ["Temperature", "Humidity", "Pressure"][int(np.argmax(feature_errors))]

        if ai_anomaly:
            ai_confidence = min(100.0, max(0.0, ((anomaly_score - realtime_threshold) / max(realtime_threshold, 1e-9)) * 100.0))
        else:
            ai_confidence = min(100.0, max(0.0, (1.0 - anomaly_score / max(realtime_threshold, 1e-9)) * 100.0))

        decision_confidence = float(min(100.0, max(0.0, 0.50 * ai_confidence + 50.0 * fused_anomaly_score)))

        # Re-classify now that AI evidence is known.
        anomaly_type, anomaly_type_reason = classify_anomaly_type(
            current_values, temporal_info, multivariate_info,
            frozen_sensors, ai_anomaly
        )

        weather_classification, weather_reason = classify_weather_vs_sensor(
            anomaly_type, temporal_info, multivariate_info,
            spatial_context, ai_anomaly
        )

        root_cause = classify_root_cause(
            anomaly_type, weather_classification, temporal_info,
            multivariate_info, spatial_context, frozen_sensors
        )

        maintenance_health, maintenance_status, maintenance_reason, maintenance_risk, forecast_health, degradation_slope = update_maintenance_health(
            station_key, anomaly_type, is_anomaly,
            temporal_score, multivariate_score,
            frozen_value_detected, decision_confidence
        )

        if not is_anomaly:
            explanation = "Observation is consistent with learned, temporal, and multivariate behavior."
        else:
            explanation = f"{anomaly_type.replace('_', ' ').title()}: {anomaly_type_reason}"
            explanation += f" Root cause: {root_cause.replace('_', ' ').title()}."
            explanation += f" Weather/fault assessment: {weather_classification.replace('_', ' ').title()}."
            if spatial_context.get("neighbor_count", 0) > 0:
                explanation += f" Spatial context: {spatial_context['status'].replace('_', ' ').lower()}."

        results.append({
            "state": row["state"], "city": row["city"],
            "lat": row["lat"], "lon": row["lon"],
            "temperature_C": row["temperature_C"],
            "humidity_pct": row["humidity_pct"],
            "pressure_hPa": row["pressure_hPa"],
            "anomaly_score": anomaly_score,
            "threshold": float(realtime_threshold),
            "is_anomaly": is_anomaly,
            "risk_level": risk_level,
            "decision_confidence": decision_confidence,
            "dominant_sensor": dominant_sensor,
            "anomaly_type": anomaly_type,
            "anomaly_type_reason": anomaly_type_reason,
            "weather_classification": weather_classification,
            "weather_classification_reason": weather_reason,
            "root_cause": root_cause,
            "temporal_score": temporal_score,
            "multivariate_score": multivariate_score,
            "temporal_status": (
                "SUDDEN_SPIKE" if temporal_info["spike_sensors"] else
                "SUDDEN_DROP" if temporal_info["drop_sensors"] else
                "SENSOR_DRIFT" if temporal_info["drift_sensors"] else
                "MULTI_SENSOR_CHANGE" if len(temporal_info["unusual_sensors"]) >= 2 else
                "UNUSUAL_CHANGE" if temporal_info["unusual_sensors"] else "CONSISTENT"
            ),
            "multivariate_status": multivariate_info["status"],
            "fused_anomaly_score": fused_anomaly_score,
            "maintenance_health": maintenance_health,
            "maintenance_status": maintenance_status,
            "maintenance_reason": maintenance_reason,
            "maintenance_risk_percent": maintenance_risk,
            "forecast_health": forecast_health,
            "degradation_slope": degradation_slope,
            "explanation": explanation,
            "frozen_value_detected": frozen_value_detected,
            "frozen_sensors": frozen_sensor_text,
            "frozen_repeat_count": max(repeat_counts.values(), default=0),
            "observation_time": row["observation_time"]
        })

    return pd.DataFrame(results)


# ============================================================
# RUN SKYGUARD AI
# ============================================================

with st.spinner(
    f"Running SkyGuard AI across {len(live_weather)} live locations..."
):

    live_ai_results = analyze_live_network(
        live_weather
    )


# ============================================================
# LIVE NETWORK SUMMARY
# ============================================================

# Always create valid_results so the dashboard remains stable
# even when a location filter produces zero valid observations.
if not live_ai_results.empty and "anomaly_score" in live_ai_results.columns:
    valid_results = live_ai_results[
        live_ai_results["anomaly_score"].notna()
    ].copy()
else:
    valid_results = live_ai_results.copy()

# Ensure downstream dashboard sections can safely render an empty result set.
required_live_columns = {
    "anomaly_score": np.nan,
    "is_anomaly": False,
    "risk_level": "NORMAL",
    "frozen_value_detected": False,
    "frozen_sensors": "",
    "frozen_repeat_count": 0,
    "fused_anomaly_score": np.nan,
    "maintenance_health": np.nan,
    "maintenance_status": "HEALTHY",
    "maintenance_risk_percent": np.nan,
    "forecast_health": np.nan,
    "degradation_slope": np.nan,
    "anomaly_type": "NORMAL_OPERATION",
    "root_cause": "NORMAL_OPERATION",
    "weather_classification": "NORMAL",
    "temporal_status": "CONSISTENT",
    "multivariate_status": "NORMAL",
    "dominant_sensor": "None",
    "state": "",
    "city": "",
    "temperature_C": np.nan,
    "humidity_pct": np.nan,
    "pressure_hPa": np.nan,
    "explanation": "",
}

for _column, _default in required_live_columns.items():
    if _column not in valid_results.columns:
        valid_results[_column] = pd.Series(dtype=object)

# The summary variables must also exist when filtering returns no rows.
total_stations = len(valid_results)
anomaly_stations = int(valid_results["is_anomaly"].sum()) if total_stations else 0
normal_stations = total_stations - anomaly_stations
warning_stations = int((valid_results["risk_level"] == "WARNING").sum()) if total_stations else 0
high_stations = int((valid_results["risk_level"] == "HIGH").sum()) if total_stations else 0
critical_stations = int((valid_results["risk_level"] == "CRITICAL").sum()) if total_stations else 0

live_anomaly_rate = (anomaly_stations / total_stations) * 100 if total_stations > 0 else 0.0
live_network_health = max(0, min(100, 100 - (live_anomaly_rate * 5)))

# ========================================================
# DISPLAY LIVE AI SECTION
# ========================================================

st.markdown(
    "## 🧠 Live SkyGuard AI Intelligence"
)

frozen_value_count = int(
    valid_results["frozen_value_detected"].sum()
)

# Interactive summary cards
# Each summary card is itself a clickable Streamlit button.  There are no
# separate "View details" buttons underneath the cards.
if "live_summary_detail" not in st.session_state:
    st.session_state.live_summary_detail = None

# Style the buttons so they look like the existing metric cards while still
# behaving as real clickable controls.
st.markdown("""
<style>
/* ============================================================
   SKYGUARD LIGHT CLICKABLE SUMMARY CARDS
   IMPORTANT: Streamlit renders st.button as a separate block,
   so the markdown wrapper cannot reliably style the button.
   We therefore target Streamlit's actual button element directly.
   ============================================================ */
.skyguard-card {
    width: 100%;
}

/* All Streamlit secondary buttons use this stable test id.
   Force them into the same light visual language so dark-theme
   defaults cannot turn the cards black. */
div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"] {
    width: 100% !important;
    min-height: 112px !important;
    border-radius: 16px !important;
    border: 1px solid #dbe7f3 !important;
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #17233c !important;
    -webkit-text-fill-color: #17233c !important;
    box-shadow: 0 6px 18px rgba(31, 55, 88, 0.10) !important;
    text-align: center !important;
    padding: 18px 12px !important;
    opacity: 1 !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease !important;
}

div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:hover {
    transform: translateY(-2px) !important;
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #17233c !important;
    -webkit-text-fill-color: #17233c !important;
    box-shadow: 0 10px 24px rgba(31, 55, 88, 0.14) !important;
    border-color: #b9d5ef !important;
}

div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:focus,
div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:active {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #17233c !important;
    -webkit-text-fill-color: #17233c !important;
    border-color: #8fc2ed !important;
}

/* Text inside Streamlit's actual button. */
div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"] p,
div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"] span,
div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"] div,
div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"] {
    color: #17233c !important;
    -webkit-text-fill-color: #17233c !important;
}

div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"] p {
    white-space: pre-wrap !important;
    line-height: 1.45 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

card_columns = st.columns(6)
card_definitions = [
    ("stations_analyzed", "Stations Analyzed", total_stations),
    ("normal", "Normal", normal_stations),
    ("ai_anomalies", "AI Anomalies", anomaly_stations),
    ("critical", "Critical", critical_stations),
    ("frozen", "🧊 Frozen Values", frozen_value_count),
    ("network_health", "Network Health", f"{live_network_health:.1f}%"),
]

for column, (mode, label, value) in zip(card_columns, card_definitions):
    with column:
        st.markdown('<div class="skyguard-card">', unsafe_allow_html=True)
        if st.button(
            f"{label}\n\n{value}",
            key=f"summary_card_{mode}",
            use_container_width=True,
        ):
            st.session_state.live_summary_detail = mode
        st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# Interactive summary-card details
# --------------------------------------------------------
detail_mode = st.session_state.live_summary_detail

if detail_mode == "stations_analyzed":
    st.markdown("### 📍 All Analyzed Station Details")

    if valid_results.empty:
        st.info("No stations are currently available.")
    else:
        station_view = valid_results[[
            "state", "city", "temperature_C", "humidity_pct", "pressure_hPa",
            "is_anomaly", "anomaly_type", "risk_level", "dominant_sensor",
            "maintenance_health", "maintenance_status", "root_cause"
        ]].copy()
        station_view.columns = [
            "State", "City", "Temperature (°C)", "Humidity (%)", "Pressure (hPa)",
            "AI Anomaly", "Anomaly Type", "Risk", "Affected Sensor",
            "Sensor Health", "Maintenance Status", "Root Cause"
        ]
        st.dataframe(station_view, use_container_width=True, hide_index=True)
        st.info(f"Showing details for all {len(station_view)} analyzed station(s).")

elif detail_mode == "normal":
    st.markdown("### 🟢 Normal Station Details")

    detail = valid_results[valid_results["is_anomaly"] == False].copy()
    if detail.empty:
        st.success("No stations are currently classified as normal.")
    else:
        normal_view = detail[[
            "state", "city", "temperature_C", "humidity_pct", "pressure_hPa",
            "maintenance_health", "maintenance_status", "dominant_sensor", "root_cause"
        ]].copy()
        normal_view.columns = [
            "State", "City", "Temperature (°C)", "Humidity (%)", "Pressure (hPa)",
            "Sensor Health", "Maintenance Status", "Affected Sensor", "Root Cause"
        ]
        st.dataframe(normal_view, use_container_width=True, hide_index=True)
        st.success(f"{len(normal_view)} station(s) are currently operating normally.")

elif detail_mode == "ai_anomalies":
    st.markdown("### 🚨 AI Anomaly Details")

    detail = valid_results[valid_results["is_anomaly"] == True].copy()

    if detail.empty:
        st.success("No AI anomalies are currently detected.")
    else:
        detail = detail.sort_values("fused_anomaly_score", ascending=False)

        detail_view = detail[[
            "state", "city", "dominant_sensor",
            "temperature_C", "humidity_pct", "pressure_hPa",
            "anomaly_score", "fused_anomaly_score", "risk_level",
            "anomaly_type", "weather_classification", "root_cause",
            "temporal_status", "multivariate_status",
            "maintenance_health", "maintenance_status",
            "explanation"
        ]].copy()

        detail_view.columns = [
            "State", "City", "Affected Sensor",
            "Temperature (°C)", "Humidity (%)", "Pressure (hPa)",
            "AI Score", "Fused Score", "Risk",
            "Anomaly Type", "Weather/Fault", "Root Cause",
            "Temporal Evidence", "Multivariate Evidence",
            "Sensor Health", "Maintenance Status", "Explanation"
        ]

        st.dataframe(detail_view, use_container_width=True, hide_index=True)
        st.info(
            "Each row identifies the exact area, affected sensor, current readings, "
            "anomaly evidence, likely root cause, sensor health, and AI explanation."
        )

elif detail_mode == "critical":
    st.markdown("### 🔴 Critical Station Details")

    detail = valid_results[valid_results["risk_level"] == "CRITICAL"].copy()
    if detail.empty:
        st.success("No stations are currently at critical risk.")
    else:
        detail = detail.sort_values("fused_anomaly_score", ascending=False)
        critical_view = detail[[
            "state", "city", "dominant_sensor",
            "temperature_C", "humidity_pct", "pressure_hPa",
            "fused_anomaly_score", "anomaly_type", "root_cause",
            "weather_classification", "maintenance_health", "maintenance_status",
            "explanation"
        ]].copy()
        critical_view.columns = [
            "State", "City", "Affected Sensor",
            "Temperature (°C)", "Humidity (%)", "Pressure (hPa)",
            "Risk Score", "Anomaly Type", "Root Cause",
            "Weather/Fault", "Sensor Health", "Maintenance Status", "Explanation"
        ]
        st.dataframe(critical_view, use_container_width=True, hide_index=True)
        st.error(f"{len(critical_view)} station(s) require immediate attention.")

elif detail_mode == "frozen":
    st.markdown("### 🧊 Frozen Sensor Details")

    frozen_detail = valid_results[
        valid_results["frozen_value_detected"] == True
    ].copy()

    if frozen_detail.empty:
        st.success("No frozen sensors are currently detected.")
    else:
        frozen_rows = []

        for _, row in frozen_detail.iterrows():
            sensors = row.get("frozen_sensors", [])

            if isinstance(sensors, str):
                sensors = [x.strip() for x in sensors.split(",") if x.strip()]
            elif not isinstance(sensors, (list, tuple, set)):
                sensors = []

            if not sensors:
                sensors = [str(row.get("dominant_sensor", "Unknown sensor"))]

            for sensor in sensors:
                sensor_key = {
                    "Temperature": "temperature_C",
                    "Humidity": "humidity_pct",
                    "Pressure": "pressure_hPa",
                    "temperature_C": "temperature_C",
                    "humidity_pct": "humidity_pct",
                    "pressure_hPa": "pressure_hPa"
                }.get(str(sensor), None)

                frozen_rows.append({
                    "State": row.get("state", ""),
                    "City": row.get("city", ""),
                    "Frozen Sensor": str(sensor),
                    "Frozen Value": row.get(sensor_key, np.nan) if sensor_key else np.nan,
                    "Temperature (°C)": row.get("temperature_C", np.nan),
                    "Humidity (%)": row.get("humidity_pct", np.nan),
                    "Pressure (hPa)": row.get("pressure_hPa", np.nan),
                    "Consecutive Observations": row.get("frozen_repeat_count", 0),
                    "Risk": row.get("risk_level", "NORMAL"),
                    "Anomaly Type": row.get("anomaly_type", ""),
                    "Root Cause": row.get("root_cause", ""),
                    "Sensor Health": row.get("maintenance_health", np.nan),
                    "Maintenance Status": row.get("maintenance_status", ""),
                    "Explanation": row.get("explanation", "")
                })

        frozen_view = pd.DataFrame(frozen_rows)
        st.dataframe(frozen_view, use_container_width=True, hide_index=True)
        st.warning(
            "A frozen sensor is a sensor-health condition. It is not automatically "
            "counted as an AI anomaly unless independent anomaly evidence is present."
        )

elif detail_mode == "network_health":
    st.markdown("### 🌐 Network Health Details")

    if valid_results.empty:
        st.info("No live stations are available for network-health analysis.")
    else:
        health_view = valid_results[[
            "state", "city", "maintenance_health", "maintenance_status",
            "maintenance_risk_percent", "degradation_slope",
            "is_anomaly", "risk_level", "anomaly_type", "root_cause",
            "dominant_sensor"
        ]].copy()

        health_view = health_view.sort_values("maintenance_health", ascending=True)
        health_view.columns = [
            "State", "City", "Sensor Health", "Maintenance Status",
            "Maintenance Risk (%)", "Health Trend", "Anomaly", "Risk",
            "Latest Anomaly", "Root Cause", "Affected Sensor"
        ]

        st.dataframe(health_view, use_container_width=True, hide_index=True)

        declining = int((valid_results["degradation_slope"] < -1.0).sum())
        urgent = int((valid_results["maintenance_status"] == "MAINTENANCE_URGENT").sum())
        recommended = int((valid_results["maintenance_status"] == "MAINTENANCE_RECOMMENDED").sum())

        st.info(
            f"Network health is {live_network_health:.1f}%. "
            f"{declining} station(s) show declining health, "
            f"{recommended} require recommended maintenance, and "
            f"{urgent} require urgent maintenance."
        )


# ========================================================
# RISK DISTRIBUTION
# ========================================================

risk_counts = (
    valid_results["risk_level"]
    .value_counts()
    .reindex(
        [
            "NORMAL",
            "WARNING",
            "HIGH",
            "CRITICAL"
        ],
        fill_value=0
    )
    .reset_index()
)

risk_counts.columns = [
    "Risk Level",
    "Stations"
]

fig_live_risk = px.bar(
    risk_counts,
    x="Risk Level",
    y="Stations",
    title="Live AI Risk Distribution"
)

st.plotly_chart(
    fig_live_risk,
    use_container_width=True
)


# ========================================================
# LIVE ANOMALY TABLE
# ========================================================

st.markdown(
    "### 🚨 Live AI Anomaly Detection"
)

anomaly_table = valid_results[
    valid_results["is_anomaly"] == True
].copy()

if not anomaly_table.empty:

    anomaly_table = anomaly_table.sort_values(
        "anomaly_score",
        ascending=False
    )

    display_anomalies = anomaly_table[
        [
            "state",
            "city",
            "temperature_C",
            "humidity_pct",
            "pressure_hPa",
            "anomaly_score",
            "fused_anomaly_score",
            "risk_level",
            "anomaly_type",
            "weather_classification",
            "root_cause",
            "temporal_status",
            "multivariate_status",
            "maintenance_health",
            "maintenance_status",
            "dominant_sensor",
            "frozen_value_detected",
            "frozen_sensors",
            "frozen_repeat_count",
            "explanation"
        ]
    ].copy()

    display_anomalies.columns = [
        "State",
        "City",
        "Temperature (°C)",
        "Humidity (%)",
        "Pressure (hPa)",
        "AI Anomaly Score",
        "Fused Score",
        "Risk Level",
        "Anomaly Type",
        "Weather/Fault",
        "Root Cause",
        "Temporal",
        "Multivariate",
        "Maintenance Health",
        "Maintenance Status",
        "Dominant Feature",
        "Frozen Value",
        "Frozen Sensors",
        "Repeat Count",
        "AI Explanation"
    ]

    st.dataframe(
        display_anomalies,
        use_container_width=True,
        hide_index=True
    )

else:

    st.success(
        "No live AI anomalies detected across "
        "the monitored network."
    )

# ============================================================
# PRIORITY 10 - PREDICTIVE MAINTENANCE
# ============================================================

st.markdown("### 🔧 Predictive Maintenance & Sensor Degradation")

maintenance_columns = [
    "state", "city", "maintenance_health", "maintenance_status",
    "maintenance_risk_percent", "forecast_health", "degradation_slope",
    "anomaly_type", "root_cause", "maintenance_reason"
]

if not valid_results.empty:
    maintenance_table = valid_results[
        [c for c in maintenance_columns if c in valid_results.columns]
    ].copy()

    maintenance_table = maintenance_table.sort_values(
        "maintenance_health", ascending=True
    )

    maintenance_table.columns = [
        "State", "City", "Current Health", "Maintenance Status",
        "Maintenance Risk (%)", "Forecast Health", "Health Trend",
        "Latest Anomaly", "Root Cause", "Maintenance Assessment"
    ]

    st.dataframe(
        maintenance_table,
        use_container_width=True,
        hide_index=True
    )

    urgent = int((valid_results["maintenance_status"] == "MAINTENANCE_URGENT").sum())
    recommended = int((valid_results["maintenance_status"] == "MAINTENANCE_RECOMMENDED").sum())
    watch = int((valid_results["maintenance_status"] == "WATCH").sum())

    r1, r2 = st.columns(2)
    with r1:
        st.metric(
            "Average Predicted Maintenance Risk",
            f"{valid_results['maintenance_risk_percent'].mean():.1f}%"
        )
    with r2:
        declining = int((valid_results["degradation_slope"] < -1.0).sum())
        st.metric("Stations With Declining Health", declining)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Healthy Stations", int((valid_results["maintenance_status"] == "HEALTHY").sum()))
    with m2:
        st.metric("Watch Stations", watch)
    with m3:
        st.metric("Maintenance Recommended", recommended)
    with m4:
        st.metric("Urgent Maintenance", urgent)

# ============================================================
# FROZEN-VALUE ALERTS
# ============================================================

frozen_alerts = valid_results[
    valid_results["frozen_value_detected"] == True
].copy()

st.markdown("### 🧊 Frozen-Value Detection")

if not frozen_alerts.empty:

    st.warning(
        f"Possible frozen values detected at "
        f"{len(frozen_alerts)} monitored location(s)."
    )

    frozen_display = frozen_alerts[
        [
            "state",
            "city",
            "temperature_C",
            "humidity_pct",
            "pressure_hPa",
            "frozen_sensors",
            "frozen_repeat_count",
            "risk_level"
        ]
    ].copy()

    frozen_display.columns = [
        "State",
        "City",
        "Temperature (°C)",
        "Humidity (%)",
        "Pressure (hPa)",
        "Frozen Sensors",
        "Consecutive Observations",
        "AI Risk Level"
    ]

    st.dataframe(
        frozen_display,
        use_container_width=True,
        hide_index=True
    )

else:

    st.success(
        "No frozen sensor values detected "
        "across the monitored network."
    )

# ============================================================
# STEP 14 - PROFESSIONAL SKYGUARD UI
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN APPLICATION
       ======================================================== */

    .stApp {
        background: #f4f7fb;
        color: #172033;
    }

    /* ========================================================
       STREAMLIT HEADER
       ======================================================== */

    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* ========================================================
       ALL NORMAL TEXT
       ======================================================== */

    h1, h2, h3, h4, h5, h6 {
        color: #172033 !important;
        font-weight: 700;
    }

    [data-testid="stMarkdownContainer"] p {
        color: #172033 !important;
    }

    /* ========================================================
       KPI CARDS
       ======================================================== */

    div[data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #d9e2ef;
        box-shadow: 0 4px 14px rgba(40, 70, 110, 0.08);
    }

    div[data-testid="stMetricLabel"] {
        color: #52627a !important;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #172033 !important;
        font-weight: 800;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #e8f0fa;
        border-right: 1px solid #d3deeb;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #172033 !important;
    }

    /* ========================================================
       SELECTBOX
       ======================================================== */

    div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border-radius: 8px;
    }

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
    }

    div[data-baseweb="select"] span {
        color: #172033 !important;
    }

    /* Dropdown menu */
    ul[role="listbox"] {
        background-color: #ffffff !important;
    }

    ul[role="listbox"] li {
        color: #172033 !important;
        background-color: #ffffff !important;
    }

    ul[role="listbox"] li:hover {
        background-color: #eef4fb !important;
    }

    /* ========================================================
       TEXT INPUTS / SEARCH BOXES
       ======================================================== */

    input {
        background-color: #ffffff !important;
        color: #172033 !important;
        border: 1px solid #cbd6e5 !important;
        border-radius: 8px !important;
    }

    input::placeholder {
        color: #718096 !important;
    }

    textarea {
        background-color: #ffffff !important;
        color: #172033 !important;
        border: 1px solid #cbd6e5 !important;
        border-radius: 8px !important;
    }

    textarea::placeholder {
        color: #718096 !important;
    }

    /* ========================================================
       SEARCH / INPUT CONTAINERS
       ======================================================== */

    div[data-baseweb="input"] {
        background-color: #ffffff !important;
    }

    div[data-baseweb="input"] > div {
        background-color: #ffffff !important;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    button {
        color: #172033 !important;
    }

    /* ========================================================
       ALERT BOXES
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* ========================================================
       DATA TABLES
       ======================================================== */

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #d9e2ef;
    }

    /* ========================================================
       CAPTIONS
       ======================================================== */

    [data-testid="stCaptionContainer"] {
        color: #52627a !important;
    }

    /* ========================================================
       SEPARATORS
       ======================================================== */

    hr {
        border-color: #d9e2ef;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    city = pd.read_csv(
        BASE_DIR / "skyguard_city_anomaly_summary.csv"
    )

    coordinates = pd.read_csv(
        BASE_DIR / "skyguard_city_coordinates.csv"
    )

    state = pd.read_csv(
        BASE_DIR / "skyguard_state_anomaly_summary.csv"
    )

    # --------------------------------------------------------
    # MERGE CITY DATA WITH COORDINATES
    # --------------------------------------------------------

    city = city.merge(
        coordinates,
        on=["state", "city"],
        how="left"
    )

    return (
        city,
        state
    )


(
    city,
    state
) = load_data()

# ============================================================
# GLOBAL FILTER FOR HISTORICAL / CITY ANALYSIS
# ============================================================

filtered_city = city.copy()

if search_text:
    historical_search_mask = (
        filtered_city["state"].astype(str).str.contains(
            search_text, case=False, na=False
        )
        | filtered_city["city"].astype(str).str.contains(
            search_text, case=False, na=False
        )
    )
    filtered_city = filtered_city[historical_search_mask]


# ============================================================
# HEADER
# ============================================================

st.title("🌦️ SkyGuard V2")

st.subheader(
    "AI/ML-Based Intelligent Anomaly Detection "
    "for Automatic Weather Stations"
)

st.markdown(
    """
    **SkyGuard V2** uses an autoencoder-based anomaly detection
    system to identify unusual weather-station observations
    and potential sensor/data anomalies in real time.
    """
)
st.divider()

filter_label = []
if search_text:
    filter_label.append(f"location: **{search_text}**")

if filter_label:
    st.info(
        "🔎 Global filter active — "
        + " • ".join(filter_label)
        + f" • {len(live_filter)} live location(s)"
    )
else:
    st.caption(
        f"🔎 Global filter: All locations • {len(live_locations)} live locations"
    )


# ============================================================
# STEP 17D - UNIFIED INDIA ANOMALY MAP
# ============================================================

st.header("🗺️ SkyGuard AWS Network Map")

st.caption(
    "Switch between historical anomaly intelligence "
    "and real-time AI monitoring using the same map."
)

# ============================================================
# MAP VIEW SELECTOR
# ============================================================

map_mode = st.radio(
    "Select Map View",
    [
        "Historical Intelligence",
        "Live AI Monitoring"
    ],
    horizontal=True
)


# ============================================================
# HISTORICAL INTELLIGENCE MAP
# ============================================================

if map_mode == "Historical Intelligence":

    st.subheader("📊 Historical Anomaly Intelligence")

    st.caption(
        "Historical anomaly distribution across the "
        "analyzed weather station network."
    )

    # Use the SAME dataframe already used by your dashboard
    map_data = filtered_city.dropna(
        subset=["lat", "lon"]
    ).copy()

    if len(map_data) > 0:

        fig_map = px.scatter_map(
            map_data,

            lat="lat",
            lon="lon",

            size="anomalies",

            color="anomaly_rate_percent",

            hover_name="city",

            hover_data={
                "state": True,
                "anomalies": True,
                "avg_score": ":.3f",
                "anomaly_rate_percent": ":.2f",
                "lat": False,
                "lon": False
            },

            center={"lat": 22.5, "lon": 79.0},

            zoom=4.3,

            height=650,

            title="SkyGuard V2 - Historical Anomaly Locations",

            color_continuous_scale="Turbo"
        )

        fig_map.update_layout(
            map_style="open-street-map",
            map={
                "center": {"lat": 22.5, "lon": 79.0},
                "zoom": 4.3
            },
            margin={"r": 0, "t": 55, "l": 0, "b": 0}
        )

        st.plotly_chart(
            fig_map,
            use_container_width=True
        )

    else:

        st.warning(
            "No coordinate data available for "
            "the selected state."
        )


# ============================================================
# LIVE AI MONITORING MAP
# ============================================================

else:

    st.subheader("🧠 Live AI Monitoring")

    st.caption(
        "Real-time AI assessment of the monitored "
        "weather observation network."
    )

    # --------------------------------------------------------
    # Make sure live AI results exist
    # --------------------------------------------------------

    if (
        "live_ai_results" in locals()
        and not live_ai_results.empty
    ):

        live_map_data = live_ai_results.dropna(
            subset=["lat", "lon"]
        ).copy()

        # ----------------------------------------------------
        # LIVE MAP
        # ----------------------------------------------------

        if len(live_map_data) > 0:

            fig_live_map = px.scatter_map(

                live_map_data,

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

                center={"lat": 22.5, "lon": 79.0},

                zoom=4.3,

                height=650,

                title="SkyGuard AI - Live AWS Network Monitoring"
            )

            fig_live_map.update_layout(
                map_style="open-street-map",
                map={
                    "center": {"lat": 22.5, "lon": 79.0},
                    "zoom": 4.3
                },
                margin={"r": 0, "t": 55, "l": 0, "b": 0}
            )

            st.plotly_chart(
                fig_live_map,
                use_container_width=True
            )

            # ------------------------------------------------
            # LIVE NETWORK STATUS
            # ------------------------------------------------

            st.markdown(
                "### 📡 Live Network Status"
            )

            total_live = len(live_map_data)

            normal_live = int(
                (
                    live_map_data["risk_level"]
                    == "NORMAL"
                ).sum()
            )

            warning_live = int(
                (
                    live_map_data["risk_level"]
                    == "WARNING"
                ).sum()
            )

            high_live = int(
                (
                    live_map_data["risk_level"]
                    == "HIGH"
                ).sum()
            )

            critical_live = int(
                (
                    live_map_data["risk_level"]
                    == "CRITICAL"
                ).sum()
            )

            anomaly_live = (
                warning_live +
                high_live +
                critical_live
            )

            live_health = max(
                0,
                min(
                    100,
                    100 -
                    (
                        anomaly_live /
                        total_live *
                        100 *
                        5
                    )
                )
            ) if total_live > 0 else 0


            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:

                st.metric(
                    "Locations",
                    total_live
                )

            with col2:

                st.metric(
                    "Normal",
                    normal_live
                )

            with col3:

                st.metric(
                    "Anomalies",
                    anomaly_live
                )

            with col4:

                st.metric(
                    "Critical",
                    critical_live
                )

            with col5:

                st.metric(
                    "Network Health",
                    f"{live_health:.1f}%"
                )


            # ------------------------------------------------
            # LIVE STATION INVESTIGATION
            # ------------------------------------------------

            st.markdown(
                "### 🔎 Investigate Live Station"
            )

            location_options = (
                live_map_data["state"].astype(str)
                + " — "
                + live_map_data["city"].astype(str)
            ).tolist()

            selected_live_location = st.selectbox(
                "Select a monitored location",
                location_options,
                key="live_station_selector"
            )

            selected_mask = (
                live_map_data["state"].astype(str)
                + " — "
                + live_map_data["city"].astype(str)
            ) == selected_live_location

            selected_station = live_map_data[
                selected_mask
            ].iloc[0]

            # ------------------------------------------------
            # STATION WEATHER DATA
            # ------------------------------------------------

            st.markdown(
                f"#### 📍 {selected_station['city']}, "
                f"{selected_station['state']}"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Temperature",
                    f"{selected_station['temperature_C']:.1f} °C"
                )

            with col2:

                st.metric(
                    "Humidity",
                    f"{selected_station['humidity_pct']:.1f} %"
                )

            with col3:

                st.metric(
                    "Pressure",
                    f"{selected_station['pressure_hPa']:.1f} hPa"
                )

            # ------------------------------------------------
            # FROZEN-VALUE STATUS
            # ------------------------------------------------

            frozen_detected = bool(
                selected_station.get(
                    "frozen_value_detected",
                    False
                )
            )

            frozen_sensors = selected_station.get(
                "frozen_sensors",
                []
            )

            frozen_repeat_count = int(
                selected_station.get(
                    "frozen_repeat_count",
                    0
                )
            )

            if frozen_detected:

                st.warning(
                    "🧊 Possible frozen value detected"
                )

                if isinstance(frozen_sensors, list):
                    sensor_text = ", ".join(
                        str(sensor)
                        for sensor in frozen_sensors
                    )
                else:
                    sensor_text = str(frozen_sensors)

                st.info(
                    f"Frozen sensor(s): {sensor_text}  |  "
                    f"Consecutive observations: "
                    f"{frozen_repeat_count}"
                )

            else:

                st.success(
                    "✅ No frozen sensor values detected"
                )

            # ------------------------------------------------
            # AI ASSESSMENT
            # ------------------------------------------------

            st.markdown(
                "##### 🧠 SkyGuard AI Assessment"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "AI Anomaly Score",
                    f"{selected_station['anomaly_score']:.5f}"
                )

            with col2:

                st.metric(
                    "Detection Threshold",
                    f"{selected_station['threshold']:.5f}"
                )

            with col3:

                st.metric(
                    "Risk Level",
                    selected_station["risk_level"]
                )

            with col4:

                st.metric(
                    "AI Decision Confidence",
                    f"{selected_station['decision_confidence']:.1f}%"
                )


            # ------------------------------------------------
            # AI EXPLANATION
            # ------------------------------------------------

            risk = selected_station["risk_level"]

            if risk == "NORMAL":

                st.success(
                    "🟢 "
                    + selected_station["explanation"]
                )

            elif risk == "WARNING":

                st.warning(
                    "🟡 "
                    + selected_station["explanation"]
                )

            elif risk == "HIGH":

                st.error(
                    "🟠 "
                    + selected_station["explanation"]
                )

            elif risk == "CRITICAL":

                st.error(
                    "🔴 "
                    + selected_station["explanation"]
                )

            else:

                st.info(
                    selected_station["explanation"]
                )


            # ------------------------------------------------
            # DOMINANT SENSOR
            # ------------------------------------------------

            if risk != "NORMAL":

                st.info(
                    "🎯 Dominant sensor contribution: "
                    + str(
                        selected_station[
                            "dominant_sensor"
                        ]
                    )
                )


            # ------------------------------------------------
            # OBSERVATION TIME
            # ------------------------------------------------

            if "observation_time" in selected_station:

                st.caption(
                    "🕐 Observation time: "
                    + str(
                        selected_station[
                            "observation_time"
                        ]
                    )
                )

        else:

            st.warning(
                "No live location coordinates are available."
            )

    else:

        st.info(
            "Live AI monitoring data is not available yet."
        )

# ============================================================
# STEP 17E - HISTORICAL AI VALIDATION
# PRE-COMPUTED RESULT FILE MODE
# ============================================================
# Uses the already-generated SkyGuard V2 result file directly.
# NO historical dataset loading and NO historical AI re-processing.
# ============================================================

st.markdown("---")
st.header("📊 Historical AI Validation")

HISTORICAL_RESULT_FILE = BASE_DIR / "skyguard_v2_results.csv"

@st.cache_data

def load_historical_results():
    """Load the pre-computed SkyGuard V2 historical results only."""
    if not HISTORICAL_RESULT_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(HISTORICAL_RESULT_FILE)

historical_ai_results = load_historical_results()

if not historical_ai_results.empty:

    total_historical = len(historical_ai_results)

    historical_anomalies = int(
        historical_ai_results["is_anomaly"].sum()
    )

    historical_normal = total_historical - historical_anomalies

    historical_anomaly_rate = (
        historical_anomalies / total_historical * 100
    ) if total_historical else 0.0

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Historical Records", f"{total_historical:,}")

    with col2:
        st.metric("AI Anomalies", f"{historical_anomalies:,}")

    with col3:
        st.metric("Normal Records", f"{historical_normal:,}")

    with col4:
        st.metric("Anomaly Rate", f"{historical_anomaly_rate:.2f}%")

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    if "risk_level" in historical_ai_results.columns:
        st.subheader("Historical AI Risk Distribution")

        historical_risk = (
            historical_ai_results["risk_level"]
            .value_counts()
            .reindex(
                ["NORMAL", "WARNING", "HIGH", "CRITICAL"],
                fill_value=0
            )
            .reset_index()
        )

        historical_risk.columns = ["Risk Level", "Records"]

        fig_historical_risk = px.bar(
            historical_risk,
            x="Risk Level",
            y="Records",
            title="3-Feature AI Historical Risk Distribution"
        )

        st.plotly_chart(
            fig_historical_risk,
            use_container_width=True
        )

    # --------------------------------------------------------
    # TOP HISTORICAL ANOMALIES
    # --------------------------------------------------------

    st.subheader("🚨 Highest-Scoring Historical Observations")

    top_historical = historical_ai_results[
        historical_ai_results["is_anomaly"] == True
    ].sort_values(
        "anomaly_score", ascending=False
    ).head(20)

    if not top_historical.empty:
        display_columns = [
            "datetime",
            "state",
            "city",
            "temperature_C",
            "humidity_pct",
            "pressure_hPa",
            "anomaly_score",
            "risk_level",
            "dominant_sensor"
        ]

        display_columns = [
            c for c in display_columns
            if c in top_historical.columns
        ]

        st.dataframe(
            top_historical[display_columns],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("No historical anomalies were detected.")

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.info(
        "Historical validation uses the pre-computed "
        "SkyGuard V2 result file. No historical AI inference "
        "is performed by the dashboard."
    )

else:
    st.warning(
        "skyguard_v2_results.csv was not found. "
        "No historical dataset will be processed automatically."
    )


# ============================================================
# TOP LOCATIONS
# ============================================================

st.subheader(
    "🏆 Highest Anomaly-Rate Locations"
)

top_locations = (
    filtered_city
    .sort_values(
        "anomaly_rate_percent",
        ascending=False
    )
    .head(10)
)

st.dataframe(
    top_locations[
        [
            "state",
            "city",
            "total_records",
            "anomalies",
            "avg_score",
            "anomaly_rate_percent"
        ]
    ],
    use_container_width=True,
    hide_index=True
)




# ============================================================
# STEP 14B - EXECUTIVE COMMAND CENTER
# ============================================================

st.header("🎯 Executive Command Center")

st.markdown(
    """
    **SkyGuard AI Executive Command Center**

    A high-level operational view of the Automatic Weather Station
    network, converting AI anomaly-detection results into actionable
    information for monitoring teams and decision makers.
    """
)

# ------------------------------------------------------------
# EXECUTIVE ANALYSIS
# ------------------------------------------------------------

if len(filtered_city) > 0:

    # Find highest-risk location
    executive_location = filtered_city.loc[
        filtered_city["anomaly_rate_percent"].idxmax()
    ]

    executive_city = str(
        executive_location["city"]
    )

    executive_state = str(
        executive_location["state"]
    )

    executive_rate = float(
        executive_location["anomaly_rate_percent"]
    )

    executive_anomalies = int(
        executive_location["anomalies"]
    )

    executive_score = float(
        executive_location["avg_score"]
    )

    # --------------------------------------------------------
    # THREAT LEVEL
    # --------------------------------------------------------

    if executive_rate >= 50:

        executive_level = "CRITICAL"
        executive_icon = "🔴"

    elif executive_rate >= 25:

        executive_level = "HIGH"
        executive_icon = "🟠"

    elif executive_rate >= 10:

        executive_level = "WARNING"
        executive_icon = "🟡"

    else:

        executive_level = "NORMAL"
        executive_icon = "🟢"

    # --------------------------------------------------------
    # OVERALL NETWORK ANOMALY RATE
    # --------------------------------------------------------

    total_network_records = filtered_city["total_records"].sum()
    total_network_anomalies = filtered_city["anomalies"].sum()

    if total_network_records > 0:
        network_anomaly_rate = (
            total_network_anomalies /
            total_network_records
        ) * 100
    else:
        network_anomaly_rate = 0.0

    # --------------------------------------------------------
    # NETWORK HEALTH SCORE
    # --------------------------------------------------------

    health_score = max(
        0,
        min(
            100,
            100 - (network_anomaly_rate * 5)
        )
    )
    # --------------------------------------------------------
    # EXECUTIVE KPI CARDS
    # --------------------------------------------------------

    e1, e2, e3, e4 = st.columns(4)

    with e1:

        st.metric(
            "🟢 Network Health",
            f"{health_score:.1f}/100"
        )

    with e2:

        st.metric(
            "🚨 Threat Level",
            f"{executive_icon} {executive_level}"
        )

    with e3:

        st.metric(
            "📍 Priority Location",
            executive_city
        )

    with e4:

        st.metric(
            "⚠️ Priority Anomaly Rate",
            f"{executive_rate:.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # CURRENT NETWORK SITUATION
    # --------------------------------------------------------

    st.subheader("📡 Current Network Situation")

    situation1, situation2 = st.columns(2)

    with situation1:

        st.markdown(
            f"""
            ### {executive_icon} {executive_level} Situation

            **Priority Location**

            {executive_city}, {executive_state}

            **Anomalous Observations**

            {executive_anomalies:,}

            **Average AI Anomaly Score**

            {executive_score:.3f}

            **Overall Network Anomaly Rate**

            {network_anomaly_rate:.2f}%
            """
        )

    with situation2:

        st.markdown(
            "### 🏆 Top Priority Locations"
        )

        priority_locations = (
            filtered_city
            .sort_values(
                "anomaly_rate_percent",
                ascending=False
            )
            .head(5)
            [
                [
                    "state",
                    "city",
                    "anomaly_rate_percent",
                    "anomalies"
                ]
            ]
            .copy()
        )

        priority_locations.columns = [
            "State",
            "Location",
            "Anomaly Rate (%)",
            "Anomalies"
        ]

        st.dataframe(
            priority_locations,
            use_container_width=True,
            hide_index=True
        )

   
    # --------------------------------------------------------
    # EXECUTIVE AI DECISION SUPPORT
    # --------------------------------------------------------

    st.subheader("🤖 Executive AI Decision Support")

    if executive_level == "CRITICAL":

        executive_message = f"""
🔴 **IMMEDIATE ATTENTION REQUIRED**

SkyGuard AI has identified **{executive_city},
{executive_state}** as the highest-priority location.

The detected anomaly rate is **{executive_rate:.2f}%**
with **{executive_anomalies:,} anomalous observations**.

### Recommended Decision

Prioritize immediate AWS inspection, sensor verification,
calibration checks and comparison with nearby weather stations.
"""

    elif executive_level == "HIGH":

        executive_message = f"""
🟠 **HIGH PRIORITY**

SkyGuard AI has identified **{executive_city},
{executive_state}** as the highest-risk location.

The detected anomaly rate is **{executive_rate:.2f}%**.

### Recommended Decision

Schedule AWS inspection, verify the affected sensors
and investigate repeated anomalous observations.
"""

    elif executive_level == "WARNING":

        executive_message = f"""
🟡 **MONITORING REQUIRED**

SkyGuard AI detected elevated anomaly activity at
**{executive_city}, {executive_state}**.

The detected anomaly rate is **{executive_rate:.2f}%**.

### Recommended Decision

Continue monitoring and review recent sensor measurements
for increasing anomaly activity.
"""

    else:

        executive_message = f"""
🟢 **NETWORK OPERATING WITHIN NORMAL RISK LEVELS**

The highest detected anomaly rate is **{executive_rate:.2f}%**
at **{executive_city}, {executive_state}**.

### Recommended Decision

Continue normal monitoring and periodic sensor
data-quality verification.
"""

    st.info(executive_message)

else:

    st.warning(
        "No locations are currently available for "
        "Executive Command Center analysis."
    )



# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SkyGuard V2 • AI/ML Weather Station Anomaly Detection • "
    "Intel Arc XPU Accelerated"
)
