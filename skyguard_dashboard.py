
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ============================================================
# SKYGUARD V2 - AI WEATHER STATION ANOMALY DASHBOARD
# ============================================================

st.set_page_config(
    page_title="SkyGuard V2",
    page_icon="🌦️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

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

    sensor = pd.read_csv(
        BASE_DIR / "skyguard_sensor_anomaly_summary.csv"
    )

    contribution = pd.read_csv(
        BASE_DIR / "skyguard_sensor_contributions.csv"
    )

    monthly = pd.read_csv(
        BASE_DIR / "skyguard_monthly_anomaly_summary.csv"
    )

    hourly = pd.read_csv(
        BASE_DIR / "skyguard_hourly_anomaly_summary.csv"
    )

    persistent = pd.read_csv(
        BASE_DIR / "skyguard_persistent_events.csv"
    )

    top = pd.read_csv(
        BASE_DIR / "skyguard_top_anomalies.csv"
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
        state,
        sensor,
        contribution,
        monthly,
        hourly,
        persistent,
        top
    )


(
    city,
    state,
    sensor,
    contribution,
    monthly,
    hourly,
    persistent,
    top
) = load_data()

# ============================================================
# SIDEBAR FILTER
# ============================================================

st.sidebar.title("🔎 SkyGuard Filters")

states = ["All States"] + sorted(
    state["state"].dropna().unique().tolist()
)

selected_state = st.sidebar.selectbox(
    "Select State",
    states
)

if selected_state == "All States":
    filtered_city = city.copy()
else:
    filtered_city = city[
        city["state"] == selected_state
    ].copy()


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
    system to identify unusual weather-station observations,
    sensor anomalies and persistent anomaly events.
    """
)

st.divider()


# ============================================================
# OVERALL METRICS
# ============================================================


# ============================================================
# STEP 8 - OVERALL SYSTEM KPI & RISK SUMMARY
# ============================================================

TOTAL_RECORDS = 46_082_160
TOTAL_ANOMALIES = 1_056_291
ANOMALY_RATE = 2.29

NORMAL = 45_025_869
WARNING = 747_194
HIGH = 246_475
CRITICAL = 62_622

# ------------------------------------------------------------
# MAIN KPI CARDS
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📊 Records Analysed",
        f"{TOTAL_RECORDS:,}"
    )

with col2:
    st.metric(
        "🚨 Anomalies Detected",
        f"{TOTAL_ANOMALIES:,}"
    )

with col3:
    st.metric(
        "⚠️ Anomaly Rate",
        f"{ANOMALY_RATE:.2f}%"
    )
with col4:
    st.metric(
        "📍 Locations Monitored",
        f"{len(filtered_city):,}"
    )
# ------------------------------------------------------------
# RISK SUMMARY
# ------------------------------------------------------------

st.subheader("🚦 System Risk Status")

risk1, risk2, risk3, risk4 = st.columns(4)

with risk1:
    st.metric(
        "🟢 Normal",
        f"{NORMAL:,}"
    )

with risk2:
    st.metric(
        "🟡 Warning",
        f"{WARNING:,}"
    )

with risk3:
    st.metric(
        "🟠 High",
        f"{HIGH:,}"
    )

with risk4:
    st.metric(
        "🔴 Critical",
        f"{CRITICAL:,}"
    )
# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.header("🚨 Risk Distribution")

risk_df = pd.DataFrame({
    "Severity": [
        "Normal",
        "Warning",
        "High",
        "Critical"
    ],
    "Records": [
        45_025_869,
        WARNING,
        HIGH,
        CRITICAL
    ]
})

fig_risk = px.bar(
    risk_df,
    x="Severity",
    y="Records",
    title="Overall Anomaly Severity"
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)

# ============================================================
# STEP 9 - SENSOR INTELLIGENCE
# ============================================================

st.header("🧠 Sensor Intelligence")

# ------------------------------------------------------------
# FIND MOST IMPORTANT SENSOR
# ------------------------------------------------------------

top_sensor = sensor.sort_values(
    "anomaly_count",
    ascending=False
).iloc[0]

top_contribution = contribution.sort_values(
    "contribution_percent",
    ascending=False
).iloc[0]

# ------------------------------------------------------------
# SENSOR KPIs
# ------------------------------------------------------------

s1, s2, s3 = st.columns(3)

with s1:
    st.metric(
        "🚨 Most Anomalous Sensor",
        str(top_sensor["sensor"])
    )

with s2:
    st.metric(
        "📊 Anomaly Count",
        f"{int(top_sensor['anomaly_count']):,}"
    )

with s3:
    st.metric(
        "🤖 Highest Error Contribution",
        f"{top_contribution['contribution_percent']:.2f}%"
    )

# ------------------------------------------------------------
# AI INTERPRETATION
# ------------------------------------------------------------

st.subheader("🔍 AI Interpretation")

st.info(
    f"""
    **SkyGuard AI Analysis**

    The sensor showing the highest number of detected anomalies is
    **{top_sensor['sensor']}**.

    The sensor contributing the most to the autoencoder reconstruction
    error is **{top_contribution['feature']}**, with a contribution of
    **{top_contribution['contribution_percent']:.2f}%**.

    These sensors should receive higher priority during AWS
    maintenance and data-quality investigation.
    """
)
# ============================================================
# INDIA ANOMALY MAP
# ============================================================

st.header("🗺️ India Anomaly Map")

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
        zoom=4,
        height=650,
        title="SkyGuard V2 - Anomaly Locations",
        color_continuous_scale="Turbo"
    )

    fig_map.update_layout(
        map_style="open-street-map"
    )

    st.plotly_chart(
        fig_map,
        use_container_width=True
    )

else:

    st.warning(
        "No coordinate data available for the selected state."
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
# SENSOR ANALYSIS
# ============================================================

st.header("🌡️ Sensor Anomaly Analysis")

fig_sensor = px.bar(
    sensor,
    x="sensor",
    y="anomaly_count",
    title="Dominant Sensor in Anomalies"
)

fig_sensor.update_layout(
    xaxis_tickangle=-45
)

st.plotly_chart(
    fig_sensor,
    use_container_width=True
)


# ============================================================
# SENSOR CONTRIBUTION
# ============================================================

st.subheader(
    "🤖 Sensor Contribution to Reconstruction Error"
)

fig_contribution = px.bar(
    contribution,
    x="feature",
    y="contribution_percent",
    title="Sensor Contribution"
)

fig_contribution.update_layout(
    xaxis_tickangle=-45
)

st.plotly_chart(
    fig_contribution,
    use_container_width=True
)


# ============================================================
# STEP 10 - TEMPORAL INTELLIGENCE
# ============================================================

st.header("📅 Temporal Intelligence")

# ------------------------------------------------------------
# FIND PEAK MONTH AND HOUR
# ------------------------------------------------------------

peak_month = monthly.loc[
    monthly["anomaly_rate_percent"].idxmax()
]

peak_hour = hourly.loc[
    hourly["anomaly_rate_percent"].idxmax()
]

# ------------------------------------------------------------
# TEMPORAL KPIs
# ------------------------------------------------------------

t1, t2, t3 = st.columns(3)

with t1:
    st.metric(
        "📅 Highest-Risk Month",
        str(peak_month["month"])
    )

with t2:
    st.metric(
        "📈 Peak Monthly Rate",
        f"{peak_month['anomaly_rate_percent']:.2f}%"
    )

with t3:
    st.metric(
        "🕐 Highest-Risk Hour",
        f"{int(peak_hour['hour']):02d}:00"
    )

# ------------------------------------------------------------
# MONTHLY TREND
# ------------------------------------------------------------

st.subheader("📈 Monthly Anomaly Trend")

fig_month = px.line(
    monthly,
    x="month",
    y="anomaly_rate_percent",
    markers=True,
    title="Monthly Anomaly Rate"
)

st.plotly_chart(
    fig_month,
    use_container_width=True
)

# ------------------------------------------------------------
# HOURLY TREND
# ------------------------------------------------------------

st.subheader("⏰ Hourly Anomaly Pattern")

fig_hour = px.line(
    hourly,
    x="hour",
    y="anomaly_rate_percent",
    markers=True,
    title="Hourly Anomaly Rate"
)

st.plotly_chart(
    fig_hour,
    use_container_width=True
)

# ============================================================
# STEP 11 - PERSISTENT ANOMALY INTELLIGENCE
# ============================================================

st.header("⏱️ Persistent Anomaly Intelligence")

st.markdown(
    """
    Persistent anomalies are observations that remain anomalous
    across repeated weather-station measurements. These events
    are especially important because they may indicate sensor
    degradation, calibration problems or sustained abnormal
    environmental conditions.
    """
)

# ------------------------------------------------------------
# BASIC PERSISTENT EVENT ANALYSIS
# ------------------------------------------------------------

persistent_count = len(persistent)

# Try to identify useful columns automatically
persistent_columns = persistent.columns.tolist()

# ------------------------------------------------------------
# PERSISTENT ANOMALY KPI
# ------------------------------------------------------------

p1, p2, p3 = st.columns(3)

with p1:
    st.metric(
        "⏱️ Persistent Records",
        f"{persistent_count:,}"
    )

with p2:
    st.metric(
        "📋 Available Fields",
        f"{len(persistent_columns)}"
    )

with p3:
    st.metric(
        "📊 Top Events Displayed",
        "20"
    )

# ------------------------------------------------------------
# COLUMN INFORMATION
# ------------------------------------------------------------

st.subheader("📋 Persistent Event Data")

st.caption(
    "The table below contains the persistent anomaly records "
    "identified by the SkyGuard anomaly-detection pipeline."
)

st.dataframe(
    persistent.head(20),
    use_container_width=True,
    hide_index=True
)

# ------------------------------------------------------------
# NUMERIC ANALYSIS
# ------------------------------------------------------------

numeric_columns = persistent.select_dtypes(
    include="number"
).columns.tolist()

if len(numeric_columns) > 0:

    st.subheader("📊 Persistent Anomaly Statistics")

    stats = persistent[numeric_columns].describe().T

    st.dataframe(
        stats,
        use_container_width=True
    )

# ------------------------------------------------------------
# LOCATION ANALYSIS
# ------------------------------------------------------------

if "city" in persistent.columns:

    st.subheader("📍 Locations with Persistent Anomalies")

    persistent_city = (
        persistent["city"]
        .value_counts()
        .reset_index()
    )

    persistent_city.columns = [
        "city",
        "persistent_anomaly_records"
    ]

    persistent_city = persistent_city.head(15)

    fig_persistent_city = px.bar(
        persistent_city,
        x="city",
        y="persistent_anomaly_records",
        title="Top Locations by Persistent Anomaly Records"
    )

    fig_persistent_city.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig_persistent_city,
        use_container_width=True
    )

# ------------------------------------------------------------
# SENSOR ANALYSIS
# ------------------------------------------------------------

if "sensor" in persistent.columns:

    st.subheader("🌡️ Sensors Involved in Persistent Anomalies")

    persistent_sensor = (
        persistent["sensor"]
        .value_counts()
        .reset_index()
    )

    persistent_sensor.columns = [
        "sensor",
        "persistent_anomaly_records"
    ]

    fig_persistent_sensor = px.bar(
        persistent_sensor,
        x="sensor",
        y="persistent_anomaly_records",
        title="Persistent Anomalies by Sensor"
    )

    fig_persistent_sensor.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig_persistent_sensor,
        use_container_width=True
    )

# ------------------------------------------------------------
# AI INTERPRETATION
# ------------------------------------------------------------

st.subheader("🤖 Persistent Anomaly AI Insight")

st.info(
    f"""
    **SkyGuard AI Interpretation**

    SkyGuard identified **{persistent_count:,} persistent anomaly
    records** in the processed dataset.

    Persistent anomalies are important because repeated anomalous
    readings may indicate a sensor-quality problem rather than
    a single isolated measurement.

    These events can therefore be prioritized for AWS inspection,
    sensor calibration and data-quality verification.
    """
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
    # NETWORK HEALTH SCORE
    # --------------------------------------------------------

    health_score = max(
        0,
        min(
            100,
            100 - (ANOMALY_RATE * 5)
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

            {ANOMALY_RATE:.2f}%
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
    # SENSOR PRIORITY
    # --------------------------------------------------------

    st.subheader("🔧 Sensor Priority")

    if len(sensor) > 0:

        executive_sensor = sensor.sort_values(
            "anomaly_count",
            ascending=False
        ).iloc[0]

        sensor1, sensor2, sensor3 = st.columns(3)

        with sensor1:

            st.metric(
                "Most Anomalous Sensor",
                str(
                    executive_sensor["sensor"]
                )
            )

        with sensor2:

            st.metric(
                "Sensor Anomalies",
                f"{int(executive_sensor['anomaly_count']):,}"
            )

        with sensor3:

            if len(contribution) > 0:

                executive_feature = contribution.sort_values(
                    "contribution_percent",
                    ascending=False
                ).iloc[0]

                st.metric(
                    "Highest Error Contributor",
                    str(
                        executive_feature["feature"]
                    )
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
# STEP 12 - AI ALERT & ANOMALY EXPLANATION
# ============================================================

st.header("🚨 AI Alert & Anomaly Explanation")

st.markdown(
    """
    SkyGuard automatically identifies the locations with the
    highest anomaly rates and converts the detection results
    into an understandable operational alert.
    """
)

# ------------------------------------------------------------
# FIND MOST SEVERE LOCATION
# ------------------------------------------------------------

if len(filtered_city) > 0:

    severe_location = filtered_city.loc[
        filtered_city["anomaly_rate_percent"].idxmax()
    ]

    severe_city = str(severe_location["city"])
    severe_state = str(severe_location["state"])

    severe_rate = float(
        severe_location["anomaly_rate_percent"]
    )

    severe_anomalies = int(
        severe_location["anomalies"]
    )

    severe_score = float(
        severe_location["avg_score"]
    )

    # --------------------------------------------------------
    # DETERMINE ALERT LEVEL
    # --------------------------------------------------------

    if severe_rate >= 50:
        alert_level = "CRITICAL"
        alert_icon = "🔴"

    elif severe_rate >= 25:
        alert_level = "HIGH"
        alert_icon = "🟠"

    elif severe_rate >= 10:
        alert_level = "WARNING"
        alert_icon = "🟡"

    else:
        alert_level = "NORMAL"
        alert_icon = "🟢"

    # --------------------------------------------------------
    # ALERT KPI CARDS
    # --------------------------------------------------------

    a1, a2, a3, a4 = st.columns(4)

    with a1:
        st.metric(
            "🚨 Alert Level",
            alert_level
        )

    with a2:
        st.metric(
            "📍 Location",
            severe_city
        )

    with a3:
        st.metric(
            "📊 Anomaly Rate",
            f"{severe_rate:.2f}%"
        )

    with a4:
        st.metric(
            "🤖 Anomaly Score",
            f"{severe_score:.3f}"
        )

    # --------------------------------------------------------
    # ALERT MESSAGE
    # --------------------------------------------------------

    if alert_level == "CRITICAL":

        st.error(
            f"""
            🔴 **CRITICAL ALERT**

            **{severe_city}, {severe_state}** has the highest
            detected anomaly rate in the selected region.

            **Anomaly rate:** {severe_rate:.2f}%

            **Anomalous observations:** {severe_anomalies:,}

            The location should receive immediate investigation
            for possible sensor malfunction, calibration problems
            or sustained abnormal environmental conditions.
            """
        )

    elif alert_level == "HIGH":

        st.warning(
            f"""
            🟠 **HIGH-RISK ALERT**

            **{severe_city}, {severe_state}** shows a high level
            of anomalous weather-station observations.

            **Anomaly rate:** {severe_rate:.2f}%

            **Anomalous observations:** {severe_anomalies:,}

            Priority inspection and data-quality verification
            are recommended.
            """
        )

    elif alert_level == "WARNING":

        st.warning(
            f"""
            🟡 **WARNING**

            **{severe_city}, {severe_state}** shows an elevated
            anomaly rate of **{severe_rate:.2f}%**.

            Continued monitoring is recommended.
            """
        )

    else:

        st.success(
            f"""
            🟢 **NORMAL**

            No high-risk location is currently identified in
            the selected region.

            The highest anomaly rate is **{severe_rate:.2f}%**
            at **{severe_city}, {severe_state}**.
            """
        )

    # --------------------------------------------------------
    # RECOMMENDED ACTION
    # --------------------------------------------------------

    st.subheader("🛠️ Recommended Action")

    if alert_level == "CRITICAL":

        action = """
        1. Inspect the affected AWS immediately.
        2. Verify the sensor responsible for abnormal readings.
        3. Check sensor calibration and physical connections.
        4. Compare readings with nearby weather stations.
        5. Continue monitoring after maintenance.
        """

    elif alert_level == "HIGH":

        action = """
        1. Schedule AWS inspection.
        2. Review the affected sensor readings.
        3. Compare observations with nearby stations.
        4. Check for repeated anomalous measurements.
        """

    elif alert_level == "WARNING":

        action = """
        1. Continue monitoring the location.
        2. Review recent sensor measurements.
        3. Investigate if the anomaly rate continues increasing.
        """

    else:

        action = """
        1. Continue normal monitoring.
        2. Periodically verify sensor data quality.
        3. Investigate only if anomaly rates increase.
        """

    st.info(action)

else:

    st.success(
        "No locations are available for AI alert analysis."
    )

# ============================================================
# STEP 13 - INTERACTIVE ANOMALY INVESTIGATION
# ============================================================

st.header("🔎 Interactive Anomaly Investigation")

st.markdown(
    """
    Select a location to investigate its anomaly behaviour in detail.
    SkyGuard provides location-specific anomaly statistics and
    operational recommendations.
    """
)

# ------------------------------------------------------------
# CITY SELECTION
# ------------------------------------------------------------

if len(filtered_city) > 0:

    available_cities = sorted(
        filtered_city["city"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_city = st.selectbox(
        "📍 Select City for Investigation",
        available_cities
    )

    # --------------------------------------------------------
    # SELECT CITY DATA
    # --------------------------------------------------------

    selected_location = filtered_city[
        filtered_city["city"] == selected_city
    ]

    if len(selected_location) > 0:

        location = selected_location.iloc[0]

        city_state = str(location["state"])
        city_name = str(location["city"])

        city_records = int(
            location["total_records"]
        )

        city_anomalies = int(
            location["anomalies"]
        )

        city_score = float(
            location["avg_score"]
        )

        city_rate = float(
            location["anomaly_rate_percent"]
        )

        # ----------------------------------------------------
        # LOCATION KPIs
        # ----------------------------------------------------

        st.subheader(
            f"📍 Investigation: {city_name}, {city_state}"
        )

        i1, i2, i3, i4 = st.columns(4)

        with i1:
            st.metric(
                "📊 Records",
                f"{city_records:,}"
            )

        with i2:
            st.metric(
                "🚨 Anomalies",
                f"{city_anomalies:,}"
            )

        with i3:
            st.metric(
                "⚠️ Anomaly Rate",
                f"{city_rate:.2f}%"
            )

        with i4:
            st.metric(
                "🤖 Avg Anomaly Score",
                f"{city_score:.3f}"
            )

        # ----------------------------------------------------
        # LOCATION RISK CLASSIFICATION
        # ----------------------------------------------------

        if city_rate >= 50:

            location_risk = "🔴 CRITICAL"

        elif city_rate >= 25:

            location_risk = "🟠 HIGH"

        elif city_rate >= 10:

            location_risk = "🟡 WARNING"

        else:

            location_risk = "🟢 NORMAL"

        st.subheader("🚦 Location Risk Classification")

        st.info(
            f"""
            **{city_name}, {city_state}**

            Current SkyGuard classification:
            **{location_risk}**

            Detected anomaly rate:
            **{city_rate:.2f}%**
            """
        )

        # ----------------------------------------------------
        # ANOMALY VS NORMAL
        # ----------------------------------------------------

        st.subheader("📊 Normal vs Anomalous Observations")

        normal_records = max(
            city_records - city_anomalies,
            0
        )

        city_distribution = pd.DataFrame({
            "Category": [
                "Normal",
                "Anomalous"
            ],
            "Records": [
                normal_records,
                city_anomalies
            ]
        })

        fig_city_distribution = px.pie(
            city_distribution,
            names="Category",
            values="Records",
            title=f"Observation Distribution — {city_name}"
        )

        st.plotly_chart(
            fig_city_distribution,
            use_container_width=True
        )

        # ----------------------------------------------------
        # LOCATION DETAILS
        # ----------------------------------------------------

        st.subheader("📋 Location Details")

        location_details = pd.DataFrame({
            "Metric": [
                "State",
                "City",
                "Total Records",
                "Anomalies",
                "Anomaly Rate (%)",
                "Average Anomaly Score"
            ],
            "Value": [
                city_state,
                city_name,
                f"{city_records:,}",
                f"{city_anomalies:,}",
                f"{city_rate:.2f}",
                f"{city_score:.3f}"
            ]
        })

        st.dataframe(
            location_details,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # CITY RECOMMENDATION
        # ----------------------------------------------------

        st.subheader("🤖 Investigation Recommendation")

        if location_risk == "🔴 CRITICAL":

            recommendation = """
            🔴 **Immediate inspection recommended.**

            Check AWS sensor calibration, physical connections,
            recent maintenance history and consistency with
            nearby weather stations.
            """

        elif location_risk == "🟠 HIGH":

            recommendation = """
            🟠 **Priority inspection recommended.**

            Review the affected station's sensor readings and
            investigate repeated anomalous observations.
            """

        elif location_risk == "🟡 WARNING":

            recommendation = """
            🟡 **Increased monitoring recommended.**

            Continue observing the station and investigate if
            the anomaly rate continues to increase.
            """

        else:

            recommendation = """
            🟢 **Normal monitoring recommended.**

            No immediate intervention is indicated from the
            current location-level anomaly rate.
            """

        st.info(recommendation)

else:

    st.warning(
        "No cities are available for the selected state."
    )

# ============================================================
# END OF STEP 13
# ============================================================

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SkyGuard V2 • AI/ML Weather Station Anomaly Detection • "
    "Intel Arc XPU Accelerated"
)
