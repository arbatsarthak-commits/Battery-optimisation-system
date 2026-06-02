
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
try:
    from streamlit_autorefresh import st_autorefresh
except ModuleNotFoundError:  # fallback for environments where dependency is not installed yet
    st_autorefresh = None

from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------------
# Page config + global theme
# ---------------------------------------------------------------------------------
st.set_page_config(
    page_title="EV Battery Optimization Dashboard | CPS",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Dark theme adjustments */
    html, body, [class*="stApp"] {
        background: #0b1220;
        color: #e8eefc;
    }
    .st-ae, .st-bq, .st-df {
        background: transparent;
    }
    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    }
    .metric-title { color: #b8c6ff; font-weight: 700; font-size: .95rem; }
    .metric-value { font-size: 2rem; font-weight: 800; letter-spacing: .2px; margin-top: .25rem; }
    .metric-sub { color: rgba(232,238,252,0.75); font-size: .9rem; margin-top: .25rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.03);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Section header */
    .section-title {
        font-size: 1.4rem;
        font-weight: 900;
        color: #cfe0ff;
        margin-top: 1.2rem;
        margin-bottom: .5rem;
    }

    /* Info boxes */
    .info-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 4px solid #38bdf8;
        padding: 1rem 1.1rem;
        border-radius: 12px;
        color: rgba(232,238,252,0.92);
    }

    /* Alert boxes */
    .alert-good { border-left: 6px solid #22c55e; background: rgba(34,197,94,0.12); }
    .alert-mod { border-left: 6px solid #f59e0b; background: rgba(245,158,11,0.12); }
    .alert-bad { border-left: 6px solid #ef4444; background: rgba(239,68,68,0.12); }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------------
# Data loading / column detection
# ---------------------------------------------------------------------------------

DATA_PATH = r"D:\software\CPS PROJECT\battery_data.csv"


def _normalize_colname(c: str) -> str:
    return "".join(ch.lower() for ch in str(c).strip() if ch.isalnum() or ch in ["_", "-", " "]).replace("-", "_")


@dataclass
class ColumnMapping:
    time: Optional[str]
    voltage: Optional[str]
    current: Optional[str]
    temperature: Optional[str]


def detect_columns(df: pd.DataFrame) -> ColumnMapping:
    cols = list(df.columns)
    norm = {_c: _normalize_colname(_c) for _c in cols}

    def find_best(candidates: List[str]) -> Optional[str]:
        # Exact match priority, then partial contains
        for cand in candidates:
            for col, n in norm.items():
                if n == cand:
                    return col
        for cand in candidates:
            for col, n in norm.items():
                if cand in n:
                    return col
        return None

    time = find_best(["time", "t", "times", "time_s", "time_seconds", "seconds", "timestamp", "time(s)"])
    voltage = find_best(["voltage", "v", "vbat", "battery_voltage", "terminal_voltage", "u", "volts"])
    current = find_best(["current", "i", "ibat", "charge_current", "discharge_current", "amp", "amps", "current_a", "i_a"])
    temperature = find_best(["temperature", "temp", "tbat", "battery_temperature", "therm", "degc", "temp_c"])

    # If voltage/current columns are not found, try numeric-position heuristics
    if voltage is None or current is None:
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 2:
            # Assume first numeric ~ time, second ~ voltage, third ~ current
            if voltage is None and len(numeric_cols) >= 2:
                voltage = numeric_cols[1]
            if current is None and len(numeric_cols) >= 3:
                current = numeric_cols[2]

    return ColumnMapping(time=time, voltage=voltage, current=current, temperature=temperature)


def load_battery_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    # Try reading with header inference first.
    # Dataset appears to have 3 columns in your open tab preview (no header). We'll handle both.
    try:
        df_try = pd.read_csv(path)
        if df_try.shape[1] >= 3:
            return df_try
    except Exception:
        pass

    # Fallback: no header, generic columns.
    df = pd.read_csv(path, header=None)
    # Assign default names based on number of columns.
    if df.shape[1] == 3:
        df.columns = ["Time", "Voltage", "Current"]
    elif df.shape[1] == 4:
        df.columns = ["Time", "Voltage", "Current", "Temperature"]
    else:
        # Generic names
        df.columns = [f"Col_{i}" for i in range(df.shape[1])]
    return df


@st.cache_data(show_spinner=False)
def load_and_prepare() -> Tuple[pd.DataFrame, ColumnMapping]:
    df_raw = load_battery_csv(DATA_PATH)

    # Ensure time axis exists; if missing, synthesize sequential index as time (sec)
    mapping = detect_columns(df_raw)

    df = df_raw.copy()
    # Coerce to numeric when possible
    for c in [mapping.time, mapping.voltage, mapping.current, mapping.temperature]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if mapping.time is None:
        df["Time_s"] = np.arange(len(df), dtype=float)
        mapping = ColumnMapping(time="Time_s", voltage=mapping.voltage, current=mapping.current, temperature=mapping.temperature)

    # If voltage/current still missing, stop gracefully
    if mapping.voltage is None or mapping.current is None:
        raise ValueError(
            "Could not auto-detect Voltage and/or Current columns in the dataset. "
            "Please ensure the CSV contains recognizable columns for Voltage and Current."
        )

    # Drop rows where essential columns are missing
    essential = [mapping.time, mapping.voltage, mapping.current]
    df = df.dropna(subset=essential).reset_index(drop=True)

    # Derived metrics
    df["Power_W"] = df[mapping.voltage] * df[mapping.current]

    # Resistance = V / I, handle I=0
    df["Resistance_Ohm"] = np.where(df[mapping.current].abs() > 1e-12, df[mapping.voltage] / df[mapping.current], np.nan)

    # Energy (J) from power over time if time is monotonic-ish
    t = df[mapping.time].to_numpy(dtype=float)
    p = df["Power_W"].to_numpy(dtype=float)

    # trapezoid integration; if time is decreasing, sort
    if len(t) >= 2 and np.nanmin(np.diff(t)) < 0:
        order = np.argsort(t)
        t_sorted = t[order]
        p_sorted = p[order]
        energy_j = float(np.trapezoid(p_sorted, t_sorted))
    else:
        energy_j = float(np.trapezoid(p, t))

    # Cumulative energy (for charts)
    sort_idx = np.argsort(t)
    df_sorted = df.iloc[sort_idx].copy()
    df_sorted["Energy_J"] = np.nan
    # cumulative trapezoid
    cumulative = [0.0]
    for i in range(1, len(df_sorted)):
        dt = df_sorted.iloc[i][mapping.time] - df_sorted.iloc[i - 1][mapping.time]
        cumulative.append(cumulative[-1] + 0.5 * (df_sorted.iloc[i]["Power_W"] + df_sorted.iloc[i - 1]["Power_W"]) * dt)
    df_sorted["Energy_J"] = cumulative

    # Restore original order indices
    df.loc[sort_idx, "Energy_J"] = df_sorted["Energy_J"].to_numpy()

    # Efficiency: no explicit charge/discharge energy columns.
    # Use a pragmatic proxy: ratio of mean positive power to mean absolute power.
    # This stays dataset-driven and avoids fabricating extra columns.
    p_pos = np.nanmean(np.where(df["Power_W"] > 0, df["Power_W"], np.nan))
    p_abs = np.nanmean(np.abs(df["Power_W"]))
    efficiency_proxy = float(p_pos / p_abs) if (p_abs and p_abs > 1e-12) else np.nan
    df["Efficiency_proxy"] = efficiency_proxy

    # Peak power timestamp
    # (computed later in UI using sorted arrays)

    return df, mapping


# ---------------------------------------------------------------------------------
# SOC estimation
# ---------------------------------------------------------------------------------


def estimate_soc(df: pd.DataFrame, mapping: ColumnMapping) -> pd.Series:
    """SOC estimation without explicit capacity/state labels.

    Strategy:
    - If voltage exists: normalize voltage between observed min/max.
    - Bound to [0, 1].

    This is a heuristic suitable for academic dashboards when capacity label is missing.
    """
    v = df[mapping.voltage].to_numpy(dtype=float)
    v_min = np.nanpercentile(v, 2)
    v_max = np.nanpercentile(v, 98)

    if not np.isfinite(v_min) or not np.isfinite(v_max) or abs(v_max - v_min) < 1e-9:
        # Fallback: index-based linear SOC
        soc = np.linspace(0, 1, num=len(df), dtype=float)
    else:
        soc = (v - v_min) / (v_max - v_min)

    soc = np.clip(soc, 0.0, 1.0)
    return pd.Series(soc, index=df.index, name="SOC")


# ---------------------------------------------------------------------------------
# SOH prediction (ML)
# ---------------------------------------------------------------------------------


def _derive_health_target_proxy(df: pd.DataFrame, mapping: ColumnMapping) -> pd.Series:
    """Create a proxy SOH target from dataset signals.

    Since the dataset typically doesn't include ground-truth SOH labels,
    we compute a health proxy using:
    - Internal resistance trend (higher resistance => lower health)
    - Voltage stability proxy

    Returns target in [0, 1].
    """
    # Resistance proxy
    r = df["Resistance_Ohm"].to_numpy(dtype=float)
    r_finite = r[np.isfinite(r)]
    if len(r_finite) < 5:
        r_health = np.ones(len(df), dtype=float)
    else:
        r_min = np.nanpercentile(r_finite, 5)
        r_max = np.nanpercentile(r_finite, 95)
        if abs(r_max - r_min) < 1e-12:
            r_norm = np.zeros(len(df), dtype=float)
        else:
            r_norm = (r - r_min) / (r_max - r_min)
        # higher resistance => lower health
        r_health = 1.0 - np.clip(r_norm, 0.0, 1.0)

    # Voltage proxy (stability): smaller deviation => healthier
    v = df[mapping.voltage].to_numpy(dtype=float)
    v_med = np.nanmedian(v)
    v_dev = np.abs(v - v_med)
    v_dev_scale = np.nanpercentile(v_dev, 90) if np.isfinite(v_dev).any() else 1.0
    if v_dev_scale < 1e-12:
        v_health = np.ones(len(df), dtype=float)
    else:
        v_health = 1.0 - np.clip(v_dev / v_dev_scale, 0.0, 1.0)

    # Combine proxies
    target = 0.65 * r_health + 0.35 * v_health
    target = np.clip(target, 0.0, 1.0)
    return pd.Series(target, index=df.index, name="SOH_target_proxy")


def train_soh_model(df: pd.DataFrame, mapping: ColumnMapping) -> Tuple[object, List[str]]:
    features = [mapping.voltage, mapping.current, "Power_W"]
    if mapping.temperature is not None:
        features.append(mapping.temperature)

    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    y = _derive_health_target_proxy(df, mapping)

    # Drop rows with NaN in features/target
    valid = X.notna().all(axis=1) & y.notna()
    Xv = X.loc[valid]
    yv = y.loc[valid]

    if len(Xv) < 30:
        # Too little data; fallback to linear regression on whatever is available
        model = LinearRegression()
        model.fit(Xv, yv)
        return model, features

    # Random forest first
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    rf.fit(Xv, yv)

    # If model outputs are extremely flat, fallback to linear
    pred = rf.predict(Xv)
    if np.nanstd(pred) < 1e-3:
        model = LinearRegression()
        model.fit(Xv, yv)
        return model, features

    return rf, features


def predict_soh(df: pd.DataFrame, mapping: ColumnMapping, model: object, features: List[str]) -> pd.Series:
    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1)
    soh = np.full(len(df), np.nan, dtype=float)
    soh[valid] = model.predict(X.loc[valid])
    soh = np.clip(soh, 0.0, 1.0)
    return pd.Series(soh, index=df.index, name="SOH")


def soh_condition(soh_pct: float) -> str:
    if soh_pct >= 80:
        return "GOOD"
    if soh_pct >= 60:
        return "MODERATE"
    return "CRITICAL"


# ---------------------------------------------------------------------------------
# Fault detection
# ---------------------------------------------------------------------------------


def detect_faults(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    soc: pd.Series,
    soh: pd.Series,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Returns:
    - alerts_df: per-row flags
    - latest_alerts: list of alert dicts for latest row
    """

    temperature_available = mapping.temperature is not None

    t = df[mapping.time]
    v = df[mapping.voltage]
    i = df[mapping.current]
    temp = df[mapping.temperature] if temperature_available else None

    # Use quantile-based thresholds to stay dataset-driven.
    v_hi = np.nanpercentile(v, 98)
    v_lo = np.nanpercentile(v, 2)
    i_hi = np.nanpercentile(np.abs(i), 98)

    # Overheating threshold if temperature exists
    if temperature_available:
        temp_hi = np.nanpercentile(temp, 98)
    else:
        temp_hi = np.nan

    # overcharging: high voltage + positive current (charging convention may vary)
    # We'll interpret i>0 as charging if power>0.
    charging_mask = df["Power_W"] > 0

    # voltage abnormal: outside normal quantiles
    voltage_abnormal = (v > v_hi) | (v < v_lo)

    # high current: magnitude too high
    high_current = np.abs(i) > i_hi

    # overheating
    overheating = pd.Series(False, index=df.index)
    if temperature_available:
        overheating = temp > temp_hi

    # overcharging
    overcharging = pd.Series(False, index=df.index)
    overcharging = ((v > v_hi) & charging_mask)

    # resistance spike (extra fault)
    r = df["Resistance_Ohm"]
    r_hi = np.nanpercentile(r[np.isfinite(r)], 98) if np.isfinite(r).any() else np.nan
    resistance_abnormal = pd.Series(False, index=df.index)
    if np.isfinite(r_hi):
        resistance_abnormal = r > r_hi

    alerts_df = pd.DataFrame(
        {
            "overheating": overheating,
            "overcharging": overcharging,
            "voltage_abnormal": voltage_abnormal,
            "high_current": high_current,
            "resistance_abnormal": resistance_abnormal,
        },
        index=df.index,
    )

    latest_idx = df.index[-1]
    latest_alerts: List[Dict] = []

    soh_pct = float(soh.loc[latest_idx] * 100.0) if pd.notna(soh.loc[latest_idx]) else np.nan
    soc_pct = float(soc.loc[latest_idx] * 100.0) if pd.notna(soc.loc[latest_idx]) else np.nan

    def add_if(flag: bool, title: str, severity: str, detail: str):
        if bool(flag):
            latest_alerts.append(
                {
                    "title": title,
                    "severity": severity,
                    "detail": detail,
                    "soh_pct": soh_pct,
                    "soc_pct": soc_pct,
                }
            )

    add_if(alerts_df.loc[latest_idx, "overheating"], "Overheating", "ERROR", f"Temperature exceeded threshold ({temp_hi:.2f} K/C proxy).")
    add_if(alerts_df.loc[latest_idx, "overcharging"], "Overcharging Risk", "WARNING", "High voltage detected while charging condition is active.")
    add_if(alerts_df.loc[latest_idx, "voltage_abnormal"], "Voltage Abnormality", "WARNING", f"Voltage outside normal range [{v_lo:.3f}, {v_hi:.3f}].")
    add_if(alerts_df.loc[latest_idx, "high_current"], "High Current", "WARNING", f"|Current| above threshold (p98): {i_hi:.3f} A (abs).")
    add_if(alerts_df.loc[latest_idx, "resistance_abnormal"], "Resistance Spike", "WARNING", "Internal resistance proxy increased significantly.")

    # If none
    if not latest_alerts:
        latest_alerts.append(
            {
                "title": "System Status",
                "severity": "GOOD",
                "detail": "No faults detected at the latest sample using dataset-driven thresholds.",
                "soh_pct": soh_pct,
                "soc_pct": soc_pct,
            }
        )

    return alerts_df, latest_alerts


# ---------------------------------------------------------------------------------
# Smart charging recommendation
# ---------------------------------------------------------------------------------


def recommend_charging(
    soc: pd.Series,
    soh: pd.Series,
    df: pd.DataFrame,
    mapping: ColumnMapping,
) -> Dict:
    latest = df.index[-1]
    soc_pct = float(soc.loc[latest] * 100.0) if pd.notna(soc.loc[latest]) else np.nan
    soh_pct = float(soh.loc[latest] * 100.0) if pd.notna(soh.loc[latest]) else np.nan

    temperature_available = mapping.temperature is not None
    temp_latest = float(df[mapping.temperature].loc[latest]) if temperature_available else np.nan

    # Thresholds dataset-driven
    v = df[mapping.voltage]
    v_hi = np.nanpercentile(v, 98)

    temp_hi = np.nan
    if temperature_available:
        temp_hi = float(np.nanpercentile(df[mapping.temperature], 98))

    # Determine current direction from power
    charging_active = bool(df["Power_W"].loc[latest] > 0)

    reasons: List[str] = []

    # Stop charging if too hot
    if temperature_available and np.isfinite(temp_hi) and temp_latest > temp_hi:
        reasons.append(f"Temperature too high (latest={temp_latest:.2f} > p98={temp_hi:.2f}).")
        return {
            "mode": "STOP",
            "recommendation": "Stop charging to prevent thermal runaway risk.",
            "optimization_status": "CRITICAL",
            "reasons": reasons,
        }

    # Reduce rate if SOH critical
    if np.isfinite(soh_pct) and soh_pct < 60:
        reasons.append(f"SOH is low ({soh_pct:.1f}%).")
        return {
            "mode": "SLOW",
            "recommendation": "Use slow charging / conservative current limits due to poor battery health.",
            "optimization_status": "SAFE (HEALTH LIMITED)",
            "reasons": reasons,
        }

    # Fast if SOC is low
    if np.isfinite(soc_pct) and soc_pct < 20:
        reasons.append(f"SOC is low ({soc_pct:.1f}%).")
        return {
            "mode": "FAST",
            "recommendation": "Fast charging is recommended because SOC is below 20% and thermal constraints are satisfied.",
            "optimization_status": "ENERGY-OPTIMIZED",
            "reasons": reasons,
        }

    # If high voltage near upper bound, avoid overcharging
    if float(df[mapping.voltage].loc[latest]) > v_hi:
        reasons.append("Voltage near upper quantile bound; reduce charging rate to avoid overcharge.")
        return {
            "mode": "TRICKLE",
            "recommendation": "Use trickle/float charging to reduce overcharge risk near voltage limit.",
            "optimization_status": "VOLTAGE-CONTROLLED",
            "reasons": reasons,
        }

    # Default
    if charging_active:
        reasons.append("Conditions appear stable based on latest sensor values.")
        return {
            "mode": "BALANCED",
            "recommendation": "Balanced charging strategy recommended (monitor voltage/current/temperature continuously).",
            "optimization_status": "STABLE",
            "reasons": reasons,
        }
    return {
        "mode": "STANDBY",
        "recommendation": "Charging not active. Maintain readiness and continue monitoring.",
        "optimization_status": "STANDBY",
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------------
# Remaining range prediction
# ---------------------------------------------------------------------------------


def predict_remaining_distance(
    soc: pd.Series,
    df: pd.DataFrame,
) -> float:
    """Estimate remaining driving range (km).

    Model:
        range_km = base_range_km * SOC * efficiency_scaling

    efficiency_scaling uses dataset-driven power proxy.
    """
    latest = df.index[-1]
    soc_pct = float(soc.loc[latest]) if pd.notna(soc.loc[latest]) else np.nan

    # Efficiency scaling from power proxy stored in dataframe
    eff = float(df["Efficiency_proxy"].iloc[0]) if "Efficiency_proxy" in df.columns else np.nan
    if not np.isfinite(eff):
        eff = 0.85
    eff = float(np.clip(eff, 0.1, 1.0))

    base_range_km = 300.0  # assumption for dashboard; documented and not dataset-generated
    dist = base_range_km * soc_pct * eff
    return float(np.clip(dist, 0.0, base_range_km))


# ---------------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------------


def plot_time_series(df: pd.DataFrame, x_col: str, y_col: str, title: str, color: str, yaxis_title: str):
    fig = px.line(df, x=x_col, y=y_col, title=title, color_discrete_sequence=[color])
    fig.update_traces(mode="lines", line_width=2)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title=yaxis_title,
        margin=dict(l=30, r=20, t=60, b=30),
    )
    return fig


def soc_gauge(soc_pct: float) -> go.Figure:
    value = float(np.clip(soc_pct, 0, 100))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 26, "color": "#e8eefc"}},
            title={"text": "State of Charge (SOC)", "font": {"size": 16, "color": "#b8c6ff"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#cfe0ff"},
                "bar": {"color": "#38bdf8"},
                "bgcolor": "rgba(255,255,255,0.08)",
                "steps": [
                    {"range": [0, 20], "color": "rgba(239,68,68,0.22)"},
                    {"range": [20, 60], "color": "rgba(245,158,11,0.20)"},
                    {"range": [60, 100], "color": "rgba(34,197,94,0.18)"},
                ],
            },
        )
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
    return fig


def health_badge_html(condition: str) -> str:
    if condition == "GOOD":
        cls = "alert-good"
    elif condition == "MODERATE":
        cls = "alert-mod"
    else:
        cls = "alert-bad"

    return f"""
    <div class="info-box {cls}">
        <div style="font-weight:900;font-size:1.05rem;">SOH Condition: {condition}</div>
    </div>
    """


# ---------------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------------


def metric_card(title: str, value: str, subtitle: str):
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtitle}</div>
    </div>
    """


def main():
    # LIVE CPS feel: true real-time behavior.
    # Prefer streamlit-autorefresh if available.
    if st_autorefresh is not None:
        st_autorefresh(interval=2000, limit=None, key="ev_battery_autorefresh")

    # Synchronized live cursor (single source of truth for ALL live components)
    if "cursor" not in st.session_state:
        st.session_state.cursor = 0

    # Auto-advance cursor each refresh to simulate real telemetry arrival.
    # We will clamp using dataset length after we load df.






    # Time display (helps demonstrate “live” behavior even when auto-refresh is disabled)
    from datetime import datetime
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    if st_autorefresh is None:
        # Minimal fallback: allow user to trigger refresh manually; rerun loop not used.
        st.info("Auto-refresh disabled (streamlit-autorefresh not installed). Install dependencies and refresh again.")


    st.markdown(
        """
        <div style="padding: 1.25rem; border-radius: 18px; background: linear-gradient(90deg, rgba(56,189,248,0.18), rgba(167,139,250,0.18)); border: 1px solid rgba(255,255,255,0.08);">
            <div style="font-size: 2rem; font-weight: 1000; letter-spacing: .3px;">🔋 EV Battery Monitoring & Optimization Dashboard</div>
            <div style="color: rgba(232,238,252,0.8); font-weight: 600; margin-top: .25rem;">
                Industry-style analytics • SOC/SOH estimation • Fault detection • Smart charging • CPS digital twin workflow
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load data + compute derived fields
    # NOTE: For “LIVE” behavior, we recompute SOC/SOH/alerts every app rerun,
    # and the auto-refresh triggers reruns.
    with st.spinner("Loading and preparing battery dataset..."):
        df, mapping = load_and_prepare()



    # Compute SOC / SOH / Alerts
    # NOTE: These are computed for the entire dataset, but UI values are indexed by `cursor`.
    with st.spinner("Computing SOC, SOH, and alerts..."):
        soc = estimate_soc(df, mapping)
        df["SOC"] = soc

        model, features = train_soh_model(df, mapping)
        soh = predict_soh(df, mapping, model, features)
        df["SOH"] = soh

        alerts_df, latest_alerts = detect_faults(df, mapping, soc=soc, soh=soh)

        charging_rec = recommend_charging(soc=df["SOC"], soh=df["SOH"], df=df, mapping=mapping)
        remaining_km = predict_remaining_distance(df["SOC"], df)

        # Synchronization cursor clamp (single source of truth)
        cursor = int(st.session_state.cursor)
        cursor = max(0, min(cursor, len(df) - 1))
        st.session_state.cursor = cursor

        latest = cursor

    # Advance cursor for next refresh (simulate new telemetry sample arrival)
    # (We clamp after we advance on the next run.)
    st.session_state.cursor = min(st.session_state.cursor + 1, len(df) - 1)

    # Shared streaming window (all charts must use this slice)
    WINDOW = 200
    start = max(0, latest - WINDOW)
    df_window = df.iloc[start : latest + 1]


    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        (
            "Overview",
            "Battery Analytics",
            "SOC",
            "SOH (ML Prediction)",
            "Fault Detection",
            "Smart Charging",
            "Remaining Distance",
            "CPS Digital Twin",
            "Report",
            "Raw Data",
        ),
        index=0,
    )

    x_col = mapping.time
    v_col = mapping.voltage
    i_col = mapping.current
    t_col = mapping.temperature

    # Common metrics (LIVE window aware where appropriate)
    max_power = float(df_window["Power_W"].max())
    energy_total_j = (
        float(df_window["Energy_J"].iloc[np.nanargmax(df_window["Energy_J"].to_numpy())])
        if "Energy_J" in df_window.columns
        else float(np.nan)
    )
    avg_eff = float(df_window["Efficiency_proxy"].iloc[0]) if "Efficiency_proxy" in df_window.columns else np.nan

    latest_soc_pct = float(df["SOC"].iloc[latest] * 100.0)
    latest_soh_pct = float(df["SOH"].iloc[latest] * 100.0) if pd.notna(df["SOH"].iloc[latest]) else np.nan

    latest_condition = soh_condition(latest_soh_pct) if np.isfinite(latest_soh_pct) else "UNKNOWN"

    # ---------------------------------------------------------------------------------
    # Pages
    # ---------------------------------------------------------------------------------
    if page == "Overview":
        st.markdown("<div class='section-title'>Executive Metrics</div>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("Max Power", f"{max_power:.2f}", "W"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Total Energy", f"{float(df_window['Energy_J'].iloc[-1])/3.6e6:.2f}", "kWh (approx)"), unsafe_allow_html=True)

        with c3:
            st.markdown(metric_card("SOC", f"{latest_soc_pct:.1f}", "%"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("SOH", f"{latest_soh_pct:.1f}", f"({latest_condition})"), unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Latest System Alerts</div>", unsafe_allow_html=True)

        for a in latest_alerts:
            sev = a.get("severity")
            if sev == "GOOD":
                st.markdown(
                    f"<div class='info-box alert-good'><b>{a['title']}</b><div style='margin-top:.35rem;color:rgba(232,238,252,0.85)'>{a['detail']}</div></div>",
                    unsafe_allow_html=True,
                )
            elif sev == "ERROR":
                st.markdown(
                    f"<div class='info-box alert-bad'><b>{a['title']}</b><div style='margin-top:.35rem;color:rgba(232,238,252,0.85)'>{a['detail']}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='info-box alert-mod'><b>{a['title']}</b><div style='margin-top:.35rem;color:rgba(232,238,252,0.85)'>{a['detail']}</div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div class='section-title'>Quick Charging Recommendation</div>", unsafe_allow_html=True)
        st.info(f"Mode: **{charging_rec['mode']}** | Optimization: **{charging_rec['optimization_status']}**\n\nRecommendation: {charging_rec['recommendation']}" )

        if charging_rec["reasons"]:
            st.caption("Reasons: " + " | ".join(charging_rec["reasons"]))

    elif page == "Battery Analytics":
        st.markdown("<div class='section-title'>Battery Analytics (Power, Resistance, Energy, Efficiency)</div>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("Average Voltage", f"{df[v_col].mean():.3f}", "V"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Average Current", f"{df[i_col].mean():.3f}", "A"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Average Efficiency", f"{avg_eff*100:.1f}%" if np.isfinite(avg_eff) else "N/A", "proxy"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("Energy (J)", f"{float(df['Energy_J'].iloc[-1]):.2e}", "approx"), unsafe_allow_html=True)

        # Interactive plots (LIVE window)
        fig_v = plot_time_series(df_window, x_col=x_col, y_col=v_col, title="Voltage vs Time (live)", color="#c55a11", yaxis_title="Voltage (V)")
        fig_i = plot_time_series(df_window, x_col=x_col, y_col=i_col, title="Current vs Time (live)", color="#1f4e79", yaxis_title="Current (A)")
        fig_p = plot_time_series(df_window, x_col=x_col, y_col="Power_W", title="Power vs Time (live)", color="#548235", yaxis_title="Power (W)")


        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(fig_v, use_container_width=True)
        with r2:
            st.plotly_chart(fig_i, use_container_width=True)

        st.plotly_chart(fig_p, use_container_width=True)

        # Resistance plot if present
        if "Resistance_Ohm" in df.columns and df["Resistance_Ohm"].notna().any():
            fig_r = plot_time_series(df_window, x_col=x_col, y_col="Resistance_Ohm", title="Resistance (V/I) vs Time (live)", color="#8e44ad", yaxis_title="Resistance (Ω)")
            st.plotly_chart(fig_r, use_container_width=True)


    elif page == "SOC":
        st.markdown("<div class='section-title'>State of Charge (SOC)</div>", unsafe_allow_html=True)

        latest_soc_pct = float(df["SOC"].iloc[-1] * 100.0)
        g = soc_gauge(latest_soc_pct)
        st.plotly_chart(g, use_container_width=False)

        soc_fig = plot_time_series(df_window, x_col=x_col, y_col="SOC", title="SOC Trend vs Time (live)", color="#38bdf8", yaxis_title="SOC (0-1)")

        soc_fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(soc_fig, use_container_width=True)

    elif page == "SOH (ML Prediction)":
        st.markdown("<div class='section-title'>State of Health (SOH) — ML Prediction</div>", unsafe_allow_html=True)

        latest_soh_pct = float(df["SOH"].iloc[-1] * 100.0)
        st.markdown(health_badge_html(soh_condition(latest_soh_pct)), unsafe_allow_html=True)

        # Trend
        soh_fig = plot_time_series(df_window, x_col=x_col, y_col="SOH", title="SOH Trend (Model Output) vs Time (live)", color="#a78bfa", yaxis_title="SOH (0-1)")

        soh_fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(soh_fig, use_container_width=True)

        # Feature transparency
        st.markdown("<div class='section-title' style='font-size:1.05rem;margin-top:1rem;'>Model Inputs</div>", unsafe_allow_html=True)
        st.code("Features used: " + ", ".join(str(f) for f in features))

    elif page == "Fault Detection":
        st.markdown("<div class='section-title'>Fault Detection System</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class='info-box'>Fault thresholds are <b>dataset-driven</b> using robust quantiles (e.g., p98/p2) so the dashboard adapts to your Simscape export.</div>
            """,
            unsafe_allow_html=True,
        )

        # Latest alerts
        for a in latest_alerts:
            sev = a.get("severity")
            if sev == "GOOD":
                st.markdown(
                    f"<div class='info-box alert-good'><b>{a['title']}</b><div style='margin-top:.35rem;color:rgba(232,238,252,0.85)'>{a['detail']}</div></div>",
                    unsafe_allow_html=True,
                )
            elif sev == "ERROR":
                st.markdown(
                    f"<div class='info-box alert-bad'><b>{a['title']}</b><div style='margin-top:.35rem;color:rgba(232,238,252,0.85)'>{a['detail']}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='info-box alert-mod'><b>{a['title']}</b><div style='margin-top:.35rem;color:rgba(232,238,252,0.85)'>{a['detail']}</div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div class='section-title' style='font-size:1.05rem;margin-top:1rem;'>Fault Flags Snapshot</div>", unsafe_allow_html=True)
        cols = ["overheating", "overcharging", "voltage_abnormal", "high_current", "resistance_abnormal"]
        snap = alerts_df[cols].tail(25).copy()
        snap[x_col] = df[x_col].tail(25).to_numpy()

        st.dataframe(snap, use_container_width=True, hide_index=True)

    elif page == "Smart Charging":
        st.markdown("<div class='section-title'>Smart Charging Recommendation</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class='info-box'>Recommendation uses: <b>SOC</b> (energy state), <b>SOH</b> (health constraint), and <b>temperature</b> (thermal safety).</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="info-box" style="border-left-color:#38bdf8;background:rgba(56,189,248,0.08);">
              <div style="font-weight:900;font-size:1.1rem;">Charging Mode: {charging_rec['mode']}</div>
              <div style="margin-top:.35rem;">Optimization Status: <b>{charging_rec['optimization_status']}</b></div>
              <div style="margin-top:.55rem;color:rgba(232,238,252,0.95);">Recommendation: {charging_rec['recommendation']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


        if charging_rec["reasons"]:
            st.warning("Reasons:")
            for r in charging_rec["reasons"]:
                st.write(f"• {r}")

        # Show SOC/SOH gauge context
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(soc_gauge(float(df["SOC"].iloc[-1] * 100.0)), use_container_width=True)
        with c2:
            st.plotly_chart(soc_gauge(float(df["SOH"].iloc[-1] * 100.0)), use_container_width=True)

    elif page == "Remaining Distance":
        st.markdown("<div class='section-title'>Remaining Driving Range Prediction</div>", unsafe_allow_html=True)

        remaining_km = predict_remaining_distance(df_window["SOC"], df_window)
        st.metric(label="Estimated Remaining Distance", value=f"{remaining_km:.1f} km")


        # Additional context charts
        eff = float(df["Efficiency_proxy"].iloc[0]) if "Efficiency_proxy" in df.columns else np.nan
        st.caption(f"Model uses SOC and an efficiency proxy derived from power characteristics. Current efficiency proxy: {eff*100:.1f}%" if np.isfinite(eff) else "Efficiency proxy not available.")

    elif page == "CPS Digital Twin":
        st.markdown("<div class='section-title'>Digital Twin / CPS Workflow</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='info-box'>
            <b>Simscape (Physical Layer)</b> models the battery electro-thermal behavior.<br/>
            <b>CSV Export</b> transfers the simulated state variables to disk.<br/>
            <b>AI Analytics (Cyber Layer)</b> estimates SOC/SOH, detects faults, and proposes charging strategies.<br/>
            <b>Smart Dashboard</b> visualizes everything interactively for monitoring and optimization.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Simple diagram-like visualization
        fig = go.Figure()
        nodes = [
            (0, 1, "Simscape\n(Physical Layer)"),
            (1.5, 1, "CSV Export"),
            (3, 1, "AI Analytics\n(SOC/SOH/Faults)"),
            (4.5, 1, "Smart Dashboard"),
        ]
        for x, y, label in nodes:
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers+text", text=[label], textposition="middle center",
                marker=dict(size=28, color="#1f2a44", line=dict(color="#38bdf8", width=2)),
                textfont=dict(color="#e8eefc", size=12),
            ))
        # arrows
        for (x1, _, _), (x2, _, _) in zip(nodes[:-1], nodes[1:]):
            fig.add_annotation(x=x2 - 0.6, y=1, ax=x1 + 0.6, ay=1, showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2, arrowcolor="#38bdf8")
        fig.update_layout(
            template="plotly_dark",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=20, b=20),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Report":
        st.markdown("<div class='section-title'>Project Report</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class='info-box'>
            <b>Objectives</b><br/>
            • Build a professional EV battery monitoring dashboard using cyber-physical systems concepts.<br/>
            • Compute power/energy/resistance metrics from Simscape-exported battery data.<br/>
            • Estimate SOC and predict SOH using machine learning.<br/>
            • Detect faults (thermal, electrical, abnormal signals) and recommend smart charging strategies.<br/>
            • Provide remaining range estimation for engineering decision support.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='info-box' style='margin-top:1rem;border-left-color:#a78bfa;'>
            <b>Methodology</b><br/>
            • Robust CSV ingestion with safe auto-detection of key columns.<br/>
            • Derived computations: Power = V×I, Resistance = V/I, Energy integration via trapezoidal rule.<br/>
            • SOC heuristic mapping based on observed voltage range (bounded to [0,1]).<br/>
            • SOH regression using Random Forest with a health-proxy target (resistance & voltage stability), with Linear Regression fallback.<br/>
            • Fault detection using dataset-adaptive quantile thresholds.<br/>
            • Charging recommendation driven by SOC/SOH/temperature constraints.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='info-box' style='margin-top:1rem;border-left-color:#22c55e;'>
            <b>Results</b><br/>
            • Interactive KPI cards quantify peak power, energy, SOC, and SOH.<br/>
            • Plotly charts provide zoom/hover exploration of battery dynamics.<br/>
            • SOH classification reports GOOD/MODERATE/CRITICAL health states.<br/>
            • Real-time alert boxes highlight thermal and electrical risks.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='info-box' style='margin-top:1rem;border-left-color:#ef4444;'>
            <b>Conclusion</b><br/>
            The dashboard demonstrates an end-to-end cyber-physical monitoring workflow: Simscape simulation data is transformed into actionable analytics, predictions, and control recommendations using modern Python ML and interactive visualization.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='info-box' style='margin-top:1rem;border-left-color:#f59e0b;'>
            <b>Future Scope</b><br/>
            • Integrate true capacity/SOH labels (if available) for supervised learning and validation metrics.<br/>
            • Replace heuristic SOC with physically calibrated coulomb counting (need capacity and current sign convention).<br/>
            • Add battery aging model and control-loop simulation for closed-loop smart charging.
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif page == "Raw Data":
        st.markdown("<div class='section-title'>Raw Simulation Data (Processed)</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='info-box'>Loaded from:<br/><code style='color:#e8eefc;'>{DATA_PATH}</code></div>",
            unsafe_allow_html=True,
        )

        with st.expander("Click to view dataset table", expanded=False):
            st.dataframe(df.round(6).head(2000), use_container_width=True, hide_index=True)
            st.caption("Showing first 2000 rows for performance.")

        # Download processed CSV
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Processed Data (CSV)",
            data=csv_bytes,
            file_name="battery_processed.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()

