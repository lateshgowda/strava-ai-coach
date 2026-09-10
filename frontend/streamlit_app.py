"""
Strava AI Coach — Streamlit Dashboard
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Strava AI Coach",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark-mode compatibility and polish
st.markdown(
    """
    <style>
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    .insight-box {
        background: rgba(99,179,237,0.1);
        border-left: 4px solid #63b3ed;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        white-space: pre-wrap;
        line-height: 1.7;
    }
    .warning-box {
        background: rgba(245,101,101,0.1);
        border-left: 4px solid #f56565;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .success-box {
        background: rgba(72,187,120,0.1);
        border-left: 4px solid #48bb78;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .stTabs [data-baseweb="tab"] { font-size: 15px; }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PRIMARY = "#F97316"   # Strava orange
COLOR_BLUE = "#63B3ED"
COLOR_GREEN = "#48BB78"
COLOR_RED = "#F56565"
COLOR_YELLOW = "#ECC94B"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _api_get(path: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    """GET request to backend. Returns parsed JSON or None on error."""
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach backend at {BACKEND_URL}. Is it running?")
    except requests.exceptions.Timeout:
        st.error("Backend request timed out.")
    except requests.exceptions.HTTPError as exc:
        st.error(f"Backend error: {exc.response.status_code} — {exc.response.text[:200]}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
    return None


def _api_post(path: str, params: Optional[Dict] = None, timeout: int = 120) -> Optional[Dict[str, Any]]:
    """POST request to backend. Returns parsed JSON or None on error."""
    try:
        resp = requests.post(f"{BACKEND_URL}{path}", params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach backend at {BACKEND_URL}.")
    except requests.exceptions.Timeout:
        st.error("Sync request timed out. It may still be running in the background.")
    except requests.exceptions.HTTPError as exc:
        st.error(f"Backend error: {exc.response.status_code} — {exc.response.text[:200]}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
    return None


def get_auth_status() -> Dict[str, Any]:
    result = _api_get("/auth/status")
    return result or {"connected": False, "athlete_name": None}


def get_dashboard_data() -> Optional[Dict[str, Any]]:
    return _api_get("/dashboard", timeout=30)


def _api_post_json(path: str, body: Dict, timeout: int = 60) -> Optional[Dict[str, Any]]:
    """POST with JSON body to backend. Returns parsed JSON or None on error."""
    try:
        resp = requests.post(f"{BACKEND_URL}{path}", json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach backend at {BACKEND_URL}.")
    except requests.exceptions.Timeout:
        st.error("Request timed out.")
    except requests.exceptions.HTTPError as exc:
        st.error(f"Backend error: {exc.response.status_code} — {exc.response.text[:200]}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
    return None


def sync_activities(full_sync: bool = False) -> Optional[Dict[str, Any]]:
    return _api_post("/sync/activities", params={"full_sync": str(full_sync).lower()})


@st.cache_data(ttl=300)
def get_all_activities_cached(activity_type: Optional[str] = None) -> List[Dict]:
    """Fetch activities from backend, cached for 5 min per type."""
    params = "?limit=500"
    if activity_type and activity_type != "All":
        params += f"&activity_type={activity_type}"
    result = _api_get(f"/activities{params}", timeout=30)
    return result if isinstance(result, list) else []


def get_ai_insight() -> str:
    result = _api_get("/ai/insight", timeout=60)
    return result.get("insight", "No insight available.") if result else "Failed to get insight."


def get_weekly_summary() -> str:
    result = _api_get("/ai/weekly", timeout=60)
    return result.get("summary", "No summary available.") if result else "Failed to get summary."


def get_sleep_history(days: int = 14) -> List[Dict]:
    result = _api_get(f"/health-data/history?days={days}", timeout=15)
    return result if isinstance(result, list) else []


def get_garmin_daily(days: int = 14) -> List[Dict]:
    result = _api_get(f"/garmin/daily?days={days}", timeout=15)
    return result if isinstance(result, list) else []


def get_garmin_sleep_history(days: int = 30) -> List[Dict]:
    result = _api_get(f"/garmin/sleep-history?days={days}", timeout=15)
    return result if isinstance(result, list) else []


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_pace(pace: Optional[float]) -> str:
    if pace is None or (isinstance(pace, float) and math.isnan(pace)):
        return "–"
    minutes = int(pace)
    seconds = int(round((pace - minutes) * 60))
    return f"{minutes}:{seconds:02d} /km"


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "–"
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _safe(val: Any, default: str = "–") -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return str(val)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def make_weekly_bar(weekly: List[Dict]) -> go.Figure:
    if not weekly:
        return _empty_figure("No weekly data yet")
    df = pd.DataFrame(weekly)
    fig = px.bar(
        df,
        x="week",
        y="distance_km",
        color_discrete_sequence=[COLOR_PRIMARY],
        template=PLOTLY_TEMPLATE,
        labels={"distance_km": "Distance (km)", "week": "Week"},
        title="Weekly Mileage",
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=40, r=20, t=50, b=60),
        height=350,
    )
    return fig


def make_monthly_chart(monthly: List[Dict]) -> go.Figure:
    if not monthly:
        return _empty_figure("No monthly data yet")
    df = pd.DataFrame(monthly)
    fig = go.Figure(layout=dict(template=PLOTLY_TEMPLATE))
    fig.add_bar(x=df["month"], y=df["distance_km"], name="Distance (km)",
                marker_color=COLOR_PRIMARY, opacity=0.8)
    if "run_count" in df.columns:
        fig.add_scatter(
            x=df["month"], y=df["run_count"], name="Runs",
            mode="lines+markers", marker_color=COLOR_BLUE,
            yaxis="y2", line=dict(width=2),
        )
    fig.update_layout(
        title="Monthly Volume",
        yaxis=dict(title="Distance (km)"),
        yaxis2=dict(title="Run Count", overlaying="y", side="right", showgrid=False),
        xaxis_tickangle=-45,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=60, t=50, b=70),
        height=380,
    )
    return fig


def make_pace_trend(pace_trend: List[Dict]) -> go.Figure:
    if not pace_trend:
        return _empty_figure("No pace data yet")
    df = pd.DataFrame(pace_trend)
    df = df.dropna(subset=["pace_min_per_km"])
    if df.empty:
        return _empty_figure("No pace data yet")
    fig = px.line(
        df, x="start_date", y="pace_min_per_km",
        hover_data=["name", "distance_km"],
        color_discrete_sequence=[COLOR_BLUE],
        template=PLOTLY_TEMPLATE,
        labels={"pace_min_per_km": "Pace (min/km)", "start_date": "Date"},
        title="Pace Trend",
        markers=True,
    )
    fig.update_layout(margin=dict(l=40, r=20, t=50, b=40), height=320)
    fig.update_yaxes(autorange="reversed")  # Lower pace = faster
    return fig


def make_hr_trend(hr_trend: List[Dict]) -> go.Figure:
    if not hr_trend:
        return _empty_figure("No heart rate data yet")
    df = pd.DataFrame(hr_trend)
    df = df.dropna(subset=["average_heartrate"])
    if df.empty:
        return _empty_figure("No heart rate data yet")
    fig = px.line(
        df, x="start_date", y="average_heartrate",
        hover_data=["name", "distance_km"],
        color_discrete_sequence=[COLOR_RED],
        template=PLOTLY_TEMPLATE,
        labels={"average_heartrate": "Avg HR (bpm)", "start_date": "Date"},
        title="Heart Rate Trend",
        markers=True,
    )
    fig.update_layout(margin=dict(l=40, r=20, t=50, b=40), height=320)
    return fig


def make_cadence_trend(cadence_trend: List[Dict]) -> go.Figure:
    if not cadence_trend:
        return _empty_figure("No cadence data yet")
    df = pd.DataFrame(cadence_trend)
    df = df.dropna(subset=["average_cadence"])
    if df.empty:
        return _empty_figure("No cadence data yet")
    fig = px.line(
        df, x="start_date", y="average_cadence",
        hover_data=["name"],
        color_discrete_sequence=[COLOR_GREEN],
        template=PLOTLY_TEMPLATE,
        labels={"average_cadence": "Cadence (spm)", "start_date": "Date"},
        title="Cadence Trend",
        markers=True,
    )
    fig.update_layout(margin=dict(l=40, r=20, t=50, b=40), height=320)
    return fig


def make_splits_chart(splits: List[Dict]) -> go.Figure:
    if not splits:
        return _empty_figure("No split data for this run")

    km_labels = [f"km {s.get('split', i+1)}" for i, s in enumerate(splits)]
    paces = []
    for s in splits:
        d = s.get("distance", 0)
        t = s.get("moving_time", 0)
        if d and t and d > 0:
            pace = (t / d) * (1000 / 60)
        else:
            pace = None
        paces.append(pace)

    colors = []
    for p in paces:
        if p is None:
            colors.append(COLOR_BLUE)
        else:
            # Color by pace relative to median
            colors.append(COLOR_BLUE)

    fig = go.Figure(layout=dict(template=PLOTLY_TEMPLATE))
    fig.add_bar(
        x=km_labels,
        y=paces,
        marker_color=COLOR_PRIMARY,
        name="Pace (min/km)",
    )
    fig.update_layout(
        title="Splits (min/km per km)",
        yaxis=dict(title="Pace (min/km)", autorange="reversed"),
        xaxis=dict(title="Kilometre"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=300,
    )
    return fig


def make_gauge(value: float, title: str, max_val: float = 100,
               green_threshold: float = 40, red_threshold: float = 70) -> go.Figure:
    """Create a gauge chart with green/yellow/red zones."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 16}},
            number={"font": {"size": 28}},
            gauge={
                "axis": {"range": [0, max_val], "tickwidth": 1},
                "bar": {"color": COLOR_PRIMARY, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, green_threshold], "color": "rgba(72,187,120,0.3)"},
                    {"range": [green_threshold, red_threshold], "color": "rgba(236,201,75,0.3)"},
                    {"range": [red_threshold, max_val], "color": "rgba(245,101,101,0.3)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=220,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def make_recovery_gauge(value: float) -> go.Figure:
    """Recovery gauge — higher is better, so zones are reversed."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": "Recovery Score", "font": {"size": 16}},
            number={"font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": COLOR_GREEN, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(245,101,101,0.3)"},
                    {"range": [40, 65], "color": "rgba(236,201,75,0.3)"},
                    {"range": [65, 100], "color": "rgba(72,187,120,0.3)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=220,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def make_acwr_gauge(value: float) -> go.Figure:
    """ACWR gauge — 0 to 2.0, green zone 0.8–1.3, red above 1.3."""
    clamped = min(value, 2.0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=clamped,
            title={"text": "ACWR", "font": {"size": 16}},
            number={"font": {"size": 28}, "valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, 2.0], "tickwidth": 1,
                         "tickvals": [0, 0.8, 1.3, 2.0],
                         "ticktext": ["0", "0.8", "1.3", "2.0"]},
                "bar": {"color": COLOR_BLUE, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 0.8], "color": "rgba(236,201,75,0.3)"},
                    {"range": [0.8, 1.3], "color": "rgba(72,187,120,0.3)"},
                    {"range": [1.3, 2.0], "color": "rgba(245,101,101,0.3)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": clamped,
                },
            },
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=220,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _today_recommendation(fatigue: float, recovery: float, acwr: float):
    """Return (text, css_class) for today's training recommendation box."""
    if acwr > 1.3 or fatigue > 75:
        return (
            "🛑 Rest or easy day — training load is elevated. Prioritise sleep, nutrition, and light movement.",
            "warning-box",
        )
    if fatigue > 55 or acwr > 1.1:
        return (
            "🟡 Easy running only — keep effort fully conversational. Avoid tempo or intervals today.",
            "warning-box",
        )
    if fatigue < 25 and acwr < 0.8:
        return (
            "📈 Under-training zone — you are well below your chronic load. Safely increase volume 5–10% this week.",
            "success-box",
        )
    if recovery >= 65 and fatigue < 50:
        return (
            "✅ Ready for quality — fatigue is low and recovery is strong. Tempo, intervals or long run are all appropriate.",
            "success-box",
        )
    return (
        "🟢 Moderate training OK — steady aerobic runs at comfortable effort are ideal today.",
        "success-box",
    )


def make_load_trend_chart(load_trend: List[Dict]) -> go.Figure:
    if not load_trend:
        return _empty_figure("No training load data yet")
    df = pd.DataFrame(load_trend)
    fig = px.area(
        df, x="week", y="load_score",
        color_discrete_sequence=[COLOR_YELLOW],
        template=PLOTLY_TEMPLATE,
        labels={"load_score": "Load Score", "week": "Week"},
        title="Training Load Trend (8 weeks)",
    )
    fig.update_layout(xaxis_tickangle=-45, height=300, margin=dict(l=40, r=20, t=50, b=60))
    return fig


def make_long_run_progression(long_runs: List[Dict]) -> go.Figure:
    if not long_runs:
        return _empty_figure("No long runs recorded yet (>15 km)")
    df = pd.DataFrame(long_runs)
    fig = px.bar(
        df, x="month", y="max_distance_km",
        hover_data=["run_name"] if "run_name" in df.columns else None,
        color_discrete_sequence=[COLOR_PRIMARY],
        template=PLOTLY_TEMPLATE,
        labels={"max_distance_km": "Longest Run (km)", "month": "Month"},
        title="Long Run Progression (longest run per month)",
    )
    fig.update_layout(xaxis_tickangle=-45, height=320, margin=dict(l=40, r=20, t=50, b=60))
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        annotations=[{
            "text": message,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": 0.5,
            "showarrow": False,
            "font": {"size": 16, "color": "gray"},
        }],
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# ACWR zone display
# ---------------------------------------------------------------------------


def _acwr_zone(acwr: float) -> tuple[str, str]:
    """Return (label, CSS class) for the ACWR value."""
    if acwr <= 0:
        return "No data", "metric-card"
    if acwr < 0.8:
        return f"{acwr:.2f} — Under-training zone", "metric-card"
    if acwr <= 1.3:
        return f"{acwr:.2f} — Optimal zone ✓", "success-box"
    return f"{acwr:.2f} — Injury risk zone ⚠", "warning-box"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    st.sidebar.title("Strava AI Coach")
    st.sidebar.markdown("---")

    auth = get_auth_status()
    is_connected = auth.get("connected", False)
    athlete_name = auth.get("athlete_name")

    if is_connected:
        st.sidebar.markdown(
            f'<div class="success-box">Connected as <b>{athlete_name or "Athlete"}</b></div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="warning-box">Not connected to Strava</div>',
            unsafe_allow_html=True,
        )
        st.sidebar.markdown(
            f'<a href="{BACKEND_URL}/auth/strava/login" target="_self">'
            '<button style="background:#FC4C02;color:white;border:none;padding:10px 20px;'
            'border-radius:8px;cursor:pointer;font-size:15px;width:100%;">'
            'Connect Strava</button></a>',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")

    if is_connected:
        full_sync = st.sidebar.checkbox("Full re-sync (slower)", value=False)
        if st.sidebar.button("Sync Activities", use_container_width=True, type="primary"):
            with st.spinner("Syncing activities from Strava…"):
                result = sync_activities(full_sync=full_sync)
            if result:
                st.sidebar.success(
                    f"Synced {result.get('synced', 0)} new, "
                    f"{result.get('skipped', 0)} updated."
                )
                # Invalidate cache
                if "dashboard_data" in st.session_state:
                    del st.session_state["dashboard_data"]
                st.rerun()
            else:
                st.sidebar.error("Sync failed.")

    st.sidebar.markdown("---")
    if st.sidebar.button("Refresh Dashboard", use_container_width=True):
        if "dashboard_data" in st.session_state:
            del st.session_state["dashboard_data"]
        st.rerun()

    if is_connected:
        st.sidebar.markdown("---")
        if st.sidebar.button("Disconnect Strava", use_container_width=True):
            try:
                requests.post(f"{BACKEND_URL}/auth/strava/disconnect", timeout=10)
                if "dashboard_data" in st.session_state:
                    del st.session_state["dashboard_data"]
                st.rerun()
            except Exception:
                st.sidebar.error("Failed to disconnect.")

    # --- Garmin Connect sidebar section ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Garmin Connect**")
    garmin_status = _api_get("/garmin/status") or {}
    garmin_connected = garmin_status.get("connected", False)
    garmin_email = garmin_status.get("email")

    if garmin_connected:
        st.sidebar.markdown(
            f'<div class="success-box" style="font-size:13px">Garmin: <b>{garmin_email}</b></div>',
            unsafe_allow_html=True,
        )
        g_days = st.sidebar.selectbox("Sync days", [2, 7, 14, 30], key="garmin_sync_days", label_visibility="collapsed")
        if st.sidebar.button("Sync Garmin Data", use_container_width=True):
            with st.spinner("Syncing Garmin metrics…"):
                gr = requests.post(f"{BACKEND_URL}/garmin/sync?days={g_days}", timeout=120)
            if gr.status_code == 200:
                res = gr.json()
                st.sidebar.success(f"Synced {res.get('synced', 0)} days.")
                if res.get("errors"):
                    st.sidebar.warning(f"{len(res['errors'])} errors.")
            else:
                st.sidebar.error("Garmin sync failed.")
        if st.sidebar.button("Disconnect Garmin", use_container_width=True):
            requests.delete(f"{BACKEND_URL}/garmin/disconnect", timeout=10)
            st.rerun()
    else:
        if "garmin_mfa_pending" not in st.session_state:
            st.session_state["garmin_mfa_pending"] = False

        g_email = st.sidebar.text_input("Garmin email", key="g_email", placeholder="email@example.com")
        g_pass = st.sidebar.text_input("Garmin password", key="g_pass", type="password")
        g_mfa = None
        if st.session_state["garmin_mfa_pending"]:
            g_mfa = st.sidebar.text_input("MFA / OTP code", key="g_mfa", placeholder="123456")
        if st.sidebar.button("Connect Garmin", use_container_width=True, type="primary"):
            if g_email and g_pass:
                with st.spinner("Connecting to Garmin…"):
                    body: Dict[str, Any] = {"email": g_email, "password": g_pass}
                    if g_mfa:
                        body["mfa_code"] = g_mfa
                    cr = requests.post(f"{BACKEND_URL}/garmin/connect", json=body, timeout=60)
                if cr.status_code == 200:
                    result = cr.json()
                    if result.get("status") == "mfa_required":
                        st.session_state["garmin_mfa_pending"] = True
                        st.sidebar.warning("MFA required — enter your one-time code above.")
                    elif result.get("status") == "connected":
                        st.session_state["garmin_mfa_pending"] = False
                        st.sidebar.success("Garmin connected!")
                        st.rerun()
                else:
                    st.sidebar.error(f"Failed: {cr.text[:200]}")
            else:
                st.sidebar.warning("Enter email and password.")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Backend: {BACKEND_URL}")


# ---------------------------------------------------------------------------
# Tab: Garmin
# ---------------------------------------------------------------------------


def _fmt_sleep_str(secs: Optional[int]) -> str:
    if secs is None:
        return "–"
    h = secs // 3600
    m = (secs % 3600) // 60
    return f"{h}h {m:02d}m"


def render_garmin() -> None:
    garmin_status = _api_get("/garmin/status") or {}
    if not garmin_status.get("connected"):
        st.info("Connect your Garmin account in the sidebar to see your daily metrics here.")
        return

    st.subheader("Garmin Daily Metrics")

    daily = _api_get("/garmin/daily?days=14") or []

    if not daily:
        st.warning("No Garmin data synced yet — click **Sync Garmin Data** in the sidebar.")
        return

    # --- Chart: Body Battery + Resting HR ---
    chart_dates = [d["date"] for d in reversed(daily)]
    bb_vals = [d.get("body_battery_max") for d in reversed(daily)]
    hr_vals = [d.get("resting_hr") for d in reversed(daily)]

    import plotly.graph_objects as _go
    fig = _go.Figure()
    fig.add_trace(_go.Bar(
        x=chart_dates, y=bb_vals,
        name="Body Battery",
        marker_color="rgba(56,178,172,0.7)",
        yaxis="y",
    ))
    fig.add_trace(_go.Scatter(
        x=chart_dates, y=hr_vals,
        name="Resting HR",
        mode="lines+markers",
        line=dict(color="#FC4C02", width=2),
        marker=dict(size=6),
        yaxis="y2",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=0, r=0, t=10, b=30),
        yaxis=dict(title="Body Battery", range=[0, 100], showgrid=False),
        yaxis2=dict(title="Resting HR (bpm)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Daily table (last 7 days) ---
    st.markdown("**Last 7 Days**")
    hdr = st.columns([1.4, 1.3, 1, 1.2, 1, 1])
    hdr[0].markdown("**Date**")
    hdr[1].markdown("**Body Battery**")
    hdr[2].markdown("**Resting HR**")
    hdr[3].markdown("**Sleep**")
    hdr[4].markdown("**Stress**")
    hdr[5].markdown("**VO2 Max**")

    for row in daily[:7]:
        cols = st.columns([1.4, 1.3, 1, 1.2, 1, 1])
        d = row["date"]
        bb_max = row.get("body_battery_max")
        bb_min = row.get("body_battery_min")
        bb_str = f"{bb_max}→{bb_min}" if bb_max is not None else "–"
        rhr = row.get("resting_hr")
        sleep_str = row.get("sleep_duration_str") or "–"
        stress = row.get("stress_avg")
        vo2 = row.get("vo2max")
        cols[0].markdown(f"<div style='padding-top:5px'>{d}</div>", unsafe_allow_html=True)
        cols[1].markdown(f"<div style='padding-top:5px'>{bb_str}</div>", unsafe_allow_html=True)
        cols[2].markdown(f"<div style='padding-top:5px'>{rhr or '–'} bpm</div>", unsafe_allow_html=True)
        cols[3].markdown(f"<div style='padding-top:5px'>{sleep_str}</div>", unsafe_allow_html=True)
        cols[4].markdown(f"<div style='padding-top:5px'>{stress or '–'}</div>", unsafe_allow_html=True)
        cols[5].markdown(f"<div style='padding-top:5px'>{f'{vo2:.1f}' if vo2 else '–'}</div>", unsafe_allow_html=True)

    # --- Push workouts to watch ---
    st.markdown("---")
    st.subheader("Push FM Plan to Watch")

    plan_status = _api_get("/plan/status") or {}
    if plan_status.get("total_workouts", 0) == 0:
        st.info("Import your FM Plan first (FM Plan tab).")
        return

    from datetime import date as _date
    today_iso = _date.today().strftime("%G-W%V")

    push_col1, push_col2 = st.columns([2, 1])
    with push_col1:
        st.markdown(f"Push this week's workouts ({today_iso}) to your Forerunner 645:")
    with push_col2:
        if st.button("Push This Week", type="primary", use_container_width=True):
            with st.spinner("Pushing workouts to Garmin calendar…"):
                pr = requests.post(f"{BACKEND_URL}/garmin/push-week/{today_iso}", timeout=60)
            if pr.status_code == 200:
                res = pr.json()
                st.success(f"Pushed {res.get('pushed', 0)} workout(s) to your watch calendar.")
                if res.get("errors"):
                    for e in res["errors"]:
                        st.warning(f"Error on {e['date']}: {e['error']}")
            elif pr.status_code == 404:
                st.warning("No workouts found for this week or Garmin not connected.")
            else:
                st.error(f"Push failed: {pr.text[:200]}")

    # Individual workout push
    week_data = _api_get(f"/plan/week/{today_iso}")
    if week_data and week_data.get("workouts"):
        st.markdown("**Individual workouts this week:**")
        for w in week_data["workouts"]:
            if w["status"] == "upcoming":
                wcol1, wcol2 = st.columns([3, 1])
                wcol1.markdown(
                    f"{w['plan_date']} — {w['session_type']} {w['distance_km']:.1f}km"
                    + (f" · _{w['details']}_" if w.get("details") else "")
                )
                if wcol2.button("Push", key=f"push_w_{w['id']}", use_container_width=True):
                    with st.spinner("Pushing…"):
                        pr2 = requests.post(f"{BACKEND_URL}/garmin/push-workout/{w['id']}", timeout=30)
                    if pr2.status_code == 200:
                        r2 = pr2.json()
                        st.success(f"Pushed '{r2.get('name')}' to watch for {r2.get('scheduled')}.")
                    else:
                        st.error(f"Push failed: {pr2.text[:200]}")


# ---------------------------------------------------------------------------
# Best Efforts section (shown on main dashboard)
# ---------------------------------------------------------------------------

_MEDAL = ["🥇", "🥈", "🥉"]


def render_best_efforts() -> None:
    data = _api_get("/best-efforts")
    if not data:
        return

    st.markdown("---")
    st.subheader("Best Efforts")

    buckets = [
        ("5k",      "5 K"),
        ("10k",     "10 K"),
        ("21k",     "Half Marathon"),
        ("longest", "Longest Runs"),
    ]

    last = data.get("last", {})

    cols = st.columns(4)
    for col, (key, label) in zip(cols, buckets):
        efforts: List[Dict[str, Any]] = data.get(key, [])
        last_effort: Optional[Dict[str, Any]] = last.get(key) if key != "5k" else None
        with col:
            st.markdown(f"**{label}**")
            if not efforts:
                st.caption("No data yet")
                continue

            # Top-3 best efforts
            rows_html = ""
            for rank, e in enumerate(efforts):
                medal = _MEDAL[rank] if rank < 3 else f"#{rank+1}"
                dist_str = f"{e['distance_km']:.1f} km" if key == "longest" else ""
                pace_str = f"{e['pace']}/km" if e.get("pace") else ""
                hr_str = f" · {e['avg_hr']} bpm" if e.get("avg_hr") else ""
                sub = " · ".join(filter(None, [dist_str, pace_str])) + hr_str
                rows_html += (
                    f"<div style='padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.08)'>"
                    f"<span style='font-size:18px'>{medal}</span> "
                    f"<span style='font-size:17px;font-weight:700'>{e['time_str']}</span><br>"
                    f"<span style='font-size:12px;color:#a0aec0'>{sub}</span><br>"
                    f"<span style='font-size:11px;color:#718096'>{e['date']}</span>"
                    f"</div>"
                )
            st.markdown(rows_html, unsafe_allow_html=True)

            # Last effort row
            st.markdown(
                "<div style='margin-top:10px;font-size:11px;color:#718096;text-transform:uppercase;"
                "letter-spacing:.06em'>Last</div>",
                unsafe_allow_html=True,
            )
            if last_effort:
                e = last_effort
                dist_str = f"{e['distance_km']:.1f} km" if key == "longest" else f"{e['distance_km']:.1f} km"
                pace_str = f"{e['pace']}/km" if e.get("pace") else ""
                hr_str = f" · {e['avg_hr']} bpm" if e.get("avg_hr") else ""
                sub = " · ".join(filter(None, [dist_str, pace_str])) + hr_str
                st.markdown(
                    f"<div style='padding:6px 0'>"
                    f"<span style='font-size:16px;font-weight:600'>{e['time_str']}</span><br>"
                    f"<span style='font-size:12px;color:#a0aec0'>{sub}</span><br>"
                    f"<span style='font-size:11px;color:#718096'>{e['date']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<span style='color:#718096;font-size:13px'>–</span>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: Best Efforts (year-by-year)
# ---------------------------------------------------------------------------

_BE_COLS = [
    ("5k",      "5 K"),
    ("10k",     "10 K"),
    ("21k",     "Half Marathon"),
    ("longest", "Longest"),
]


def _parse_time_secs(time_str: str) -> int:
    """Convert 'MM:SS' or 'H:MM:SS' to total seconds for comparison."""
    try:
        parts = [int(p) for p in time_str.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except Exception:
        return 999999


def _be_cell(e: Optional[Dict[str, Any]], show_dist: bool = False, is_pr: bool = False) -> str:
    if not e:
        return "<span style='color:#4a5568'>–</span>"
    dist_str = f"{e['distance_km']:.1f} km · " if show_dist else ""
    pace_str = f"{e['pace']}/km" if e.get("pace") else ""
    hr_str = f" · {e['avg_hr']} bpm" if e.get("avg_hr") else ""
    sub = dist_str + pace_str + hr_str
    pr_badge = " 🏆" if is_pr else ""
    bg = "background:rgba(236,201,75,0.12);border-left:3px solid #ECC94B;border-radius:6px;padding:6px 10px;" if is_pr else "padding:6px 0;"
    time_color = "#ECC94B" if is_pr else "inherit"
    return (
        f"<div style='{bg}'>"
        f"<span style='font-size:15px;font-weight:700;color:{time_color}'>{e['time_str']}{pr_badge}</span><br>"
        f"<span style='font-size:11px;color:#a0aec0'>{sub}</span><br>"
        f"<span style='font-size:11px;color:#718096'>{e['date']}</span>"
        f"</div>"
    )


def render_best_efforts_tab() -> None:
    data = _api_get("/best-efforts/yearly")
    if not data:
        st.info("No activity data yet.")
        return

    years = data.get("years", [])
    if not years:
        st.info("No activity data yet.")
        return

    # Find PR year per column
    pr_years: Dict[str, Optional[int]] = {}
    for key, _ in _BE_COLS:
        best_val = None
        best_yr = None
        for row in years:
            e = row.get(key)
            if not e:
                continue
            if key == "longest":
                val = e.get("distance_km", 0.0)
                if best_val is None or val > best_val:
                    best_val = val
                    best_yr = row["year"]
            else:
                val = _parse_time_secs(e.get("time_str", ""))
                if best_val is None or val < best_val:
                    best_val = val
                    best_yr = row["year"]
        pr_years[key] = best_yr

    # Header row
    hcols = st.columns([1, 1.5, 2, 2, 2, 2])
    hcols[0].markdown("**Year**")
    hcols[1].markdown("**Total KM**")
    for i, (_, label) in enumerate(_BE_COLS):
        hcols[i + 2].markdown(f"**{label}**")

    st.markdown("<hr style='margin:4px 0;border-color:rgba(255,255,255,0.1)'>", unsafe_allow_html=True)

    for row in years:
        yr = row["year"]
        rcols = st.columns([1, 1.5, 2, 2, 2, 2])
        rcols[0].markdown(f"<div style='padding-top:8px;font-weight:600'>{yr}</div>", unsafe_allow_html=True)
        total_km = row.get("total_km", 0) or 0
        rcols[1].markdown(
            f"<div style='padding-top:8px;font-size:15px;font-weight:600'>{total_km:,.0f} km</div>",
            unsafe_allow_html=True,
        )
        for i, (key, _) in enumerate(_BE_COLS):
            show_dist = key == "longest"
            is_pr = pr_years.get(key) == yr
            rcols[i + 2].markdown(_be_cell(row.get(key), show_dist=show_dist, is_pr=is_pr), unsafe_allow_html=True)
        st.markdown("<hr style='margin:2px 0;border-color:rgba(255,255,255,0.06)'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: Latest Run
# ---------------------------------------------------------------------------


def render_latest_run(data: Dict[str, Any]) -> None:
    latest = data.get("latest_run", {})

    if not latest:
        st.info("No activities yet — sync to get started.")
        return

    st.subheader(latest.get("name", "Latest Run"))
    st.caption(f"Date: {latest.get('start_date', '')[:10]}")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Distance", f"{latest.get('distance_km', 0):.2f} km")
    col2.metric("Pace", latest.get("pace_str", "–"))
    col3.metric("Time", latest.get("moving_time_str", "–"))

    hr = latest.get("average_heartrate")
    col4.metric("Avg HR", f"{hr:.0f} bpm" if hr else "–")

    cadence = latest.get("average_cadence")
    col5.metric("Cadence", f"{cadence:.0f} spm" if cadence else "–")

    elev = latest.get("elevation_gain", 0)
    col6.metric("Elevation", f"{elev:.0f} m")

    prs = latest.get("pr_count", 0)
    if prs and prs > 0:
        st.markdown(f'<div class="success-box">🏆 This run contains <b>{prs}</b> personal record(s)!</div>',
                    unsafe_allow_html=True)

    # Splits chart — fetch on-demand from Strava if not cached
    st.markdown("---")
    st.subheader("Kilometre Splits")

    strava_id = latest.get("strava_id")
    splits = latest.get("splits", [])

    if not splits and strava_id:
        cache_key = f"splits_{strava_id}"
        if cache_key in st.session_state:
            splits = st.session_state[cache_key]
        else:
            with st.spinner("Fetching splits from Strava…"):
                result = _api_get(f"/activities/{strava_id}/splits", timeout=15)
            if result and result.get("splits"):
                splits = result["splits"]
                st.session_state[cache_key] = splits

    st.plotly_chart(make_splits_chart(splits), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Weekly
# ---------------------------------------------------------------------------


def render_trends(_data: Dict[str, Any]) -> None:
    """Unified Trends tab — Strava-style area chart + 6 metric summary cards."""

    _METRICS = [
        ("distance_km",     "Distance",   "km",  lambda v: f"{v:.1f}"),
        ("avg_pace",        "Avg Pace",   "/km", lambda v: format_pace(v)),
        ("activities",      "Activities", "",    lambda v: str(int(v))),
        ("elevation_m",     "Elevation",  "m",   lambda v: f"{v:.0f}"),
        ("moving_time_sec", "Time Spent", "",    lambda v: format_duration(v)),
        ("avg_hr",          "Avg HR",     "bpm", lambda v: f"{v:.0f}"),
    ]

    # ── Period toggle ────────────────────────────────────────────────────────
    period_label = st.radio(
        "",
        ["Weeks", "Months", "Years"],
        horizontal=True,
        key="trend_period",
        label_visibility="collapsed",
    )
    period_key = {"Weeks": "week", "Months": "month", "Years": "year"}[period_label]
    period_singular = {"Weeks": "Week", "Months": "Month", "Years": "Year"}[period_label]

    # ── Fetch data ───────────────────────────────────────────────────────────
    try:
        resp = requests.get(f"{BACKEND_URL}/trends", params={"period": period_key}, timeout=10)
        trends: List[Dict] = resp.json()
    except Exception as exc:
        st.error(f"Could not load trends data: {exc}")
        return

    if not trends:
        st.info("No running data yet — sync your Strava activities first.")
        return

    current = trends[-1]
    previous = trends[-2] if len(trends) >= 2 else {}

    # ── Metric selector (pill row) ───────────────────────────────────────────
    if "trend_metric" not in st.session_state:
        st.session_state["trend_metric"] = "distance_km"

    sel_cols = st.columns(len(_METRICS))
    for i, (key, label, unit, _) in enumerate(_METRICS):
        with sel_cols[i]:
            is_sel = st.session_state["trend_metric"] == key
            if st.button(
                label,
                key=f"tmbtn_{key}_{period_key}",
                type="primary" if is_sel else "secondary",
                use_container_width=True,
            ):
                st.session_state["trend_metric"] = key
                st.rerun()

    st.markdown("")

    # ── Area chart ───────────────────────────────────────────────────────────
    sel_key = st.session_state["trend_metric"]
    sel_meta = next(m for m in _METRICS if m[0] == sel_key)
    sel_label, sel_unit, sel_fmt = sel_meta[1], sel_meta[2], sel_meta[3]

    xs = [t["display"] for t in trends]
    ys = [t.get(sel_key) for t in trends]
    cur_idx = next((i for i, t in enumerate(trends) if t.get("is_current")), len(trends) - 1)

    valid_ys = [v for v in ys if v is not None]
    avg_val = sum(valid_ys) / len(valid_ys) if valid_ys else None
    avg_str = (
        f"{len(trends)}-{period_singular} Avg: {sel_fmt(avg_val)} {sel_unit}".strip()
        if avg_val is not None else ""
    )

    import plotly.graph_objects as _go

    # Open circle for past periods, filled dark circle for current
    marker_colors = ["#1a202c" if i == cur_idx else "rgba(0,0,0,0)" for i in range(len(xs))]
    marker_sizes = [10 if i == cur_idx else 7 for i in range(len(xs))]

    fig = _go.Figure()
    fig.add_trace(_go.Scatter(
        x=xs,
        y=ys,
        mode="lines+markers",
        line=dict(color="#38b2ac", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(56,178,172,0.12)",
        marker=dict(
            size=marker_sizes,
            color=marker_colors,
            line=dict(color="#38b2ac", width=2),
        ),
        hovertemplate=f"%{{x}}<br>{sel_label}: %{{y:.1f}} {sel_unit}<extra></extra>",
    ))

    if cur_idx < len(xs):
        fig.add_vline(x=xs[cur_idx], line_width=1.5, line_color="rgba(45,55,72,0.8)")

    fig.update_layout(
        title=dict(
            text=f"<b style='font-size:17px'>{sel_label}</b>"
                 + (f"<br><span style='font-size:12px;color:#718096'>{avg_str}</span>" if avg_str else ""),
            x=0,
            xanchor="left",
            font=dict(color="#e2e8f0"),
        ),
        height=300,
        margin=dict(l=0, r=16, t=56, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc", size=11),
        showlegend=False,
        xaxis=dict(showgrid=False, tickangle=-30 if period_key == "week" else 0),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 6 metric summary cards ───────────────────────────────────────────────
    card_cols = st.columns(6)
    for i, (key, label, unit, fmt) in enumerate(_METRICS):
        curr_val = current.get(key)
        prev_val = previous.get(key)

        if curr_val is not None and prev_val is not None and prev_val != 0:
            if key == "avg_pace":
                improved = curr_val < prev_val
            else:
                improved = curr_val >= prev_val
            arrow = "↗" if improved else "↘"
            arrow_color = "#48bb78" if improved else "#fc8181"
        else:
            arrow = ""
            arrow_color = "#aaa"

        curr_str = (fmt(curr_val) + (f" {unit}" if unit else "")).strip() if curr_val is not None else "–"
        prev_str = (fmt(prev_val) + (f" {unit}" if unit else "")).strip() if prev_val is not None else "–"
        is_sel = st.session_state["trend_metric"] == key
        border = "border:2px solid #38b2ac;" if is_sel else "border:1px solid rgba(255,255,255,0.08);"

        with card_cols[i]:
            st.markdown(
                f"""<div style="background:rgba(255,255,255,0.04);border-radius:10px;
                    padding:12px 10px;{border}min-height:96px">
                  <div style="font-size:0.72em;font-weight:600;color:#a0aec0">{label}</div>
                  <div style="font-size:0.62em;color:#718096;margin-top:1px">This {period_singular}</div>
                  <div style="font-size:1.05em;font-weight:700;margin:5px 0;color:#e2e8f0">
                    {curr_str}&nbsp;<span style="font-size:0.8em;color:{arrow_color}">{arrow}</span>
                  </div>
                  <div style="font-size:0.62em;color:#718096">
                    Last {period_singular}: {prev_str}
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Tab: AI Coach
# ---------------------------------------------------------------------------


def render_ai_coach(data: Dict[str, Any]) -> None:
    if data.get("total_activities", 0) == 0:
        st.info("No activities yet. Sync your Strava data first!")
        return

    # ── Initialise session state ──────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Layout: metrics strip + chat ──────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Fatigue Score", f"{data.get('fatigue_score', 0):.1f} / 100")
    col_b.metric("Recovery Score", f"{data.get('recovery_score', 0):.1f} / 100")
    acwr = data.get("acwr", 0)
    acwr_label = "🟢 Optimal" if acwr < 1.0 else ("🟡 Moderate" if acwr <= 1.3 else "🔴 High Risk")
    col_c.metric("ACWR", f"{acwr:.2f}", delta=acwr_label, delta_color="off")

    st.markdown("---")
    st.subheader("💬 Chat with your AI Coach")
    st.caption("Ask anything about your training — pace, recovery, next workout, race goals…")

    # ── Suggested prompts ─────────────────────────────────────────────────
    suggestions = [
        "Am I recovered enough for intervals tomorrow?",
        "Why is my HR drifting in recent runs?",
        "Should I reduce mileage this week?",
        "Am I on track for a sub-2 hour half marathon?",
        "Why are my long runs fading after 12 km?",
    ]
    st.markdown("**Quick questions:**")
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion, key=f"suggest_{i}", use_container_width=True):
            st.session_state.pending_message = suggestion

    st.markdown("")

    # ── Render existing chat history ──────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        # Welcome message on first open
        if not st.session_state.chat_history:
            with st.chat_message("assistant"):
                st.markdown(
                    "👋 Hi! I'm your AI running coach. I have full access to your training data — "
                    f"**{data.get('total_activities', 0)} runs** synced. Ask me anything about your "
                    "training, recovery, pacing, or race goals!"
                )

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ── Handle suggested prompt click ─────────────────────────────────────
    if "pending_message" in st.session_state:
        user_input = st.session_state.pop("pending_message")
        _send_chat_message(user_input, data)
        st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────
    if user_input := st.chat_input("Ask your coach…"):
        _send_chat_message(user_input, data)
        st.rerun()

    # ── Clear chat button ─────────────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button("🗑 Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


def _send_chat_message(user_input: str, data: Dict[str, Any]) -> None:
    """Send a message to the AI coach and append both turns to chat_history."""
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("Coach is thinking…"):
        result = _api_post_json(
            "/ai/chat",
            body={
                "message": user_input,
                "history": st.session_state.chat_history[:-1],  # exclude the just-added user msg
            },
            timeout=60,
        )

    reply = result.get("response", "Sorry, I couldn't get a response.") if result else "Coach is unavailable right now."
    st.session_state.chat_history.append({"role": "assistant", "content": reply})


# ---------------------------------------------------------------------------
# Tab: Fatigue & Recovery
# ---------------------------------------------------------------------------


def render_fatigue(data: Dict[str, Any]) -> None:
    fatigue = data.get("fatigue_score", 0.0)
    recovery = data.get("recovery_score", 0.0)
    acwr = data.get("acwr", 0.0)
    load_trend = data.get("training_load_trend", [])
    status = data.get("training_status", {})

    if data.get("total_activities", 0) == 0:
        st.info("No activities yet — sync to see your fatigue and recovery metrics.")
        return

    # --- Today's recommendation (use rich readiness data if available) ---
    st.subheader("Today's Training Recommendation")
    readiness_data = data.get("readiness", {})
    if readiness_data:
        r_score = readiness_data.get("readiness_score", 50)
        r_label = readiness_data.get("tier_label", "")
        r_desc = readiness_data.get("tier_description", "")
        r_color = readiness_data.get("tier_color", "#63B3ED")
        st.markdown(
            f'<div style="background:rgba(99,179,237,0.08);border-left:4px solid {r_color};'
            f'border-radius:8px;padding:14px 18px;margin:8px 0;">'
            f'<b>{r_label} — Readiness {r_score:.0f}/100</b><br/>'
            f'{r_desc}</div>',
            unsafe_allow_html=True,
        )
    else:
        rec_text, rec_css = _today_recommendation(fatigue, recovery, acwr)
        st.markdown(
            f'<div class="{rec_css}" style="font-size:1.05em; padding: 14px 18px;">{rec_text}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("")

    # --- Gauges (3 base + optional Body Battery from Garmin) ---
    garmin_today = data.get("garmin_today") or {}
    body_battery = garmin_today.get("body_battery_max")

    gauge_cols = st.columns(4 if body_battery is not None else 3)

    with gauge_cols[0]:
        fig = make_gauge(fatigue, "Fatigue Score", max_val=100,
                         green_threshold=35, red_threshold=70)
        st.plotly_chart(fig, use_container_width=True)
        if fatigue > 70:
            st.markdown('<div class="warning-box">High fatigue — prioritise rest and easy runs.</div>',
                        unsafe_allow_html=True)
        elif fatigue < 35:
            st.markdown('<div class="success-box">Low fatigue — well rested.</div>',
                        unsafe_allow_html=True)

    with gauge_cols[1]:
        fig = make_recovery_gauge(recovery)
        st.plotly_chart(fig, use_container_width=True)
        if recovery >= 65:
            st.markdown('<div class="success-box">Good recovery — ready for quality training.</div>',
                        unsafe_allow_html=True)
        elif recovery < 40:
            st.markdown('<div class="warning-box">Low recovery — consider an easy day or rest.</div>',
                        unsafe_allow_html=True)

    with gauge_cols[2]:
        fig = make_acwr_gauge(acwr)
        st.plotly_chart(fig, use_container_width=True)
        if acwr > 1.3:
            st.markdown('<div class="warning-box">Injury risk zone — reduce training load this week.</div>',
                        unsafe_allow_html=True)
        elif acwr < 0.8:
            st.markdown('<div class="success-box">Under-training — safely increase load 5–10%.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">Sweet spot (0.8–1.3) — maintain current load.</div>',
                        unsafe_allow_html=True)

    if body_battery is not None:
        with gauge_cols[3]:
            fig = make_gauge(body_battery, "Body Battery", max_val=100,
                             green_threshold=70, red_threshold=30)
            st.plotly_chart(fig, use_container_width=True)
            if body_battery >= 70:
                st.markdown('<div class="success-box">High energy reserve — great time to train.</div>',
                            unsafe_allow_html=True)
            elif body_battery < 30:
                st.markdown('<div class="warning-box">Very low battery — prioritise sleep and rest.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card">Body Battery: {body_battery}%</div>',
                            unsafe_allow_html=True)

    # --- Load stats row ---
    st.markdown("---")
    if status:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Training Streak", f"{status.get('consecutive_training_days', 0)} days")
        sc2.metric("This Week's Load", f"{status.get('this_week_load', 0):.1f}")
        sc3.metric("Last Week's Load", f"{status.get('last_week_load', 0):.1f}")
        load_delta = status.get("load_change_pct", 0.0)
        sc4.metric(
            "Load Change",
            f"{load_delta:+.1f}%",
            delta=f"{load_delta:+.1f}%",
            delta_color="inverse" if load_delta > 20 else "normal",
        )
        st.markdown("")

    # --- Sleep & Recovery (Apple Health first, Garmin fallback) ---
    st.markdown("---")
    today_health = data.get("today_health")
    _garmin_td = data.get("garmin_today") or {}

    # Merge: use Apple Health when available, Garmin as fallback
    _sleep_src = "Apple Health"
    if today_health and any(today_health.get(k) for k in (
        "sleep_duration_hours", "sleep_deep_hours", "sleep_rem_hours", "resting_hr"
    )):
        h = today_health
    elif _garmin_td.get("sleep_duration_hours"):
        h = {
            "sleep_duration_hours": _garmin_td.get("sleep_duration_hours"),
            "sleep_deep_hours": _garmin_td.get("sleep_deep_hours"),
            "sleep_rem_hours": _garmin_td.get("sleep_rem_hours"),
            "sleep_core_hours": _garmin_td.get("sleep_light_hours"),
            "sleep_awake_hours": None,
            "resting_hr": _garmin_td.get("resting_hr"),
        }
        _sleep_src = "Garmin"
    else:
        h = None

    st.subheader(f"Last Night's Sleep & Recovery {'(Garmin)' if _sleep_src == 'Garmin' else ''}")
    if h:
        sleep_total = h.get("sleep_duration_hours")
        sleep_deep = h.get("sleep_deep_hours")
        sleep_rem = h.get("sleep_rem_hours")
        sleep_core = h.get("sleep_core_hours")
        sleep_awake = h.get("sleep_awake_hours", 0) or 0
        rhr = h.get("resting_hr")

        hc1, hc2, hc3, hc4 = st.columns(4)
        if sleep_total:
            sleep_color = "normal" if sleep_total >= 7 else "inverse"
            hc1.metric("Total Sleep", f"{sleep_total:.1f} h",
                       delta="Good" if sleep_total >= 7 else "Low" if sleep_total < 6 else "OK",
                       delta_color=sleep_color)
        if sleep_deep:
            deep_color = "normal" if sleep_deep >= 1.0 else "inverse"
            hc2.metric("Deep Sleep", f"{sleep_deep:.1f} h",
                       delta="Good" if sleep_deep >= 1.0 else "Low",
                       delta_color=deep_color)
        if sleep_rem:
            hc3.metric("REM Sleep", f"{sleep_rem:.1f} h")
        if rhr:
            hc4.metric("Resting HR", f"{rhr} bpm")

        # Sleep stages stacked bar (if stage data available)
        stages = {k: v for k, v in {
            "Deep": sleep_deep, "REM": sleep_rem,
            "Core/Light": sleep_core, "Awake": sleep_awake if sleep_awake else None,
        }.items() if v}
        if len(stages) > 1:
            import plotly.graph_objects as go
            stage_colors = {
                "Deep": "#4299E1", "REM": "#9F7AEA",
                "Core/Light": "#68D391", "Awake": "#FC8181",
            }
            fig_sleep = go.Figure()
            for stage, val in stages.items():
                fig_sleep.add_trace(go.Bar(
                    x=[val], y=["Sleep Stages"], orientation="h",
                    name=stage, marker_color=stage_colors.get(stage, "#ccc"),
                    text=f"{val:.1f}h", textposition="inside",
                ))
            fig_sleep.update_layout(
                barmode="stack", height=100, margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True, legend=dict(orientation="h", y=-0.3),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Hours", showgrid=False),
                yaxis=dict(showticklabels=False),
            )
            st.plotly_chart(fig_sleep, use_container_width=True)

        # Readiness impact note — show all Garmin penalties
        factors = data.get("readiness", {}).get("factors", {})
        sleep_pen = factors.get("sleep_penalty", 0)
        deep_pen = factors.get("deep_sleep_penalty", 0)
        rhr_pen = factors.get("resting_hr_penalty", 0)
        bb_pen = factors.get("body_battery_penalty", 0)
        bb_bon = factors.get("body_battery_bonus", 0)
        stress_pen = factors.get("stress_penalty", 0)
        rt_pen = factors.get("recovery_time_penalty", 0)
        total_penalty = sleep_pen + deep_pen + rhr_pen + bb_pen + stress_pen + rt_pen
        total_bonus = bb_bon
        net = total_bonus - total_penalty
        if total_penalty > 0 or total_bonus > 0:
            parts = []
            if sleep_pen: parts.append(f"sleep: -{sleep_pen:.0f}")
            if deep_pen: parts.append(f"deep sleep: -{deep_pen:.0f}")
            if rhr_pen: parts.append(f"resting HR: -{rhr_pen:.0f}")
            if bb_pen: parts.append(f"body battery: -{bb_pen:.0f}")
            if bb_bon: parts.append(f"body battery bonus: +{bb_bon:.0f}")
            if stress_pen: parts.append(f"stress: -{stress_pen:.0f}")
            if rt_pen: parts.append(f"recovery time: -{rt_pen:.0f}")
            color = "#F56565" if net < 0 else "#48BB78"
            sign = f"{net:+.0f}"
            st.markdown(
                f'<div style="background:rgba(245,101,101,0.08);border-left:4px solid {color};'
                f'border-radius:6px;padding:10px 14px;margin:6px 0;font-size:0.9em;">'
                f'Wearable data adjusted readiness by <b>{sign} points</b> '
                f'({", ".join(parts)})'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="background:rgba(99,179,237,0.08);border-left:4px solid #63B3ED;'
            'border-radius:6px;padding:12px 16px;color:#A0AEC0;">'
            'No sleep data for today yet. Sync Garmin data or set up the iOS Shortcut to '
            'automatically improve readiness accuracy.'
            '</div>',
            unsafe_allow_html=True,
        )

    # --- Load trend chart ---
    st.markdown("---")
    st.subheader("Training Load Trend (all activity types, 8 weeks)")
    st.plotly_chart(make_load_trend_chart(load_trend), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Long Runs
# ---------------------------------------------------------------------------


def render_long_runs(data: Dict[str, Any]) -> None:
    long_runs = data.get("long_runs", [])
    prs = data.get("personal_records", {})
    est_hm = data.get("estimated_hm_str")
    total = data.get("total_activities", 0)

    if total == 0:
        st.info("No activities yet — sync to see your long run history.")
        return

    st.plotly_chart(make_long_run_progression(long_runs), use_container_width=True)

    st.markdown("---")
    st.subheader("Personal Records (Estimated via Riegel Formula)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("5K", prs.get("5k_str", "–"))
    col2.metric("10K", prs.get("10k_str", "–"))
    col3.metric("Half Marathon", prs.get("half_marathon_str", "–"))
    col4.metric("HM Projection", est_hm or "–")

    if not any(prs.get(f"{d}_str") for d in ["5k", "10k", "half_marathon"]):
        st.caption(
            "PRs are estimated from your runs using the Riegel endurance formula. "
            "Run at least one 5km to see estimates."
        )


# ---------------------------------------------------------------------------
# Tab: All Activities
# ---------------------------------------------------------------------------

_TYPE_ICONS: Dict[str, str] = {
    "Run": "🏃",
    "Ride": "🚴",
    "WeightTraining": "🏋️",
    "Walk": "🚶",
    "Yoga": "🧘",
    "Swim": "🏊",
    "Hike": "🥾",
    "Elliptical": "⚙️",
    "Pilates": "🤸",
    "Crossfit": "💪",
}


def _fmt_duration(seconds: Optional[float]) -> str:
    if not seconds:
        return "–"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sc = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {sc:02d}s"


def _fmt_dist(meters: Optional[float]) -> str:
    if not meters or meters < 50:
        return "–"
    return f"{meters / 1000:.2f} km"


def render_all_activities(data: Dict[str, Any]) -> None:
    type_counts: Dict[str, int] = data.get("activity_type_counts", {})
    all_types = sorted(type_counts.keys(), key=lambda t: -type_counts[t])

    if not all_types:
        st.info("No activities yet — sync to get started.")
        return

    # ── Type filter ───────────────────────────────────────────────────────
    type_options = ["All"] + all_types
    icons = [_TYPE_ICONS.get(t, "🏅") for t in all_types]
    display_options = ["All"] + [f"{_TYPE_ICONS.get(t, '🏅')} {t} ({type_counts[t]})" for t in all_types]

    selected_display = st.selectbox("Filter by activity type", display_options, key="activity_type_filter")
    selected_type = None if selected_display == "All" else all_types[display_options.index(selected_display) - 1]

    # ── Summary cards ─────────────────────────────────────────────────────
    activities = get_all_activities_cached(selected_type)
    if not activities:
        st.info("No activities found for this type.")
        return

    df = pd.DataFrame(activities)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["distance_km"] = df["distance"] / 1000
    df["duration_str"] = df["moving_time"].apply(_fmt_duration)
    df["dist_str"] = df["distance"].apply(_fmt_dist)
    df["week"] = df["start_date"].dt.strftime("%G-W%V")
    df["date"] = df["start_date"].dt.strftime("%Y-%m-%d")

    total_time_h = df["moving_time"].sum() / 3600
    total_dist_km = df["distance_km"].sum()
    has_distance = total_dist_km > 1.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Activities", len(df))
    c2.metric("Total Time", f"{total_time_h:.1f} hrs")
    c3.metric("Total Distance", f"{total_dist_km:.1f} km" if has_distance else "–")
    if "average_heartrate" in df.columns:
        avg_hr = df["average_heartrate"].dropna()
        c4.metric("Avg Heart Rate", f"{avg_hr.mean():.0f} bpm" if not avg_hr.empty else "–")

    st.markdown("---")

    # ── Volume over time chart ────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        if selected_type in (None, "Run", "Ride", "Walk", "Hike"):
            # Distance-based chart
            weekly = df.groupby("week").agg(
                distance_km=("distance_km", "sum"),
                count=("strava_id", "count")
            ).reset_index().sort_values("week").tail(16)
            fig = px.bar(
                weekly, x="week", y="distance_km",
                title="Weekly Volume (km)",
                labels={"distance_km": "Distance (km)", "week": "Week"},
                color_discrete_sequence=[COLOR_PRIMARY],
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(xaxis_tickangle=-45, margin=dict(l=40, r=20, t=50, b=60))
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Duration-based chart
            df["duration_hrs"] = df["moving_time"] / 3600
            weekly = df.groupby("week").agg(
                duration_hrs=("duration_hrs", "sum"),
                count=("strava_id", "count")
            ).reset_index().sort_values("week").tail(16)
            fig = px.bar(
                weekly, x="week", y="duration_hrs",
                title="Weekly Volume (hours)",
                labels={"duration_hrs": "Duration (hrs)", "week": "Week"},
                color_discrete_sequence=[COLOR_PRIMARY],
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(xaxis_tickangle=-45, margin=dict(l=40, r=20, t=50, b=60))
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Activity type breakdown pie (only when showing All)
        if not selected_type and type_counts:
            pie_df = pd.DataFrame([
                {"type": f"{_TYPE_ICONS.get(t,'🏅')} {t}", "count": c}
                for t, c in type_counts.items()
            ])
            fig = px.pie(
                pie_df, names="type", values="count",
                title="Activity Mix",
                template=PLOTLY_TEMPLATE,
                hole=0.4,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            # HR trend for the type if available
            hr_data = df.dropna(subset=["average_heartrate"]).sort_values("start_date").tail(20)
            if not hr_data.empty:
                fig = px.line(
                    hr_data, x="date", y="average_heartrate",
                    title="Heart Rate Trend",
                    labels={"average_heartrate": "Avg HR (bpm)", "date": "Date"},
                    markers=True,
                    color_discrete_sequence=[COLOR_PRIMARY],
                    template=PLOTLY_TEMPLATE,
                )
                fig.update_layout(xaxis_tickangle=-45, margin=dict(l=40, r=20, t=50, b=60))
                st.plotly_chart(fig, use_container_width=True)

    # ── Activities table ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"Recent Activities — {selected_display}")

    table_cols = ["date", "name", "activity_type", "dist_str", "duration_str", "average_heartrate"]
    col_names  = ["Date", "Name", "Type", "Distance", "Duration", "Avg HR"]
    if "average_cadence" in df.columns:
        table_cols.append("average_cadence")
        col_names.append("SPM")
    display_df = df[table_cols].copy()
    display_df.columns = col_names
    display_df["Avg HR"] = display_df["Avg HR"].apply(lambda x: f"{x:.0f}" if pd.notna(x) and x else "–")
    display_df["Type"] = display_df["Type"].apply(lambda t: f"{_TYPE_ICONS.get(t, '🏅')} {t}")
    if "SPM" in display_df.columns:
        display_df["SPM"] = display_df["SPM"].apply(lambda x: f"{x:.0f}" if pd.notna(x) and x else "–")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    # ── Excel download ────────────────────────────────────────────────────
    import io
    export_cols = {
        "date": "Date",
        "name": "Name",
        "activity_type": "Type",
        "dist_str": "Distance",
        "duration_str": "Duration",
        "average_heartrate": "Avg HR (bpm)",
        "distance_km": "Distance (km)",
        "average_cadence": "SPM",
    }
    available = [c for c in export_cols if c in df.columns]
    export_df = df[available].rename(columns=export_cols)
    export_df["Avg HR (bpm)"] = export_df["Avg HR (bpm)"].apply(
        lambda x: round(x, 0) if pd.notna(x) and x else None
    )
    if "SPM" in export_df.columns:
        export_df["SPM"] = export_df["SPM"].apply(
            lambda x: round(x, 0) if pd.notna(x) and x else None
        )
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Activities")
    filename = f"activities_{selected_display.replace(' ', '_').replace('(', '').replace(')', '')}.xlsx"
    st.download_button(
        label="⬇️ Download as Excel",
        data=buf.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ---------------------------------------------------------------------------
# Tab: Training Intelligence
# ---------------------------------------------------------------------------


def _trend_badge(trend: str) -> str:
    return {"improving": "🟢 Improving", "declining": "🔴 Declining", "stable": "🟡 Stable"}.get(trend, "–")


def render_training_intelligence(data: Dict[str, Any]) -> None:
    if data.get("total_activities", 0) == 0:
        st.info("No activities yet — sync your Strava data to see training intelligence.")
        return

    ts = data.get("training_state", {})
    readiness = data.get("readiness", {})
    injury = data.get("injury_risk", {})
    race_pred = data.get("race_predictions", {})
    weekly_plan = data.get("weekly_plan", {})

    # ── Status alerts ─────────────────────────────────────────────────────
    alerts = []
    if ts.get("overreaching"):
        alerts.append(("🔴 Overreaching Detected", "Acute load significantly exceeds chronic baseline. Reduce intensity and prioritise recovery.", "warning-box"))
    if ts.get("detraining"):
        alerts.append(("🟡 Detraining Risk", "Training load has dropped >25% over the last 3 weeks. Fitness may be declining.", "warning-box"))
    if ts.get("undertraining"):
        alerts.append(("📉 Under-Training", "Volume is below optimal for your chronic baseline. Safe to increase load 5–10%.", "metric-card"))
    if injury.get("risk") == "high":
        alerts.append(("⚠️ High Injury Risk", injury.get("summary", ""), "warning-box"))

    for title, msg, css in alerts:
        st.markdown(f'<div class="{css}"><b>{title}</b> — {msg}</div>', unsafe_allow_html=True)
    if alerts:
        st.markdown("")

    # ── Core metrics row ──────────────────────────────────────────────────
    st.subheader("Training State Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Consistency", f"{ts.get('consistency_score', 0):.0f}%",
              help="% of last 8 weeks with at least 1 run")
    c2.metric("Monotony", f"{ts.get('monotony_score', 0):.2f}",
              help="mean/std daily load (< 1.5 ideal, > 2.0 risky)")
    c3.metric("Strain Score", f"{ts.get('strain_score', 0):.1f}",
              help="Weekly load × monotony (Banister strain)")
    c4.metric("Aerobic Fitness", f"{ts.get('aerobic_fitness_score', 0):.0f}/100",
              help="Pace-to-HR efficiency proxy (higher = more fit)")
    c5.metric("Recovery Debt", f"{ts.get('recovery_debt', 0):+.2f}",
              help="(ATL-CTL)/CTL — positive = accumulated fatigue")

    st.markdown("---")

    # ── Efficiency trends ─────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("HR Efficiency Trend")
        hr_trend_data = ts.get("hr_efficiency_trend", {})
        st.caption(f"Trend: {_trend_badge(hr_trend_data.get('trend', 'stable'))}")
        st.caption("HR per km over last 20 runs — declining = improving aerobic fitness")
        chart_data = hr_trend_data.get("data", [])
        if chart_data:
            df_hr = pd.DataFrame(chart_data)
            fig = px.line(df_hr, x="start_date", y="hr_per_km",
                          template=PLOTLY_TEMPLATE,
                          labels={"hr_per_km": "HR per km", "start_date": "Date"},
                          markers=True, color_discrete_sequence=[COLOR_BLUE])
            fig.update_layout(height=260, margin=dict(l=40, r=20, t=30, b=50), xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Insufficient HR data (need 4+ runs with HR)")

    with col_r:
        st.subheader("Pace Efficiency Trend")
        pe_trend_data = ts.get("pace_efficiency_trend", {})
        st.caption(f"Trend: {_trend_badge(pe_trend_data.get('trend', 'stable'))}")
        st.caption("Pace per HR unit — declining = more speed for the same HR effort")
        pe_chart = pe_trend_data.get("data", [])
        if pe_chart:
            df_pe = pd.DataFrame(pe_chart)
            fig = px.line(df_pe, x="start_date", y="pace_per_hr_unit",
                          template=PLOTLY_TEMPLATE,
                          labels={"pace_per_hr_unit": "Pace/HR unit", "start_date": "Date"},
                          markers=True, color_discrete_sequence=[COLOR_GREEN])
            fig.update_layout(height=260, margin=dict(l=40, r=20, t=30, b=50), xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Insufficient HR+pace data (need 4+ eligible runs)")

    st.markdown("---")

    # ── VO2 Max Trend (Garmin) ─────────────────────────────────────────────
    vo2max_data = get_garmin_daily(days=90)
    vo2max_rows = [r for r in vo2max_data if r.get("vo2max") is not None]
    if vo2max_rows:
        st.subheader("VO2 Max Trend (Garmin)")
        df_vo2 = pd.DataFrame(vo2max_rows)[["date", "vo2max"]].copy()
        df_vo2["date"] = pd.to_datetime(df_vo2["date"])
        df_vo2 = df_vo2.sort_values("date").drop_duplicates("date")
        latest_vo2 = df_vo2["vo2max"].iloc[-1]
        col_v1, col_v2 = st.columns([3, 1])
        with col_v1:
            fig_vo2 = px.line(
                df_vo2, x="date", y="vo2max",
                template=PLOTLY_TEMPLATE,
                labels={"vo2max": "VO2 Max (mL/kg/min)", "date": "Date"},
                markers=True,
                color_discrete_sequence=[COLOR_BLUE],
            )
            fig_vo2.add_hline(y=50, line_dash="dash", line_color="#48BB78",
                              annotation_text="Elite (50+)", annotation_position="top right")
            fig_vo2.add_hline(y=42, line_dash="dash", line_color="#ECC94B",
                              annotation_text="Average (42)", annotation_position="bottom right")
            fig_vo2.update_layout(height=220, margin=dict(l=40, r=20, t=30, b=40))
            st.plotly_chart(fig_vo2, use_container_width=True)
        with col_v2:
            race_pred = data.get("race_predictions", {})
            vo2_used = race_pred.get("vo2max_used")
            st.metric("Current VO2 Max", f"{latest_vo2:.1f}", help="From Garmin Forerunner")
            trend_dir = "improving" if len(df_vo2) > 1 and df_vo2["vo2max"].iloc[-1] > df_vo2["vo2max"].iloc[0] else "stable"
            st.caption(_trend_badge(trend_dir))
            if vo2_used:
                mara = race_pred.get("predictions", {}).get("marathon", {})
                if mara.get("source") == "vo2max estimate":
                    st.caption(f"Marathon est: **{mara.get('time_str', '–')}**")
                    st.caption("(VO2max VDOT estimate)")
        st.markdown("---")

    # ── Long run analysis ─────────────────────────────────────────────────
    col_fade, col_cad = st.columns(2)

    with col_fade:
        st.subheader("Long Run Pace Fade")
        fade = ts.get("long_run_pace_fade", {})
        verdict_labels = {
            "negative_split": "🟢 Negative Split — excellent pacing",
            "even_pacing": "🟢 Even Pacing — good endurance",
            "moderate_fade": "🟡 Moderate Fade — endurance can improve",
            "significant_fade": "🔴 Significant Fade — glycogen/endurance limiter",
            "insufficient_data": "⬜ No long-run splits data yet",
        }
        verdict = fade.get("verdict", "insufficient_data")
        fade_pct = fade.get("fade_pct")
        st.markdown(verdict_labels.get(verdict, "–"))
        if fade_pct is not None:
            st.metric("Average Fade", f"{fade_pct:+.1f}%",
                      help="% slower in last third vs first third of long runs (positive = slowing)")
        if fade.get("run_name"):
            st.caption(f"Based on {fade.get('runs_analysed', 1)} long run(s) — most recent: {fade['run_name']}")

    with col_cad:
        st.subheader("Cadence Analysis")
        cad = ts.get("cadence_analysis", {})
        avg_cad = cad.get("avg_cadence")
        cad_trend = cad.get("trend", "stable")
        if avg_cad:
            ideal_low, ideal_high = 170, 180
            if avg_cad < ideal_low:
                cad_note = f"⬆️ Below target — aim to increase toward {ideal_low}–{ideal_high} spm"
            elif avg_cad > ideal_high:
                cad_note = f"⬇️ Above target — slightly high but generally fine"
            else:
                cad_note = "✅ In optimal range (170–180 spm)"
            st.metric("Avg Cadence", f"{avg_cad:.0f} spm", delta=_trend_badge(cad_trend), delta_color="off")
            st.caption(cad_note)
        else:
            st.caption("No cadence data available — enable cadence recording in your watch/phone.")

    st.markdown("---")

    # ── Consistency heatmap ───────────────────────────────────────────────
    st.subheader("Weekly Consistency (last 16 weeks)")
    weekly_cons = ts.get("weekly_consistency", [])
    if weekly_cons:
        df_cons = pd.DataFrame(weekly_cons).tail(16)
        fig = px.bar(
            df_cons, x="week", y="activity_count",
            color="total_load",
            color_continuous_scale="Oranges",
            template=PLOTLY_TEMPLATE,
            labels={"activity_count": "Activities", "week": "Week", "total_load": "Load"},
            title="",
        )
        fig.update_layout(height=220, margin=dict(l=40, r=20, t=20, b=60), xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Sync activities to see your consistency heatmap.")

    st.markdown("---")

    # ── Race predictions ──────────────────────────────────────────────────
    st.subheader("Race Predictions")
    predictions = race_pred.get("predictions", {})
    traj = race_pred.get("fitness_trajectory", "stable")
    overall_conf = race_pred.get("overall_confidence", 0)

    traj_label = {"improving": "🟢 Improving", "declining": "🔴 Declining", "stable": "🟡 Stable"}.get(traj, "–")
    col_t1, col_t2 = st.columns([2, 3])
    col_t1.metric("Fitness Trajectory", traj_label, delta_color="off")
    col_t2.metric("Prediction Confidence", f"{overall_conf}/100",
                  help="Based on consistency, aerobic fitness data quality, and freshness")

    dist_labels = {"5k": "5K", "10k": "10K", "half_marathon": "Half Marathon", "marathon": "Marathon"}
    pred_cols = st.columns(4)
    for i, (key, label) in enumerate(dist_labels.items()):
        pred = predictions.get(key, {})
        pred_cols[i].metric(
            label,
            pred.get("time_str", "–"),
            delta=f"{pred.get('confidence', 0)}% conf" if pred.get("time_str", "–") != "–" else None,
            delta_color="off",
        )

    st.markdown("---")

    # ── Adaptive weekly plan ──────────────────────────────────────────────
    st.subheader("This Week's Adaptive Plan")
    if weekly_plan:
        phase_labels = {
            "recovery_week": "🛑 Recovery Week",
            "consolidation": "🟡 Consolidation",
            "maintenance": "🟢 Maintenance",
            "build": "📈 Build Phase",
            "peak_build": "🔥 Peak Build",
            "pre_taper": "📉 Pre-Taper",
            "taper": "⬇️ Taper",
            "race_week": "🏁 Race Week",
        }
        phase = weekly_plan.get("phase", "maintenance")
        st.markdown(f"**Phase:** {phase_labels.get(phase, phase.replace('_', ' ').title())}")

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Target km", f"{weekly_plan.get('recommended_weekly_km', 0):.1f} km")
        p2.metric("Training days", weekly_plan.get("total_training_days", 0))
        p3.metric("Quality sessions", weekly_plan.get("quality_days", 0))
        p4.metric("Long run", f"{weekly_plan.get('long_run_km', 0):.1f} km")

        if weekly_plan.get("interval_workout"):
            st.markdown(f"**Intervals:** {weekly_plan['interval_workout']}")
        if weekly_plan.get("threshold_workout"):
            st.markdown(f"**Threshold:** {weekly_plan['threshold_workout']}")
        st.markdown(f"**Easy runs:** {weekly_plan.get('easy_pace_note', '')}")

        if weekly_plan.get("long_run_note"):
            st.markdown(
                f'<div class="insight-box">📏 <b>Long Run:</b> {weekly_plan["long_run_note"]}</div>',
                unsafe_allow_html=True,
            )
        if weekly_plan.get("recovery_note"):
            st.markdown(
                f'<div class="metric-card">💤 <b>Recovery:</b> {weekly_plan["recovery_note"]}</div>',
                unsafe_allow_html=True,
            )

    # ── Injury risk detail ────────────────────────────────────────────────
    if injury.get("factors"):
        st.markdown("---")
        st.subheader(f"Injury Risk: {injury.get('risk_label', '–')}")
        for factor in injury["factors"]:
            st.markdown(f"• {factor}")
        if injury.get("recommendations"):
            st.markdown("**Recommendations:**")
            for rec in injury["recommendations"]:
                st.markdown(f"→ {rec}")


# ---------------------------------------------------------------------------
# Tab: Workouts
# ---------------------------------------------------------------------------


def render_workouts(data: Dict[str, Any]) -> None:
    readiness = data.get("readiness", {})
    r_score = readiness.get("readiness_score", 50.0)
    r_desc = readiness.get("tier_description", "")

    # Readiness summary
    r_color = readiness.get("tier_color", "#63B3ED")
    st.markdown(
        f'<div style="background:rgba(99,179,237,0.1);border-left:4px solid {r_color};'
        f'border-radius:8px;padding:14px 18px;margin:8px 0;">'
        f'<b>Readiness: {r_score:.0f}/100 — {readiness.get("tier_label", "")}</b><br/>'
        f'{r_desc}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    col1, col2 = st.columns([2, 1])
    with col1:
        workout_type = st.selectbox(
            "Workout type",
            options=["easy", "intervals", "threshold", "long_run", "recovery"],
            format_func=lambda x: {
                "easy": "🏃 Easy Run",
                "intervals": "⚡ Interval Session",
                "threshold": "🔥 Threshold / Tempo",
                "long_run": "🛣️ Long Run",
                "recovery": "💤 Recovery Run",
            }.get(x, x),
        )
    with col2:
        available_min = st.number_input("Available time (min)", min_value=20, max_value=180, value=60, step=5)

    if st.button("Generate Workout", type="primary", use_container_width=True):
        with st.spinner("Generating workout…"):
            result = _api_post_json(
                "/workout/generate",
                {"workout_type": workout_type, "available_minutes": available_min},
            )

        if not result:
            st.error("Could not generate workout. Is the backend running?")
            return

        st.session_state["last_workout"] = result

    workout = st.session_state.get("last_workout")
    if not workout:
        st.caption("Select a workout type and click Generate Workout.")
        return

    st.markdown("---")
    st.subheader(workout.get("name", "Workout"))

    if workout.get("readiness_warning"):
        st.markdown(
            f'<div class="warning-box">{workout["readiness_warning"]}</div>',
            unsafe_allow_html=True,
        )

    # Key info
    k1, k2 = st.columns(2)
    if workout.get("target_hr_zone"):
        k1.markdown(f"**Target HR:** {workout['target_hr_zone']}")
    if workout.get("based_on_pace"):
        k2.markdown(f"**Based on recent pace:** {workout['based_on_pace']}")

    # Warmup
    if workout.get("warmup"):
        st.markdown(
            f'<div class="metric-card">🔥 <b>Warm-up:</b> {workout["warmup"]}</div>',
            unsafe_allow_html=True,
        )

    # Workout blocks
    st.markdown("**Workout:**")
    for i, block in enumerate(workout.get("blocks", []), 1):
        desc = block.get("description", "")
        pace = block.get("target_pace", "")
        effort = block.get("effort", "")
        duration = block.get("duration_min")
        rest = block.get("rest", "")

        detail_parts = []
        if pace and pace != "–":
            detail_parts.append(f"Target pace: **{pace}**")
        if duration:
            detail_parts.append(f"Duration: {duration} min")
        if rest:
            detail_parts.append(f"Rest: {rest}")

        st.markdown(
            f'<div class="insight-box">'
            f'<b>Block {i}: {desc}</b><br/>'
            f'{"&nbsp;&nbsp;•&nbsp;".join(detail_parts)}<br/>'
            f'<i>Effort: {effort}</i>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Cooldown
    if workout.get("cooldown"):
        st.markdown(
            f'<div class="metric-card">❄️ <b>Cool-down:</b> {workout["cooldown"]}</div>',
            unsafe_allow_html=True,
        )

    # Distance estimate
    if workout.get("estimated_distance_km"):
        st.metric("Estimated distance", f"{workout['estimated_distance_km']:.1f} km")

    # Coaching note
    if workout.get("coaching_note"):
        st.markdown(
            f'<div class="metric-card" style="margin-top:12px;">💡 <b>Coach says:</b> {workout["coaching_note"]}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab: Profile
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def _load_profile() -> Dict[str, Any]:
    result = _api_get("/profile", timeout=10)
    return result or {}


def render_profile() -> None:
    st.subheader("Athlete Profile")
    st.caption("Your profile personalises AI coaching, pace targets, race predictions, and training plans.")

    profile = _load_profile()

    with st.form("athlete_profile_form"):
        st.markdown("#### Basic Info")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", min_value=10, max_value=90, value=int(profile.get("age") or 30), step=1)
        weight = c2.number_input("Weight (kg)", min_value=30.0, max_value=200.0,
                                  value=float(profile.get("weight_kg") or 70.0), step=0.5)
        experience = c3.selectbox(
            "Running experience",
            ["beginner", "intermediate", "advanced"],
            index=["beginner", "intermediate", "advanced"].index(
                profile.get("running_experience") or "intermediate"
            ),
        )

        st.markdown("#### Heart Rate")
        h1, h2 = st.columns(2)
        resting_hr = h1.number_input("Resting HR (bpm)", min_value=30, max_value=100,
                                      value=int(profile.get("resting_hr") or 55), step=1)
        max_hr = h2.number_input("Max HR (bpm)", min_value=100, max_value=230,
                                  value=int(profile.get("max_hr") or 185), step=1)

        st.markdown("#### Goals & Race")
        g1, g2 = st.columns(2)
        goal_options = ["hm", "10k", "marathon", "base_building", "weight_loss", "fitness"]
        goal_labels = {
            "hm": "Half Marathon", "10k": "10K", "marathon": "Marathon",
            "base_building": "Base Building", "weight_loss": "Weight Loss", "fitness": "General Fitness",
        }
        current_goal = profile.get("primary_goal") or "hm"
        if current_goal not in goal_options:
            current_goal = "hm"
        primary_goal = g1.selectbox(
            "Primary goal",
            goal_options,
            index=goal_options.index(current_goal),
            format_func=lambda x: goal_labels.get(x, x),
        )
        target_race_date_str = profile.get("target_race_date") or ""
        if target_race_date_str:
            try:
                import datetime as _datetime
                target_default = _datetime.date.fromisoformat(target_race_date_str[:10])
            except Exception:
                target_default = None
        else:
            target_default = None
        target_race_date = g2.date_input("Target race date (optional)", value=target_default)

        st.markdown("#### Training Preferences")
        t1, t2, t3 = st.columns(3)
        preferred_km = t1.number_input("Preferred weekly km", min_value=0.0, max_value=200.0,
                                        value=float(profile.get("preferred_weekly_km") or 40.0), step=5.0)
        weekly_hrs = t2.number_input("Available training hours/week", min_value=1.0, max_value=30.0,
                                      value=float(profile.get("weekly_training_hours") or 6.0), step=0.5)
        strength_days = t3.number_input("Strength training days/week", min_value=0, max_value=7,
                                         value=int(profile.get("strength_training_days") or 2), step=1)

        day_options = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        current_day = profile.get("preferred_long_run_day") or "saturday"
        if current_day not in day_options:
            current_day = "saturday"
        long_run_day = st.selectbox(
            "Preferred long run day",
            day_options,
            index=day_options.index(current_day),
            format_func=str.capitalize,
        )

        st.markdown("#### Health & Injury")
        injury_history = st.text_area(
            "Injury history (optional — helps the AI coach tailor advice)",
            value=profile.get("injury_history") or "",
            placeholder="e.g. Left knee ITB syndrome 2023, plantar fasciitis 2022",
            height=80,
        )
        notes = st.text_area(
            "Additional notes for your coach (optional)",
            value=profile.get("notes") or "",
            placeholder="e.g. Training for Berlin marathon, prefer morning runs, avoid hills",
            height=80,
        )

        st.markdown("#### Sleep & Recovery (optional — future wearable integration)")
        sl1, sl2, sl3 = st.columns(3)
        sleep_hours = sl1.number_input("Last night's sleep (hrs)", min_value=0.0, max_value=14.0,
                                        value=float(profile.get("sleep_hours") or 0.0), step=0.5)
        resting_hr_today = sl2.number_input("This morning's resting HR", min_value=0, max_value=120,
                                             value=int(profile.get("resting_hr_today") or 0), step=1)
        hrv = sl3.number_input("HRV (ms, if available)", min_value=0.0, max_value=300.0,
                                value=float(profile.get("hrv") or 0.0), step=1.0)

        submitted = st.form_submit_button("Save Profile", type="primary", use_container_width=True)

    if submitted:
        payload: Dict[str, Any] = {
            "age": int(age),
            "weight_kg": float(weight),
            "running_experience": experience,
            "resting_hr": int(resting_hr),
            "max_hr": int(max_hr),
            "primary_goal": primary_goal,
            "preferred_weekly_km": float(preferred_km),
            "weekly_training_hours": float(weekly_hrs),
            "strength_training_days": int(strength_days),
            "preferred_long_run_day": long_run_day,
            "injury_history": injury_history or None,
            "notes": notes or None,
        }
        if target_race_date:
            payload["target_race_date"] = str(target_race_date)
        if sleep_hours > 0:
            payload["sleep_hours"] = float(sleep_hours)
        if resting_hr_today > 0:
            payload["resting_hr_today"] = int(resting_hr_today)
        if hrv > 0:
            payload["hrv"] = float(hrv)

        result = _api_post_json("/profile", payload)
        if result and result.get("status") == "saved":
            st.success("Profile saved! Dashboard will use your profile on the next load.")
            _load_profile.clear()
            if "dashboard_data" in st.session_state:
                del st.session_state["dashboard_data"]
        else:
            st.error("Could not save profile. Check the backend logs.")

    # Clear chat memory section
    st.markdown("---")
    st.subheader("AI Coach Memory")
    st.caption("The AI coach remembers your recent conversations across sessions.")
    if st.button("Clear Chat Memory", help="Remove all stored chat history"):
        try:
            resp = requests.delete(f"{BACKEND_URL}/ai/chat/memory", timeout=10)
            if resp.ok:
                st.success("Chat memory cleared.")
            else:
                st.error(f"Failed to clear memory: {resp.status_code}")
        except Exception as exc:
            st.error(f"Error: {exc}")


# ---------------------------------------------------------------------------
# Tab: Sleep
# ---------------------------------------------------------------------------


def render_sleep() -> None:
    st.subheader("Sleep & Health Sync")

    history = get_sleep_history(days=30)
    _garmin_sleep = None

    # --- Sync status / Garmin fallback ---
    if not history:
        # Try Garmin as fallback
        garmin_history = get_garmin_sleep_history(days=30)
        if garmin_history:
            history = garmin_history
            _garmin_sleep = True
            st.caption("Showing Garmin sleep data (no Apple Health data synced yet)")
        else:
            st.markdown(
                '<div style="background:rgba(99,179,237,0.08);border-left:4px solid #63B3ED;'
                'border-radius:6px;padding:14px 18px;">'
                '<b>No sleep data synced yet.</b><br/>'
                'Sync Garmin data or run your iOS Shortcut manually. '
                'Once data arrives it will appear here.'
                '</div>',
                unsafe_allow_html=True,
            )
            return
    else:
        st.caption("Data synced from Apple Health via your iOS Shortcut")

    # Most recent entry
    latest = history[0]
    latest_date = latest.get("date", "")
    sleep_ok = latest.get("sleep_duration_hours") is not None
    rhr_ok = latest.get("resting_hr") is not None

    status_parts = []
    if sleep_ok:
        status_parts.append(f"Sleep: {latest.get('sleep_duration_hours'):.1f} h")
    if rhr_ok:
        status_parts.append(f"Resting HR: {latest.get('resting_hr')} bpm")
    stages = [
        ("Deep", latest.get("sleep_deep_hours")),
        ("REM", latest.get("sleep_rem_hours")),
        ("Core", latest.get("sleep_core_hours")),
    ]
    stage_parts = [f"{n}: {v:.1f}h" for n, v in stages if v]
    if stage_parts:
        status_parts.append("Stages — " + ", ".join(stage_parts))

    sync_color = "#48BB78" if sleep_ok else "#ECC94B"
    sync_msg = " · ".join(status_parts) if status_parts else "Partial data (sleep stages missing)"
    st.markdown(
        f'<div style="background:rgba(72,187,120,0.08);border-left:4px solid {sync_color};'
        f'border-radius:6px;padding:12px 16px;margin-bottom:12px;">'
        f'<b>Latest sync: {latest_date}</b><br/>{sync_msg}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Build DataFrame
    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # --- Sleep duration chart ---
    st.markdown("---")
    st.subheader("Sleep Duration (last 30 days)")
    sleep_rows = df[df["sleep_duration_hours"].notna()]
    if not sleep_rows.empty:
        colors = [
            "#48BB78" if h >= 7 else "#ECC94B" if h >= 6 else "#F56565"
            for h in sleep_rows["sleep_duration_hours"]
        ]
        fig_dur = go.Figure(go.Bar(
            x=sleep_rows["date"].dt.strftime("%b %d"),
            y=sleep_rows["sleep_duration_hours"],
            marker_color=colors,
            text=[f"{h:.1f}h" for h in sleep_rows["sleep_duration_hours"]],
            textposition="outside",
        ))
        fig_dur.add_hline(y=7, line_dash="dash", line_color="#48BB78",
                          annotation_text="Target 7h", annotation_position="top right")
        fig_dur.update_layout(
            height=280, margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Hours", range=[0, max(sleep_rows["sleep_duration_hours"].max() + 1, 9)]),
            xaxis=dict(title=""),
            showlegend=False,
        )
        st.plotly_chart(fig_dur, use_container_width=True)
        avg = sleep_rows["sleep_duration_hours"].mean()
        nights_ok = (sleep_rows["sleep_duration_hours"] >= 7).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Average Sleep", f"{avg:.1f} h")
        c2.metric("Nights ≥ 7h", f"{nights_ok}/{len(sleep_rows)}")
        if _garmin_sleep and "sleep_score" in df.columns:
            score_rows = df[df["sleep_score"].notna()]
            avg_score = score_rows["sleep_score"].mean() if not score_rows.empty else None
            c3.metric("Avg Sleep Score", f"{avg_score:.0f}/100" if avg_score else "–")
        else:
            c3.metric("Days Synced", len(history))
    else:
        st.info("Sleep duration data not yet available — sync Garmin or run the iOS Shortcut.")

    # --- Sleep stages stacked bar ---
    stage_rows = df[df[["sleep_deep_hours", "sleep_rem_hours", "sleep_core_hours"]].notna().any(axis=1)]
    if not stage_rows.empty:
        st.markdown("---")
        st.subheader("Sleep Stage Breakdown")
        fig_stages = go.Figure()
        stage_map = [
            ("Deep", "sleep_deep_hours", "#4299E1"),
            ("REM", "sleep_rem_hours", "#9F7AEA"),
            ("Core / Light", "sleep_core_hours", "#68D391"),
            ("Awake", "sleep_awake_hours", "#FC8181"),
        ]
        for label, col, color in stage_map:
            vals = stage_rows[col].fillna(0) if col in stage_rows.columns else [0] * len(stage_rows)
            fig_stages.add_trace(go.Bar(
                name=label,
                x=stage_rows["date"].dt.strftime("%b %d"),
                y=vals,
                marker_color=color,
            ))
        fig_stages.update_layout(
            barmode="stack", height=280, margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Hours"),
            xaxis=dict(title=""),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_stages, use_container_width=True)

    # --- Resting HR trend ---
    rhr_rows = df[df["resting_hr"].notna()]
    if not rhr_rows.empty:
        st.markdown("---")
        st.subheader("Resting Heart Rate Trend")
        fig_rhr = go.Figure(go.Scatter(
            x=rhr_rows["date"].dt.strftime("%b %d"),
            y=rhr_rows["resting_hr"],
            mode="lines+markers",
            line=dict(color="#63B3ED", width=2),
            marker=dict(size=7),
            fill="tozeroy",
            fillcolor="rgba(99,179,237,0.08)",
        ))
        fig_rhr.update_layout(
            height=220, margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="bpm"),
            xaxis=dict(title=""),
            showlegend=False,
        )
        st.plotly_chart(fig_rhr, use_container_width=True)
        avg_rhr = rhr_rows["resting_hr"].mean()
        min_rhr = rhr_rows["resting_hr"].min()
        max_rhr = rhr_rows["resting_hr"].max()
        r1, r2, r3 = st.columns(3)
        r1.metric("Average", f"{avg_rhr:.0f} bpm")
        r2.metric("Best (lowest)", f"{min_rhr:.0f} bpm")
        r3.metric("Highest", f"{max_rhr:.0f} bpm")

    # --- Raw data table ---
    st.markdown("---")
    st.subheader("Raw Sync Log")
    st.caption("Every row is one Shortcut run. Use this to verify what data is coming through.")
    display_df = df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df = display_df[[c for c in [
        "date", "sleep_duration_hours", "sleep_deep_hours",
        "sleep_rem_hours", "sleep_core_hours", "sleep_awake_hours", "resting_hr", "hrv"
    ] if c in display_df.columns]]
    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
    st.dataframe(display_df.sort_values("Date", ascending=False), use_container_width=True)


# ---------------------------------------------------------------------------
# FM Plan Tracker
# ---------------------------------------------------------------------------

_STATUS_ICON = {"completed": "✅", "missed": "❌", "upcoming": "⏳"}
_STATUS_COLOR = {"completed": "#276749", "missed": "#9b2c2c", "upcoming": "#744210"}


def _plan_badge(status: str) -> str:
    icon = _STATUS_ICON.get(status, "⏳")
    color = _STATUS_COLOR.get(status, "#333")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:0.75em;font-weight:600">{icon} {status.upper()}</span>'
    )


def render_fm_plan() -> None:
    st.header("FM Plan Tracker")
    st.caption("Full marathon training plan — Sep 2026 → Sep 2027 target.")

    # ---- plan status ----
    try:
        status_resp = requests.get(f"{BACKEND_URL}/plan/status", timeout=10)
        status_data = status_resp.json()
    except Exception as exc:
        st.error(f"Cannot reach backend: {exc}")
        return

    total_workouts = status_data.get("total_workouts", 0)

    if total_workouts == 0:
        st.info("No plan loaded yet. Click below to import from the xlsx file.")
        if st.button("Import Plan from File", type="primary"):
            with st.spinner("Importing plan…"):
                imp_resp = requests.post(f"{BACKEND_URL}/plan/import", timeout=30)
            if imp_resp.status_code == 200:
                r = imp_resp.json()
                st.success(f"Plan imported: {r.get('inserted', 0)} new, {r.get('updated', 0)} updated workouts.")
                st.rerun()
            else:
                st.error(f"Import failed: {imp_resp.text}")
        return

    # ---- header metrics ----
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Workouts", total_workouts)
    mc2.metric("Total Weeks", status_data.get("total_weeks", 0))
    mc3.metric("Planned KM", f"{status_data.get('total_planned_km', 0):.0f}")

    if st.button("Re-import Plan", help="Re-parse the xlsx and update DB"):
        with st.spinner("Re-importing…"):
            imp_resp = requests.post(f"{BACKEND_URL}/plan/import", timeout=30)
        if imp_resp.status_code == 200:
            r = imp_resp.json()
            st.success(f"Re-imported: {r.get('updated', 0)} updated, {r.get('inserted', 0)} new.")
            st.rerun()
        else:
            st.error(f"Import failed: {imp_resp.text}")

    st.markdown("---")

    # ---- weeks list ----
    try:
        weeks_resp = requests.get(f"{BACKEND_URL}/plan/weeks", timeout=10)
        weeks: list = weeks_resp.json()
    except Exception as exc:
        st.error(f"Failed to load weeks: {exc}")
        return

    if not weeks:
        st.warning("No weeks found. Import the plan first.")
        return

    # Default selection: current week if present, else last past week
    from datetime import date as _date
    current_iso = _date.today().strftime("%G-W%V")
    week_labels = [w["iso_week"] for w in weeks]
    phase_map = {w["iso_week"]: w["phase"] for w in weeks}
    planned_km_map = {w["iso_week"]: w["planned_km"] for w in weeks}

    default_idx = next(
        (i for i, w in enumerate(weeks) if w["iso_week"] == current_iso), None
    )
    if default_idx is None:
        # Pick last past week
        past = [i for i, w in enumerate(weeks) if w["phase"] == "past"]
        default_idx = past[-1] if past else 0

    def _week_label(w: dict) -> str:
        phase_tag = {"current": " (current)", "future": " (upcoming)", "past": ""}.get(w["phase"], "")
        return f"{w['iso_week']}  ·  {w['planned_km']:.0f} km planned{phase_tag}"

    selected_idx = st.selectbox(
        "Select training week",
        options=range(len(weeks)),
        index=default_idx,
        format_func=lambda i: _week_label(weeks[i]),
    )
    selected_week = weeks[selected_idx]["iso_week"]

    # ---- week detail ----
    try:
        week_resp = requests.get(f"{BACKEND_URL}/plan/week/{selected_week}", timeout=10)
        week_data = week_resp.json()
    except Exception as exc:
        st.error(f"Failed to load week detail: {exc}")
        return

    workouts = week_data.get("workouts", [])
    planned_km_total = week_data.get("planned_km", 0)
    actual_km_total = week_data.get("actual_km", 0)

    wc1, wc2, wc3 = st.columns(3)
    wc1.metric("Week Planned", f"{planned_km_total:.1f} km")
    wc2.metric("Week Actual", f"{actual_km_total:.1f} km")
    wc3.metric(
        "Distance Diff",
        f"{actual_km_total - planned_km_total:+.1f} km",
        delta_color="off",
    )

    st.markdown("")

    # ---- individual runs ----
    for workout in workouts:
        status = workout["status"]
        actual = workout.get("actual")
        strava_id = actual["strava_id"] if actual else None

        with st.container(border=True):
            row_top = st.columns([3, 2, 2, 1])
            with row_top[0]:
                st.markdown(
                    f"{_plan_badge(status)}&nbsp;&nbsp;<b>{workout['plan_date']}</b>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{workout['session_type']}** · {workout['distance_km']:.1f} km planned")
                if workout.get("details"):
                    st.caption(workout["details"])

            with row_top[1]:
                if actual:
                    st.metric("Actual Dist", f"{actual['distance_km']:.2f} km")
                    st.metric("Pace", actual.get("pace") or "N/A")
                else:
                    st.markdown("*No matching run found*")

            with row_top[2]:
                if actual:
                    hr = actual.get("avg_hr")
                    spm = actual.get("spm")
                    st.metric("Avg HR", f"{hr} bpm" if hr else "N/A")
                    st.metric("SPM", spm if spm else "N/A")

            with row_top[3]:
                if status == "completed" and strava_id:
                    review_key = f"plan_review_{strava_id}"
                    cached_review = workout.get("ai_review")
                    if not cached_review and review_key not in st.session_state:
                        if st.button("AI Review", key=f"btn_review_{strava_id}"):
                            with st.spinner("Generating…"):
                                r = requests.post(
                                    f"{BACKEND_URL}/plan/review/{strava_id}",
                                    timeout=30,
                                )
                            if r.status_code == 200:
                                st.session_state[review_key] = r.json()["review"]
                            st.rerun()

            # Show review (either from DB or generated this session)
            review_text = (
                workout.get("ai_review")
                or st.session_state.get(f"plan_review_{strava_id}")
            )
            if review_text:
                st.markdown(
                    f'<div class="insight-box">{review_text}</div>',
                    unsafe_allow_html=True,
                )

    # ---- weekly AI summary ----
    st.markdown("---")
    st.subheader(f"Weekly AI Summary — {selected_week}")

    summary_key = f"plan_weekly_summary_{selected_week}"
    saved_summary = week_data.get("weekly_summary")

    if not saved_summary and summary_key not in st.session_state:
        if st.button("Generate Weekly Summary", key=f"btn_summary_{selected_week}", type="primary"):
            with st.spinner("Generating weekly summary…"):
                sr = requests.post(
                    f"{BACKEND_URL}/plan/weekly-summary/{selected_week}",
                    timeout=60,
                )
            if sr.status_code == 200:
                st.session_state[summary_key] = sr.json()["summary"]
            elif sr.status_code == 404:
                st.warning("No workouts found for this week.")
            else:
                st.error(f"Failed: {sr.text}")
            st.rerun()
    else:
        summary_text = saved_summary or st.session_state.get(summary_key)
        if summary_text:
            st.markdown(
                f'<div class="insight-box">{summary_text}</div>',
                unsafe_allow_html=True,
            )
            if st.button(
                "Regenerate Summary",
                key=f"btn_regen_summary_{selected_week}",
                help="Generate a fresh summary (overwrites saved)",
            ):
                # Clear cached entry in DB by force-calling without cached check
                # We delete from session state so the next save goes through
                if summary_key in st.session_state:
                    del st.session_state[summary_key]
                with st.spinner("Regenerating…"):
                    # Delete existing from DB via direct re-post (server will overwrite)
                    sr = requests.post(
                        f"{BACKEND_URL}/plan/weekly-summary/{selected_week}?force=1",
                        timeout=60,
                    )
                if sr.status_code == 200:
                    st.session_state[summary_key] = sr.json()["summary"]
                st.rerun()

    # ---- monthly progress section ----
    st.markdown("---")
    st.subheader("Monthly Progress")

    try:
        months_resp = requests.get(f"{BACKEND_URL}/plan/months", timeout=10)
        months: list = months_resp.json()
    except Exception as exc:
        st.warning(f"Could not load months: {exc}")
        months = []

    if months:
        from datetime import date as _mdate
        cur_month_key = _mdate.today().strftime("%Y-%m")
        default_m = next(
            (i for i, m in enumerate(months) if m["year_month"] == cur_month_key), None
        )
        if default_m is None:
            past_m = [i for i, m in enumerate(months) if m["phase"] == "past"]
            default_m = past_m[-1] if past_m else 0

        def _month_label(m: dict) -> str:
            tag = {"current": " (current)", "future": " (upcoming)", "past": ""}.get(m["phase"], "")
            return f"{m['display']}{tag}"

        sel_month_idx = st.selectbox(
            "Select month",
            options=range(len(months)),
            index=default_m,
            format_func=lambda i: _month_label(months[i]),
            key="month_selector",
        )
        sel_month_key = months[sel_month_idx]["year_month"]

        try:
            mdata_resp = requests.get(f"{BACKEND_URL}/plan/month/{sel_month_key}", timeout=15)
            mdata = mdata_resp.json()
        except Exception as exc:
            st.error(f"Failed to load month data: {exc}")
            mdata = None

        if mdata:
            mm1, mm2, mm3, mm4 = st.columns(4)
            mm1.metric("Planned KM", f"{mdata['planned_km']:.0f} km")
            mm2.metric("Actual KM", f"{mdata['actual_km']:.0f} km")
            mm3.metric("Completion", f"{mdata['completion_pct']}%")
            mm4.metric(
                "Runs Done",
                f"{mdata['completed']} / {mdata['completed'] + mdata['missed'] + mdata['upcoming']}",
                help=f"Missed: {mdata['missed']}  ·  Upcoming: {mdata['upcoming']}",
            )

            chart_data = mdata.get("week_chart", [])
            if chart_data:
                import plotly.graph_objects as _go
                weeks_x = [c["week"] for c in chart_data]
                fig = _go.Figure(data=[
                    _go.Bar(
                        name="Planned km",
                        x=weeks_x,
                        y=[c["planned_km"] for c in chart_data],
                        marker_color="#63b3ed",
                    ),
                    _go.Bar(
                        name="Actual km",
                        x=weeks_x,
                        y=[c["actual_km"] for c in chart_data],
                        marker_color="#48bb78",
                    ),
                ])
                fig.update_layout(
                    barmode="group",
                    height=260,
                    margin=dict(l=0, r=0, t=8, b=0),
                    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ccc", size=12),
                    yaxis_title="km",
                )
                fig.update_xaxes(showgrid=False, tickangle=-30)
                fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("")
            m_summary_key = f"plan_month_summary_{sel_month_key}"
            saved_m_summary = mdata.get("monthly_summary")

            if not saved_m_summary and m_summary_key not in st.session_state:
                if st.button(
                    "Generate Monthly Summary",
                    key=f"btn_msummary_{sel_month_key}",
                    type="primary",
                ):
                    with st.spinner("Generating monthly summary…"):
                        mr = requests.post(
                            f"{BACKEND_URL}/plan/monthly-summary/{sel_month_key}",
                            timeout=60,
                        )
                    if mr.status_code == 200:
                        st.session_state[m_summary_key] = mr.json()["summary"]
                    elif mr.status_code == 404:
                        st.warning("No workouts found for this month.")
                    else:
                        st.error(f"Failed: {mr.text}")
                    st.rerun()
            else:
                m_summary_text = saved_m_summary or st.session_state.get(m_summary_key)
                if m_summary_text:
                    st.markdown(
                        f'<div class="insight-box">{m_summary_text}</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        "Regenerate Monthly Summary",
                        key=f"btn_mregen_{sel_month_key}",
                        help="Generate a fresh summary (overwrites saved)",
                    ):
                        if m_summary_key in st.session_state:
                            del st.session_state[m_summary_key]
                        with st.spinner("Regenerating…"):
                            mr = requests.post(
                                f"{BACKEND_URL}/plan/monthly-summary/{sel_month_key}?force=1",
                                timeout=60,
                            )
                        if mr.status_code == 200:
                            st.session_state[m_summary_key] = mr.json()["summary"]
                        st.rerun()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    render_sidebar()

    st.title("Strava AI Coach Dashboard")

    # Load dashboard data (cached in session_state)
    if "dashboard_data" not in st.session_state:
        with st.spinner("Loading dashboard data…"):
            data = get_dashboard_data()
        if data:
            st.session_state["dashboard_data"] = data
        else:
            data = {}
    else:
        data = st.session_state["dashboard_data"]

    # Top-level summary metrics + readiness banner
    if data:
        total = data.get("total_activities", 0)
        is_connected = data.get("is_connected", False)
        athlete_name = data.get("athlete_name")

        info_col1, info_col2, info_col3 = st.columns([2, 1, 1])
        with info_col1:
            if is_connected:
                st.markdown(
                    f'<div class="success-box">Connected as <b>{athlete_name or "Athlete"}</b> — '
                    f'{total} activities loaded.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="warning-box">Not connected to Strava. '
                    f'Visit <a href="{BACKEND_URL}/auth/strava/login">this link</a> to connect.</div>',
                    unsafe_allow_html=True,
                )

        # Readiness + intelligence banner (only when data is available)
        if total > 0:
            readiness = data.get("readiness", {})
            injury = data.get("injury_risk", {})
            race_pred = data.get("race_predictions", {})
            if readiness:
                r_score = readiness.get("readiness_score", 0)
                r_label = readiness.get("tier_label", "")
                i_label = injury.get("risk_label", "🟢 Low")
                hm_pred = race_pred.get("predictions", {}).get("half_marathon", {}).get("time_str", "–")
                traj = race_pred.get("fitness_trajectory", "stable")
                traj_icon = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(traj, "➡️")
                # Check for today's Garmin body battery
                garmin_today = _api_get("/garmin/daily?days=1") or []
                bb_today = garmin_today[0].get("body_battery_max") if garmin_today else None

                if bb_today is not None:
                    bc1, bc2, bc3, bc4, bc5 = st.columns(5)
                    bc5.metric("Body Battery", f"{bb_today}%", delta_color="off")
                else:
                    bc1, bc2, bc3, bc4 = st.columns(4)
                bc1.metric("Today's Readiness", f"{r_score:.0f}/100", delta=r_label, delta_color="off")
                bc2.metric("Injury Risk", i_label, delta_color="off")
                bc3.metric("HM Projection", hm_pred)
                bc4.metric("Fitness Trend", f"{traj_icon} {traj.capitalize()}", delta_color="off")
                st.markdown("")

        if total > 0:
            render_best_efforts()

    # Tabs
    tab_latest, tab_garmin, tab_trends, tab_be, tab_ai, tab_fatigue, tab_sleep, tab_long, tab_all, tab_intel, tab_workout, tab_plan, tab_profile = st.tabs([
        "Latest Run",
        "Garmin",
        "Trends",
        "Best Efforts",
        "AI Coach",
        "Fatigue & Recovery",
        "Sleep",
        "Long Runs",
        "All Activities",
        "Training Intelligence",
        "Workouts",
        "FM Plan",
        "Profile",
    ])

    with tab_latest:
        render_latest_run(data)

    with tab_garmin:
        render_garmin()

    with tab_trends:
        render_trends(data)

    with tab_be:
        render_best_efforts_tab()

    with tab_ai:
        render_ai_coach(data)

    with tab_fatigue:
        render_fatigue(data)

    with tab_sleep:
        render_sleep()

    with tab_long:
        render_long_runs(data)

    with tab_all:
        render_all_activities(data)

    with tab_intel:
        render_training_intelligence(data)

    with tab_workout:
        render_workouts(data)

    with tab_plan:
        render_fm_plan()

    with tab_profile:
        render_profile()


if __name__ == "__main__":
    main()
