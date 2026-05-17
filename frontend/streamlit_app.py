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

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Backend: {BACKEND_URL}")


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


def render_weekly(data: Dict[str, Any]) -> None:
    weekly = data.get("weekly", [])

    if not weekly:
        st.info("No weekly data yet — sync activities to see your mileage.")
        return

    st.plotly_chart(make_weekly_bar(weekly), use_container_width=True)

    st.subheader("Recent Weeks")
    df = pd.DataFrame(weekly)
    if not df.empty:
        display_cols = {
            "week": "Week",
            "distance_km": "Distance (km)",
            "run_count": "Runs",
            "avg_pace": "Avg Pace (min/km)",
            "avg_hr": "Avg HR",
        }
        df_display = df[[c for c in display_cols if c in df.columns]].copy()
        df_display.rename(columns=display_cols, inplace=True)
        if "Distance (km)" in df_display.columns:
            df_display["Distance (km)"] = df_display["Distance (km)"].round(1)
        if "Avg Pace (min/km)" in df_display.columns:
            df_display["Avg Pace (min/km)"] = df_display["Avg Pace (min/km)"].apply(
                lambda x: format_pace(x) if pd.notna(x) else "–"
            )
        if "Avg HR" in df_display.columns:
            df_display["Avg HR"] = df_display["Avg HR"].apply(
                lambda x: f"{x:.0f}" if pd.notna(x) else "–"
            )
        st.dataframe(df_display.sort_values("Week", ascending=False), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Monthly Trends
# ---------------------------------------------------------------------------


def render_monthly(data: Dict[str, Any]) -> None:
    monthly = data.get("monthly", [])
    pace_trend = data.get("pace_trend", [])
    hr_trend = data.get("hr_trend", [])

    if not monthly and not pace_trend and not hr_trend:
        st.info("No trend data yet — sync activities to see your trends.")
        return

    st.plotly_chart(make_monthly_chart(monthly), use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(make_pace_trend(pace_trend), use_container_width=True)
    with col_right:
        st.plotly_chart(make_hr_trend(hr_trend), use_container_width=True)

    cadence_trend = data.get("cadence_trend", [])
    if cadence_trend:
        st.plotly_chart(make_cadence_trend(cadence_trend), use_container_width=True)

    # Trend summary box
    trend = data.get("recent_trend", {})
    if trend:
        st.markdown("---")
        st.subheader("4-Week Trend Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Weekly km (last 4w)",
            f"{trend.get('avg_weekly_km_last4w', 0):.1f} km",
            delta=f"{trend.get('trend_pct', 0):+.1f}%",
        )
        c2.metric(
            "Weekly km (prev 4w)",
            f"{trend.get('avg_weekly_km_prev4w', 0):.1f} km",
        )
        avg_pace = trend.get("avg_pace_last4w")
        c3.metric("Avg Pace", format_pace(avg_pace))
        avg_hr = trend.get("avg_hr_last4w")
        c4.metric("Avg HR", f"{avg_hr:.0f} bpm" if avg_hr else "–")


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

    # --- Three gauges ---
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = make_gauge(fatigue, "Fatigue Score", max_val=100,
                         green_threshold=35, red_threshold=70)
        st.plotly_chart(fig, use_container_width=True)
        if fatigue > 70:
            st.markdown('<div class="warning-box">High fatigue — prioritise rest and easy runs.</div>',
                        unsafe_allow_html=True)
        elif fatigue < 35:
            st.markdown('<div class="success-box">Low fatigue — well rested.</div>',
                        unsafe_allow_html=True)

    with col2:
        fig = make_recovery_gauge(recovery)
        st.plotly_chart(fig, use_container_width=True)
        if recovery >= 65:
            st.markdown('<div class="success-box">Good recovery — ready for quality training.</div>',
                        unsafe_allow_html=True)
        elif recovery < 40:
            st.markdown('<div class="warning-box">Low recovery — consider an easy day or rest.</div>',
                        unsafe_allow_html=True)

    with col3:
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

    # --- Sleep & Recovery (Apple Health via iOS Shortcut) ---
    st.markdown("---")
    st.subheader("Last Night's Sleep & Recovery")
    today_health = data.get("today_health")
    if today_health and any(today_health.get(k) for k in (
        "sleep_duration_hours", "sleep_deep_hours", "sleep_rem_hours", "resting_hr"
    )):
        h = today_health
        sleep_total = h.get("sleep_duration_hours")
        sleep_deep = h.get("sleep_deep_hours")
        sleep_rem = h.get("sleep_rem_hours")
        sleep_core = h.get("sleep_core_hours")
        sleep_awake = h.get("sleep_awake_hours", 0) or 0
        rhr = h.get("resting_hr")

        # Sleep metrics row
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
            "Core/Light": sleep_core, "Awake": sleep_awake,
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

        # Readiness impact note
        factors = data.get("readiness", {}).get("factors", {})
        sleep_pen = factors.get("sleep_penalty", 0)
        deep_pen = factors.get("deep_sleep_penalty", 0)
        rhr_pen = factors.get("resting_hr_penalty", 0)
        total_health_penalty = sleep_pen + deep_pen + rhr_pen
        if total_health_penalty > 0:
            st.markdown(
                f'<div style="background:rgba(245,101,101,0.08);border-left:4px solid #F56565;'
                f'border-radius:6px;padding:10px 14px;margin:6px 0;font-size:0.9em;">'
                f'Sleep & recovery deducted <b>{total_health_penalty:.0f} points</b> from today\'s readiness score '
                f'(sleep: -{sleep_pen:.0f}, deep sleep: -{deep_pen:.0f}, resting HR: -{rhr_pen:.0f})'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="background:rgba(99,179,237,0.08);border-left:4px solid #63B3ED;'
            'border-radius:6px;padding:12px 16px;color:#A0AEC0;">'
            'No sleep data for today yet. Set up the iOS Shortcut to automatically sync '
            'your Garmin sleep data each morning and improve readiness accuracy.'
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

    display_df = df[["date", "name", "activity_type", "dist_str", "duration_str", "average_heartrate"]].copy()
    display_df.columns = ["Date", "Name", "Type", "Distance", "Duration", "Avg HR"]
    display_df["Avg HR"] = display_df["Avg HR"].apply(lambda x: f"{x:.0f}" if pd.notna(x) and x else "–")
    display_df["Type"] = display_df["Type"].apply(lambda t: f"{_TYPE_ICONS.get(t, '🏅')} {t}")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400,
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
        result = _api_post("/ai/chat/memory")
        # Use DELETE via requests directly since _api_post uses POST
        try:
            import requests as _requests
            resp = _requests.delete(f"{BACKEND_URL}/ai/chat/memory", timeout=10)
            if resp.ok:
                st.success("Chat memory cleared.")
            else:
                st.error("Failed to clear memory.")
        except Exception as exc:
            st.error(f"Error: {exc}")


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
                bc1, bc2, bc3, bc4 = st.columns(4)
                bc1.metric("Today's Readiness", f"{r_score:.0f}/100", delta=r_label, delta_color="off")
                bc2.metric("Injury Risk", i_label, delta_color="off")
                bc3.metric("HM Projection", hm_pred)
                bc4.metric("Fitness Trend", f"{traj_icon} {traj.capitalize()}", delta_color="off")
                st.markdown("")

    # Tabs
    tab_latest, tab_weekly, tab_monthly, tab_ai, tab_fatigue, tab_long, tab_all, tab_intel, tab_workout, tab_profile = st.tabs([
        "Latest Run",
        "Weekly",
        "Monthly Trends",
        "AI Coach",
        "Fatigue & Recovery",
        "Long Runs",
        "All Activities",
        "Training Intelligence",
        "Workouts",
        "Profile",
    ])

    with tab_latest:
        render_latest_run(data)

    with tab_weekly:
        render_weekly(data)

    with tab_monthly:
        render_monthly(data)

    with tab_ai:
        render_ai_coach(data)

    with tab_fatigue:
        render_fatigue(data)

    with tab_long:
        render_long_runs(data)

    with tab_all:
        render_all_activities(data)

    with tab_intel:
        render_training_intelligence(data)

    with tab_workout:
        render_workouts(data)

    with tab_profile:
        render_profile()


if __name__ == "__main__":
    main()
