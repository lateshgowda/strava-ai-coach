import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from backend.db.crud import (
    clear_chat_memory,
    delete_all_activities,
    get_activities,
    get_activities_since,
    get_activity_by_strava_id,
    get_health_data_for_date,
    get_profile,
    get_recent_chat_memory,
    get_recent_health_data,
    get_token,
    save_chat_memory,
    save_health_data,
    save_profile,
    save_token,
    update_splits,
)
from backend.db.database import get_db, init_db
from backend.models.activity import ActivitySchema, SyncResponse
from backend.services.activity_service import ActivityService
from backend.strava.auth import exchange_code, get_auth_url

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Strava AI Coach API",
    description="Backend API for the Strava AI Coach dashboard.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — allow Streamlit frontend
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://frontend:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Initialising database…")
    init_db()
    logger.info("Database ready.")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


@app.get("/auth/strava/login")
async def strava_login() -> RedirectResponse:
    """Redirect the user to Strava's OAuth consent page."""
    url = get_auth_url()
    return RedirectResponse(url=url)


@app.get("/auth/strava/callback")
async def strava_callback(
    code: str = Query(..., description="Authorization code from Strava"),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    """
    Handle the OAuth callback from Strava.
    Exchanges the authorization code for tokens and persists them.
    """
    try:
        token_data = exchange_code(code)
    except Exception as exc:
        logger.error("Failed to exchange Strava code: %s", exc)
        return RedirectResponse(url="http://localhost:8501?auth=error")

    athlete = token_data.get("athlete", {})
    save_data = {
        "access_token": token_data["access_token"],
        "refresh_token": token_data["refresh_token"],
        "expires_at": token_data["expires_at"],
        "athlete_id": athlete.get("id", 0),
        "athlete_name": f"{athlete.get('firstname', '')} {athlete.get('lastname', '')}".strip()
        or None,
    }
    save_token(db, save_data)
    logger.info("Strava account connected for athlete: %s", save_data["athlete_name"])

    return RedirectResponse(url="http://localhost:8501?auth=success")


@app.post("/auth/strava/disconnect")
async def strava_disconnect(db: Session = Depends(get_db)) -> Dict[str, str]:
    """Remove the stored OAuth token, effectively disconnecting Strava."""
    from sqlalchemy import text

    db.execute(text("DELETE FROM strava_tokens"))
    db.commit()
    return {"status": "disconnected"}


@app.get("/auth/status")
async def auth_status(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return whether a Strava account is connected."""
    token = get_token(db)
    return {
        "connected": token is not None,
        "athlete_name": token.athlete_name if token else None,
        "athlete_id": token.athlete_id if token else None,
    }


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


@app.post("/sync/activities", response_model=SyncResponse)
async def sync_activities_endpoint(
    full_sync: bool = Query(False, description="Delete existing data and re-sync all activities"),
    db: Session = Depends(get_db),
) -> SyncResponse:
    """Fetch activities from Strava and persist them to the database."""
    token = get_token(db)
    if token is None:
        raise HTTPException(status_code=401, detail="Strava account not connected.")

    try:
        service = ActivityService(db)
        result = service.sync(full_sync=full_sync)
        return SyncResponse(
            synced=result["synced"],
            skipped=result["skipped"],
            total=result["total"],
            message=f"Sync complete: {result['synced']} new, {result['skipped']} updated.",
        )
    except Exception as exc:
        logger.error("Sync failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@app.get("/dashboard")
async def get_dashboard(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return the full analytics payload for the frontend dashboard."""
    try:
        service = ActivityService(db)
        return service.get_dashboard_data()
    except Exception as exc:
        logger.error("Dashboard error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# AI
# ---------------------------------------------------------------------------


@app.get("/ai/insight")
async def get_ai_insight(db: Session = Depends(get_db)) -> Dict[str, str]:
    """Generate a coaching insight based on recent training data."""
    try:
        service = ActivityService(db)
        insight = service.get_ai_insight()
        return {"insight": insight}
    except Exception as exc:
        logger.error("AI insight error: %s", exc)
        return {"insight": f"Could not generate insight: {exc}"}


@app.get("/ai/weekly")
async def get_weekly_summary(db: Session = Depends(get_db)) -> Dict[str, str]:
    """Generate a weekly training summary with guidance."""
    try:
        service = ActivityService(db)
        summary = service.get_weekly_ai_summary()
        return {"summary": summary}
    except Exception as exc:
        logger.error("AI weekly summary error: %s", exc)
        return {"summary": f"Could not generate summary: {exc}"}


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


@app.post("/ai/chat")
async def ai_chat(request: ChatRequest, db: Session = Depends(get_db)) -> Dict[str, str]:
    """Conversational AI coach endpoint with persistent memory."""
    from backend.ai.coach import chat_with_coach

    try:
        service = ActivityService(db)
        dashboard = service.get_dashboard_data()

        # Build compact metrics dict
        metrics: Dict[str, Any] = {}
        latest = dashboard.get("latest_run") or {}
        if latest:
            metrics["latest_run"] = {
                k: v for k, v in latest.items()
                if k in ("distance_km", "pace_min_per_km", "duration_str",
                         "average_heartrate", "average_cadence", "elevation_gain", "name", "date")
            }
        metrics["fatigue_score"] = dashboard.get("fatigue_score", 0)
        metrics["recovery_score"] = dashboard.get("recovery_score", 0)
        metrics["acwr"] = dashboard.get("acwr", 0)
        metrics["total_activities"] = dashboard.get("total_activities", 0)
        metrics["readiness_score"] = dashboard.get("readiness", {}).get("readiness_score", 0)
        metrics["readiness_tier"] = dashboard.get("readiness", {}).get("tier_label", "")
        metrics["injury_risk"] = dashboard.get("injury_risk", {}).get("risk_label", "")
        if dashboard.get("estimated_hm_time"):
            metrics["estimated_hm_minutes"] = round(dashboard["estimated_hm_time"], 1)
        if dashboard.get("recent_trend"):
            metrics["recent_trend"] = dashboard["recent_trend"]
        if dashboard.get("personal_records"):
            metrics["personal_records"] = dashboard["personal_records"]
        if dashboard.get("race_predictions"):
            metrics["race_predictions"] = {
                dist: pred.get("time_str", "–")
                for dist, pred in dashboard["race_predictions"].get("predictions", {}).items()
            }
        if dashboard.get("training_state"):
            ts = dashboard["training_state"]
            metrics["training_state"] = {
                "consistency_score": ts.get("consistency_score"),
                "monotony_score": ts.get("monotony_score"),
                "aerobic_fitness_score": ts.get("aerobic_fitness_score"),
                "hr_efficiency_trend": ts.get("hr_efficiency_trend", {}).get("trend"),
                "pace_efficiency_trend": ts.get("pace_efficiency_trend", {}).get("trend"),
                "long_run_fade": ts.get("long_run_pace_fade", {}).get("verdict"),
                "overreaching": ts.get("overreaching"),
            }
        if dashboard.get("weekly_plan"):
            metrics["weekly_plan"] = {
                k: v for k, v in dashboard["weekly_plan"].items()
                if k in ("recommended_weekly_km", "phase", "quality_days", "long_run_km", "intensity_budget")
            }
        metrics["total_runs"] = dashboard.get("total_runs", 0)

        # Include today's Apple Health data in coach context
        if dashboard.get("today_health"):
            th = dashboard["today_health"]
            metrics["todays_health"] = {k: v for k, v in th.items() if v is not None}

        # Include individual activities from the last 10 days so the AI can
        # answer questions like "what did I do last week?"
        try:
            from datetime import datetime as _dt, timedelta as _td
            since = _dt.utcnow() - _td(days=10)
            recent_acts = get_activities_since(db, since)
            if recent_acts:
                metrics["recent_activities"] = [
                    {
                        "date": a.start_date.strftime("%Y-%m-%d"),
                        "type": a.activity_type,
                        "name": a.name,
                        "distance_km": round(a.distance / 1000, 2) if a.distance else 0,
                        "duration_min": round(a.moving_time / 60, 1) if a.moving_time else 0,
                        "avg_hr": round(a.average_heartrate) if a.average_heartrate else None,
                        "elevation_m": round(a.elevation_gain) if a.elevation_gain else 0,
                        "pace_min_per_km": (
                            round((a.moving_time / 60) / (a.distance / 1000), 2)
                            if a.distance and a.moving_time and a.distance > 0
                            else None
                        ),
                    }
                    for a in recent_acts
                ]
        except Exception:
            pass  # don't crash chat if recent activity fetch fails

        # Load athlete profile
        profile_data: Optional[Dict[str, Any]] = None
        try:
            profile_obj = get_profile(db)
            if profile_obj:
                profile_data = {
                    "age": profile_obj.age,
                    "running_experience": profile_obj.running_experience,
                    "primary_goal": profile_obj.primary_goal,
                    "target_race_distance": profile_obj.target_race_distance,
                    "injury_history": profile_obj.injury_history,
                    "preferred_weekly_km": profile_obj.preferred_weekly_km,
                }
        except Exception:
            pass

        # Merge: explicit history (current session) + DB memory (past sessions)
        if request.history:
            # Frontend provided session history — use as-is, DB memory is implicitly in context
            history = [{"role": m.role, "content": m.content} for m in request.history]
        else:
            # Fresh conversation — load recent memory from DB as prior context
            memory_rows = get_recent_chat_memory(db, limit=12)
            history = [{"role": r.role, "content": r.content} for r in memory_rows]

        reply = chat_with_coach(
            message=request.message,
            history=history,
            metrics=metrics,
            profile=profile_data,
        )

        # Persist this exchange to memory
        save_chat_memory(db, "user", request.message)
        save_chat_memory(db, "assistant", reply)

        return {"response": reply}
    except Exception as exc:
        logger.error("AI chat error: %s", exc)
        return {"response": f"Coach is unavailable right now: {exc}"}


# ---------------------------------------------------------------------------
# Activities list
# ---------------------------------------------------------------------------


@app.get("/activities/{strava_id}/splits")
async def get_splits(strava_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return km splits for an activity. Fetches from Strava if not cached."""
    import json as _json

    activity = get_activity_by_strava_id(db, strava_id)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")

    # Return cached splits if available
    if activity.splits_metric:
        try:
            return {"strava_id": strava_id, "splits": _json.loads(activity.splits_metric), "cached": True}
        except Exception:
            pass

    # Fetch from Strava on-demand
    from backend.strava.auth import get_valid_token
    from backend.strava.client import StravaClient

    access_token = get_valid_token(db)
    if not access_token:
        raise HTTPException(status_code=401, detail="Not connected to Strava")

    try:
        client = StravaClient(access_token)
        detail = client.get_activity_detail(strava_id)
        splits = detail.get("splits_metric") or []
        if splits:
            update_splits(db, strava_id, _json.dumps(splits))
        return {"strava_id": strava_id, "splits": splits, "cached": False}
    except Exception as exc:
        logger.error("Failed to fetch splits for %s: %s", strava_id, exc)
        raise HTTPException(status_code=502, detail=f"Strava error: {exc}")


@app.get("/activities")
async def list_activities(
    limit: int = Query(100, ge=1, le=500),
    activity_type: Optional[str] = Query(None, description="Filter by type e.g. Run, Ride, WeightTraining, Walk"),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Return a paginated list of stored activities. Optionally filter by activity_type."""
    activities = get_activities(db, limit=limit, activity_type=activity_type)
    return [
        ActivitySchema.model_validate(a).model_dump(mode="json")
        for a in activities
    ]


# ---------------------------------------------------------------------------
# Athlete Profile
# ---------------------------------------------------------------------------


class ProfileRequest(BaseModel):
    age: Optional[int] = None
    weight_kg: Optional[float] = None
    resting_hr: Optional[int] = None
    max_hr: Optional[int] = None
    running_experience: Optional[str] = None   # beginner/intermediate/advanced
    primary_goal: Optional[str] = None         # hm/10k/marathon/base_building/weight_loss
    target_race_distance: Optional[str] = None
    target_race_date: Optional[str] = None     # ISO date string YYYY-MM-DD
    preferred_weekly_km: Optional[float] = None
    preferred_long_run_day: Optional[str] = None
    strength_training_days: Optional[int] = None
    injury_history: Optional[str] = None
    weekly_training_hours: Optional[float] = None
    notes: Optional[str] = None
    # Sleep/HRV placeholders (for future wearable integration)
    sleep_hours: Optional[float] = None
    resting_hr_today: Optional[int] = None
    hrv: Optional[float] = None


@app.get("/profile")
async def get_profile_endpoint(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return the stored athlete profile."""
    profile = get_profile(db)
    if not profile:
        return {}
    result: Dict[str, Any] = {}
    for col in profile.__table__.columns:
        val = getattr(profile, col.name)
        if hasattr(val, "isoformat"):
            val = val.isoformat()
        result[col.name] = val
    return result


@app.post("/profile")
async def save_profile_endpoint(
    request: ProfileRequest,
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Create or update the athlete profile."""
    from datetime import datetime as _dt
    data = request.model_dump(exclude_none=True)
    if "target_race_date" in data and data["target_race_date"]:
        try:
            data["target_race_date"] = _dt.fromisoformat(data["target_race_date"])
        except ValueError:
            data.pop("target_race_date", None)
    save_profile(db, data)
    return {"status": "saved"}


# ---------------------------------------------------------------------------
# Workout Generator
# ---------------------------------------------------------------------------


class WorkoutRequest(BaseModel):
    workout_type: str = "easy"    # easy/intervals/threshold/long_run/recovery
    available_minutes: int = 60


@app.post("/workout/generate")
async def generate_workout_endpoint(
    request: WorkoutRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Generate a structured workout based on current athlete state."""
    from backend.analytics.workout_generator import generate_workout

    activities = get_activities(db, limit=10, activity_type="Run")
    from backend.analytics.engine import AnalyticsEngine
    engine = AnalyticsEngine(activities)
    latest = engine.latest_run_stats()
    recent_pace = latest.get("pace_min_per_km")
    fatigue = engine.fatigue_score()

    profile = get_profile(db)
    goal = profile.primary_goal if profile else None
    max_hr = profile.max_hr if profile else None

    # Get readiness from full dashboard
    try:
        service = ActivityService(db)
        dash = service.get_dashboard_data()
        readiness_score = dash.get("readiness", {}).get("readiness_score", 50.0)
    except Exception:
        readiness_score = 50.0

    return generate_workout(
        workout_type=request.workout_type,
        recent_avg_pace=recent_pace,
        fatigue_score=fatigue,
        readiness_score=readiness_score,
        goal=goal,
        available_minutes=request.available_minutes,
        max_hr=max_hr,
    )


# ---------------------------------------------------------------------------
# Apple Health Data (synced via iOS Shortcut)
# ---------------------------------------------------------------------------


class HealthDataRequest(BaseModel):
    date: str                                    # YYYY-MM-DD (today's date from Shortcut)
    sleep_duration_hours: Optional[float] = None
    sleep_deep_hours: Optional[float] = None
    sleep_rem_hours: Optional[float] = None
    sleep_core_hours: Optional[float] = None
    sleep_awake_hours: Optional[float] = None
    resting_hr: Optional[int] = None
    hrv: Optional[float] = None
    notes: Optional[str] = None


@app.post("/health-data")
async def save_health_data_endpoint(
    request: HealthDataRequest,
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """
    Receive daily health metrics from the iOS Shortcut.
    Upserts one row per date — safe to call multiple times on the same day.
    """
    data = request.model_dump(exclude_none=True)
    save_health_data(db, data)
    logger.info("Health data saved for %s", request.date)
    return {"status": "saved", "date": request.date}


@app.get("/health-data/today")
async def get_today_health_endpoint(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return today's health metrics, or empty dict if not yet synced."""
    import datetime as _dt
    row = get_health_data_for_date(db, _dt.date.today())
    if not row:
        return {}
    return {
        "date": row.date.isoformat(),
        "sleep_duration_hours": row.sleep_duration_hours,
        "sleep_deep_hours": row.sleep_deep_hours,
        "sleep_rem_hours": row.sleep_rem_hours,
        "sleep_core_hours": row.sleep_core_hours,
        "sleep_awake_hours": row.sleep_awake_hours,
        "resting_hr": row.resting_hr,
        "hrv": row.hrv,
    }


@app.get("/health-data/history")
async def get_health_history_endpoint(
    days: int = Query(14, ge=1, le=90),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Return health metrics for the last N days."""
    rows = get_recent_health_data(db, days=days)
    return [
        {
            "date": r.date.isoformat(),
            "sleep_duration_hours": r.sleep_duration_hours,
            "sleep_deep_hours": r.sleep_deep_hours,
            "sleep_rem_hours": r.sleep_rem_hours,
            "sleep_core_hours": r.sleep_core_hours,
            "sleep_awake_hours": r.sleep_awake_hours,
            "resting_hr": r.resting_hr,
            "hrv": r.hrv,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Chat Memory Management
# ---------------------------------------------------------------------------


@app.delete("/ai/chat/memory")
async def clear_chat_memory_endpoint(db: Session = Depends(get_db)) -> Dict[str, str]:
    """Clear all persistent chat history."""
    clear_chat_memory(db)
    return {"status": "cleared"}


@app.get("/ai/chat/memory")
async def get_chat_memory_endpoint(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Return recent chat history."""
    rows = get_recent_chat_memory(db, limit=limit)
    return [{"role": r.role, "content": r.content, "created_at": str(r.created_at)} for r in rows]
