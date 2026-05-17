from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
)

from backend.db.database import Base


# ---------------------------------------------------------------------------
# SQLAlchemy ORM models
# ---------------------------------------------------------------------------


class StravaToken(Base):
    """Stores the single OAuth token for the authenticated athlete."""

    __tablename__ = "strava_tokens"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    access_token = Column(String, nullable=False)
    refresh_token = Column(String, nullable=False)
    expires_at = Column(Integer, nullable=False)  # Unix timestamp
    athlete_id = Column(Integer, nullable=False)
    athlete_name = Column(String, nullable=True)


class Activity(Base):
    """Persisted Strava running activity."""

    __tablename__ = "activities"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    strava_id = Column(Integer, unique=True, nullable=False)
    name = Column(String, nullable=False, default="")
    distance = Column(Float, nullable=False, default=0.0)        # metres
    moving_time = Column(Integer, nullable=False, default=0)     # seconds
    elapsed_time = Column(Integer, nullable=False, default=0)    # seconds
    elevation_gain = Column(Float, nullable=False, default=0.0)
    start_date = Column(DateTime, nullable=False)
    average_heartrate = Column(Float, nullable=True)
    max_heartrate = Column(Float, nullable=True)
    average_cadence = Column(Float, nullable=True)
    average_speed = Column(Float, nullable=False, default=0.0)   # m/s
    max_speed = Column(Float, nullable=True)
    suffer_score = Column(Integer, nullable=True)
    splits_metric = Column(String, nullable=True)                # JSON string
    activity_type = Column(String, nullable=False, default="Run")
    pr_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_activities_strava_id", "strava_id"),
        Index("ix_activities_start_date", "start_date"),
        Index("ix_activities_activity_type", "activity_type"),
    )


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ActivitySchema(BaseModel):
    """Pydantic read-schema that mirrors the Activity ORM model."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    strava_id: int
    name: str
    distance: float
    moving_time: int
    elapsed_time: int
    elevation_gain: float
    start_date: datetime
    average_heartrate: Optional[float] = None
    max_heartrate: Optional[float] = None
    average_cadence: Optional[float] = None
    average_speed: float
    max_speed: Optional[float] = None
    suffer_score: Optional[int] = None
    splits_metric: Optional[str] = None
    activity_type: str
    pr_count: Optional[int] = None
    created_at: datetime


class SyncResponse(BaseModel):
    """Response model for the /sync/activities endpoint."""

    synced: int
    skipped: int
    total: int
    message: str = ""
