"""
SQLAlchemy ORM model for daily health metrics (Apple Health via iOS Shortcut).
One row per calendar date — upserted each morning by the Shortcut.
"""
from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Float, Integer, String

from backend.db.database import Base


class DailyHealthMetric(Base):
    """Daily health snapshot synced from Apple Health (Garmin → Apple Health)."""

    __tablename__ = "daily_health_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, unique=True, nullable=False, index=True)

    # Sleep
    sleep_duration_hours = Column(Float, nullable=True)   # total sleep (hrs)
    sleep_deep_hours = Column(Float, nullable=True)       # deep/slow-wave sleep
    sleep_rem_hours = Column(Float, nullable=True)        # REM sleep
    sleep_core_hours = Column(Float, nullable=True)       # light/core sleep
    sleep_awake_hours = Column(Float, nullable=True)      # awake during night

    # Cardiovascular
    resting_hr = Column(Integer, nullable=True)           # bpm, morning reading
    hrv = Column(Float, nullable=True)                    # ms (reserved for future)

    notes = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
