from datetime import date, datetime

from sqlalchemy import Column, Date, DateTime, Float, Integer, String, Text

from backend.db.database import Base


class GarminToken(Base):
    __tablename__ = "garmin_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(256), nullable=False)
    tokenstore = Column(Text, nullable=False)  # garth serialized JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class GarminDailyMetrics(Base):
    __tablename__ = "garmin_daily_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    body_battery_max = Column(Integer, nullable=True)
    body_battery_min = Column(Integer, nullable=True)
    stress_avg = Column(Integer, nullable=True)
    resting_hr = Column(Integer, nullable=True)
    sleep_duration_sec = Column(Integer, nullable=True)
    sleep_deep_sec = Column(Integer, nullable=True)
    sleep_rem_sec = Column(Integer, nullable=True)
    sleep_light_sec = Column(Integer, nullable=True)
    sleep_awake_sec = Column(Integer, nullable=True)
    sleep_score = Column(Integer, nullable=True)
    recovery_time_hours = Column(Integer, nullable=True)
    vo2max = Column(Float, nullable=True)
    synced_at = Column(DateTime, default=datetime.utcnow)
