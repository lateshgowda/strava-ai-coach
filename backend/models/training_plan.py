from datetime import date, datetime

from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, String, Text

from backend.db.database import Base


class TrainingPlan(Base):
    __tablename__ = "training_plan"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_date = Column(Date, nullable=False, unique=True, index=True)
    distance_km = Column(Float, nullable=False)
    session_type = Column(String(128), nullable=False)
    details = Column(Text, nullable=True)
    weekly_volume_km = Column(Float, nullable=True)
    iso_week = Column(String(10), nullable=False, index=True)  # e.g. "2026-W36"


class AIRunReview(Base):
    __tablename__ = "ai_run_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strava_id = Column(Integer, nullable=False, unique=True, index=True)
    plan_workout_id = Column(Integer, ForeignKey("training_plan.id"), nullable=True)
    review_text = Column(Text, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)


class AIWeeklySummary(Base):
    __tablename__ = "ai_weekly_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    iso_week = Column(String(10), nullable=False, unique=True, index=True)
    summary_text = Column(Text, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)


class AIMonthSummary(Base):
    __tablename__ = "ai_month_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    month_key = Column(String(7), nullable=False, unique=True, index=True)  # "2026-09"
    summary_text = Column(Text, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)
