"""
SQLAlchemy ORM models for athlete profile and chat memory.
These tables are created automatically on startup via init_db().
"""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from backend.db.database import Base


class AthleteProfile(Base):
    """Persistent athlete profile — single row, upserted on save."""

    __tablename__ = "athlete_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    age = Column(Integer, nullable=True)
    weight_kg = Column(Float, nullable=True)
    resting_hr = Column(Integer, nullable=True)
    max_hr = Column(Integer, nullable=True)
    # beginner / intermediate / advanced
    running_experience = Column(String, nullable=True)
    # hm / 10k / marathon / base_building / weight_loss / fitness
    primary_goal = Column(String, nullable=True)
    target_race_distance = Column(String, nullable=True)
    target_race_date = Column(DateTime, nullable=True)
    preferred_weekly_km = Column(Float, nullable=True)
    # monday / tuesday / … / sunday
    preferred_long_run_day = Column(String, nullable=True)
    strength_training_days = Column(Integer, nullable=True, default=0)
    injury_history = Column(String, nullable=True)   # free-text description
    weekly_training_hours = Column(Float, nullable=True)
    notes = Column(String, nullable=True)
    # Placeholders for future Apple Health / wearable integration
    sleep_hours = Column(Float, nullable=True)       # nightly sleep (hrs)
    resting_hr_today = Column(Integer, nullable=True)  # morning resting HR
    hrv = Column(Float, nullable=True)               # heart-rate variability (ms)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChatMemory(Base):
    """
    Persistent chat history for the AI coach.

    The coach loads recent messages as context so it can reference
    prior conversations across sessions.
    """

    __tablename__ = "chat_memories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    role = Column(String, nullable=False)    # "user" or "assistant"
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
