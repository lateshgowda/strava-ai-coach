import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:////data/strava.db")

# SQLite needs check_same_thread=False; other databases ignore this kwarg
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a database session and ensures it is closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables defined on Base metadata."""
    # Import models so their table definitions are registered on Base.metadata
    from backend.models import activity        # noqa: F401
    from backend.models import profile         # noqa: F401
    from backend.models import health          # noqa: F401
    from backend.models import training_plan   # noqa: F401

    Base.metadata.create_all(bind=engine)
