from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config.settings import settings
from app.utils.logging_config import logger

# Ensure SQLite parent data directory exists
database_url = settings.DATABASE_URL
if database_url.startswith("sqlite:///"):
    db_path_str = database_url.replace("sqlite:///", "")
    from pathlib import Path

    db_file = Path(db_path_str)
    db_file.parent.mkdir(parents=True, exist_ok=True)

# Create SQLAlchemy engine (connect_args check_same_thread=False for SQLite)
engine = create_engine(
    database_url,
    connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


def init_db():
    """Creates all database tables if they do not exist."""
    logger.info(f"Initializing database at: {settings.DATABASE_URL}")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables successfully created.")


def get_db() -> Generator:
    """Dependency injection helper for FastAPI & service layer to obtain DB sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
