"""Database session management."""

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from tipster.db.models import Base

_engine = None
_SessionLocal = None


def init_db(db_path: str = "tipster.db") -> None:
    """Initialise the database engine and create all tables."""
    global _engine, _SessionLocal
    db_url = f"sqlite:///{db_path}"
    _engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
        echo=False,
    )
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(_engine)


def get_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session, closing it afterwards."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db() -> Session:
    """Return a new session (caller is responsible for closing)."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    return _SessionLocal()
