import os
from typing import Iterator, Optional

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

load_dotenv(find_dotenv())

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
engine = None
SessionLocal = None
DATABASE_AVAILABLE = False


def _initialize_database() -> None:
    """Initialize the engine and sessionmaker once the DATABASE_URL is available."""
    global engine, SessionLocal, DATABASE_AVAILABLE

    if not DATABASE_URL:
        DATABASE_AVAILABLE = False
        return

    if engine is not None and SessionLocal is not None:
        DATABASE_AVAILABLE = True
        return

    try:
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5},
        )
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine,
            expire_on_commit=False,
        )

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        DATABASE_AVAILABLE = True
    except (OperationalError, SQLAlchemyError, Exception):
        engine = None
        SessionLocal = None
        DATABASE_AVAILABLE = False


_initialize_database()


def test_db_connection() -> bool:
    """Verify database connectivity and print startup status to the terminal."""
    if not DATABASE_URL:
        print("WARNING: Database connection failed. Operating in CSV fallback mode. DATABASE_URL is missing.")
        return False

    _initialize_database()

    if engine is None:
        print("WARNING: Database connection failed. Operating in CSV fallback mode. Engine is not initialized.")
        return False

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("Database connection established successfully.")
        return True
    except (OperationalError, SQLAlchemyError, Exception) as exc:
        print(f"WARNING: Database connection failed. Operating in CSV fallback mode. {exc}")
        return False


def get_db() -> Iterator[Optional[Session]]:
    """Yields a database session if the database is available, otherwise yields None."""
    _initialize_database()
    if SessionLocal is None:
        yield None
        return

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
