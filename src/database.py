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


def init_db() -> bool:
    """Create database schema for all declarative models if the engine is ready.

    Returns True when the schema creation was attempted (engine available), False otherwise.
    """
    _initialize_database()
    if engine is None:
        return False

    try:
        # local import to avoid circular import at module load time
        from .models import Base

        Base.metadata.create_all(bind=engine)
        return True
    except Exception:
        return False


def seed_initial_inventory(db: Session) -> int:
    """Populate the inventory table from the raw CSV if it is empty.

    Returns the number of rows inserted.
    """
    try:
        from .models import Inventory
        import pandas as pd

        existing = db.query(Inventory).count()
        if existing and existing > 0:
            return 0

        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "srilanka_5node_pharmacy_sales.csv")
        if not os.path.exists(csv_path):
            return 0

        df = pd.read_csv(csv_path)
        # Normalize column names that exist in the dataset
        df = df.rename(columns={
            "medicine": "medicine_name",
            "stock_level": "stock_quantity",
        })

        # Group by district + medicine_name and take the last observed values
        grouped = (
            df.groupby(["district", "medicine_name"], as_index=False)
            .agg({
                "stock_quantity": "last",
                "unit_price": "last",
                "expiry_days_remaining": "last",
                "category": "first",
            })
        )

        objects = []
        for _, row in grouped.iterrows():
            obj = Inventory(
                district=str(row.get("district") or "Unknown"),
                medicine_name=str(row.get("medicine_name") or "Unknown"),
                category=str(row.get("category") or "Unknown"),
                stock_quantity=int(row.get("stock_quantity") or 0),
                expiry_days_remaining=int(row.get("expiry_days_remaining") or 0),
                unit_price=float(row.get("unit_price") or 0.0),
            )
            objects.append(obj)

        if objects:
            db.bulk_save_objects(objects)
            db.commit()
            return len(objects)
        return 0
    except Exception:
        return 0


def ensure_reference_lookup_tables(db: Session) -> None:
    """Populate missing district/medicine reference records and backfill inventory FK ids."""
    try:
        from .models import District, Medicine

        inventory_rows = db.execute(text(
            "SELECT DISTINCT district, medicine FROM public.inventory WHERE district IS NOT NULL AND medicine IS NOT NULL"
        )).mappings().all()

        for row in inventory_rows:
            district_name = str(row.get("district") or "").strip()
            medicine_name = str(row.get("medicine") or "").strip()

            if district_name:
                district = db.execute(
                    text("SELECT id FROM public.districts WHERE LOWER(name) = LOWER(:name) LIMIT 1"),
                    {"name": district_name},
                ).scalar()
                if district is None:
                    district = db.execute(text("INSERT INTO public.districts (name) VALUES (:name) RETURNING id"), {"name": district_name}).scalar()
                db.execute(
                    text("UPDATE public.inventory SET district_id = :district_id WHERE LOWER(TRIM(district)) = LOWER(TRIM(:district_name)) AND (district_id IS NULL OR district_id = 0)"),
                    {"district_id": district, "district_name": district_name},
                )

            if medicine_name:
                medicine = db.execute(
                    text("SELECT id FROM public.medicines WHERE LOWER(name) = LOWER(:name) LIMIT 1"),
                    {"name": medicine_name},
                ).scalar()
                if medicine is None:
                    medicine = db.execute(
                        text("""
                            INSERT INTO public.medicines (name, category, unit_price)
                            SELECT :name, COALESCE(category, 'Unknown'), COALESCE(unit_price, 0)
                            FROM public.inventory
                            WHERE LOWER(TRIM(medicine)) = LOWER(TRIM(:name))
                            ORDER BY id
                            LIMIT 1
                            RETURNING id
                        """),
                        {"name": medicine_name},
                    ).scalar()
                db.execute(
                    text("UPDATE public.inventory SET medicine_id = :medicine_id WHERE LOWER(TRIM(medicine)) = LOWER(TRIM(:medicine_name)) AND (medicine_id IS NULL OR medicine_id = 0)"),
                    {"medicine_id": medicine, "medicine_name": medicine_name},
                )

        db.commit()
    except Exception:
        db.rollback()


def backfill_inventory_foreign_keys(db: Session) -> int:
    """Backfill missing medicine_id and district_id in inventory by matching with LOWER(TRIM()) logic.
    
    Returns the number of rows updated.
    """
    try:
        # Backfill medicine_id for rows where medicine_id IS NULL
        medicine_updated = db.execute(text("""
            UPDATE public.inventory i
            SET medicine_id = m.id
            FROM public.medicines m
            WHERE i.medicine_id IS NULL
              AND LOWER(TRIM(i.medicine)) = LOWER(TRIM(m.name))
        """)).rowcount
        
        # Backfill district_id for rows where district_id IS NULL
        district_updated = db.execute(text("""
            UPDATE public.inventory i
            SET district_id = d.id
            FROM public.districts d
            WHERE i.district_id IS NULL
              AND LOWER(TRIM(i.district)) = LOWER(TRIM(d.name))
        """)).rowcount
        
        db.commit()
        total_updated = medicine_updated + district_updated
        if total_updated > 0:
            print(f"Backfilled {medicine_updated} medicine_id and {district_updated} district_id rows.")
        return total_updated
    except Exception as e:
        db.rollback()
        print(f"Warning: backfill_inventory_foreign_keys failed: {e}")
        return 0


def cleanup_duplicate_transfer_manifests(db: Session) -> int:
    """Remove duplicate manifest rows with the same source/destination/medicine/quantity combination."""
    try:
        deleted = db.execute(text(
            """
            WITH ranked AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY source_district_id, dest_district_id, medicine_id, quantity_to_move
                           ORDER BY id
                       ) AS rn
                FROM public.transfer_manifests
                WHERE source_district_id IS NOT NULL
                  AND dest_district_id IS NOT NULL
                  AND medicine_id IS NOT NULL
            )
            DELETE FROM public.transfer_manifests
            WHERE id IN (SELECT id FROM ranked WHERE rn > 1)
            RETURNING id
            """
        )).fetchall()
        db.commit()
        return len(deleted)
    except Exception:
        db.rollback()
        return 0


# Ensure ORM models are imported and registered with SQLAlchemy metadata when this module is loaded.
try:
    from . import models  # noqa: F401
except Exception:
    # Import errors should not break CSV fallback behavior during development.
    pass
