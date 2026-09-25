#!/usr/bin/env python3
"""Initialize the database schema and seed initial inventory from CSV.

Usage:
  python scripts/init_db.py

This script reads `DATABASE_URL` from the project's .env and will create tables
via SQLAlchemy `Base.metadata.create_all`. It will then populate `inventory`
from `data/raw/srilanka_5node_pharmacy_sales.csv` if the table is empty.
"""
import os
import sys

# Ensure project root is on sys.path so relative imports work when called from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from src.database import init_db, seed_initial_inventory, SessionLocal


def main():
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        print("ERROR: DATABASE_URL is not set in .env. Aborting.")
        return

    print("Initializing database schema...")
    ok = init_db()
    if not ok:
        print("WARNING: init_db() returned False. Ensure DATABASE_URL is reachable.")
        return
    print("Schema initialization complete.")

    if SessionLocal is None:
        print("SessionLocal not configured. Skipping seeding.")
        return

    db = None
    try:
        db = SessionLocal()
        inserted = seed_initial_inventory(db)
        if inserted:
            print(f"Seeded inventory with {inserted} rows from CSV.")
        else:
            print("No seed applied; inventory table already populated or CSV missing.")
    except Exception as exc:
        print(f"ERROR: seeding failed: {exc}")
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
