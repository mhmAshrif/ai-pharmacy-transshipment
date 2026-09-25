"""Run lightweight DB sanity checks and print counts/sample rows.

Usage:
  python scripts/check_db.py

Run this from the project root with your venv activated.
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.database import SessionLocal
from sqlalchemy import text


def main():
    if SessionLocal is None:
        print("SessionLocal is not configured. Is DATABASE_URL set and reachable?")
        return

    db = SessionLocal()
    try:
        inv_count = db.execute(text("SELECT COUNT(*) FROM public.inventory")).scalar()
        man_count = db.execute(text("SELECT COUNT(*) FROM public.transfer_manifests")).scalar()
        print(f"inventory_count = {inv_count}")
        print(f"manifests_count = {man_count}")

        print("sample_inventory (up to 10 rows):")
        rows = db.execute(text("SELECT * FROM public.inventory LIMIT 10")).mappings().all()
        for row in rows:
            print(" - ", dict(row))

        print("sample_transfer_manifests (up to 10 rows):")
        rows2 = db.execute(text("SELECT * FROM public.transfer_manifests LIMIT 10")).mappings().all()
        for row in rows2:
            print(" - ", dict(row))
    except Exception as exc:
        print(f"Error running DB checks: {exc}")
    finally:
        try:
            db.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
