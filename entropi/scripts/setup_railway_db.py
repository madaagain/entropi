"""
Run ONCE after first Railway deploy to set up the database.
Reads DATABASE_URL from environment variables.

Usage:
  DATABASE_URL=postgresql://... python scripts/setup_railway_db.py
"""

import sys
import os
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from dotenv import load_dotenv
load_dotenv(root / ".env")

from entropi.db.database import init_db
from entropi.db.api_keys import create_api_key


def main():
    print("\n  Setting up Entropi database...\n")

    # create tables
    try:
        init_db()
        print("  Tables created (api_keys, usage_logs)")
    except Exception as e:
        print(f"  Failed to create tables: {e}")
        sys.exit(1)

    # generate admin key
    try:
        api_key = create_api_key(name="admin")
        print()
        print("  Save this key now — it won't be shown again:")
        print(f"  {api_key}")
        print()
    except Exception as e:
        print(f"  Failed to generate API key: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
