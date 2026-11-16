#!/usr/bin/env python
"""
Database initialization script for Federated Learning IDS Platform.
Run this script to create all database tables.
"""

import asyncio
import sys
import os
from pathlib import Path

# Load .env file (without python-dotenv dependency)
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard.db.session import db_manager, init_database
from src.dashboard.db.models import Base


async def main():
    """Initialize database tables."""
    print("Initializing database...")
    print(f"Database URL: {db_manager.database_url}")

    try:
        await init_database()
        print("✓ Database tables created successfully")

        # Verify tables were created
        from sqlalchemy import inspect
        async with db_manager.engine.begin() as conn:
            def get_tables(sync_conn):
                inspector = inspect(sync_conn)
                return inspector.get_table_names()

            tables = await conn.run_sync(get_tables)
            print(f"Created tables: {', '.join(tables)}")

    except Exception as e:
        print(f"✗ Error initializing database: {e}")
        sys.exit(1)
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())