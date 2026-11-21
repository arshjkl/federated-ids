"""
Database session and initialization for the Federated Learning IDS Platform.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine
)
from sqlalchemy.pool import NullPool
from src.dashboard.db.models import Base
import os
from dotenv import load_dotenv


# Load environment variables from .env at the project root
load_dotenv()


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, database_url: Optional[str] = None):
        db_url = database_url or os.getenv(
            "DATABASE_URL",
            "sqlite+aiosqlite:///./federated_ids.db"
        )
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif db_url.startswith("postgresql://") and not db_url.startswith("postgresql+"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        self.database_url = db_url
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    def init_engine(self) -> None:
        """Initialize the async engine."""
        self.engine = create_async_engine(
            self.database_url,
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
            poolclass=NullPool,  # Use NullPool for async
            pool_pre_ping=True,
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    async def close(self) -> None:
        """Close the engine."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.session_factory = None

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session."""
        if not self.session_factory:
            self.init_engine()
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def create_tables(self) -> None:
        """Create all tables."""
        if not self.engine:
            self.init_engine()
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        """Drop all tables."""
        if not self.engine:
            self.init_engine()
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


# Global database manager instance
db_manager = DatabaseManager()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database session."""
    async with db_manager.session() as session:
        yield session


async def init_database() -> None:
    """Initialize database on startup."""
    await db_manager.create_tables()


async def close_database() -> None:
    """Close database on shutdown."""
    await db_manager.close()