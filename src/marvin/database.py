"""Database management for persistence.

This module provides utilities for managing database sessions and migrations.
"""

import uuid
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import TypeAdapter
from pydantic_ai.messages import RetryPromptPart
from pydantic_ai.usage import Usage
from sqlalchemy import (
    JSON,
    Engine,
    ForeignKey,
    String,
    TypeDecorator,
    create_engine,
    inspect,
)
from sqlalchemy.engine.url import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)
from sqlalchemy.pool import StaticPool

from marvin.settings import settings

from .engine.llm import Message

message_adapter: TypeAdapter[Message] = TypeAdapter(Message)
usage_adapter: TypeAdapter[Usage] = TypeAdapter(Usage)

# Module-level cache for engines
_engine_cache: dict[str, create_engine] = {}
_async_engine_cache: dict[str, AsyncEngine] = {}


def serialize_message(message: Message) -> str:
    """
    The `ctx` field in the `RetryPromptPart` is optionally dict[str, Any], which is not always serializable.
    """
    for part in message.parts:
        if isinstance(part, RetryPromptPart):
            if isinstance(part.content, list):
                for content in part.content:
                    content["ctx"] = {
                        k: str(v) for k, v in (content.get("ctx", None) or {}).items()
                    }
    return message_adapter.dump_python(message, mode="json")


def get_engine() -> create_engine:
    """Get the SQLAlchemy engine for sync operations."""
    if "default" not in _engine_cache:
        url = make_url(settings.database_url)

        if url.drivername.startswith("sqlite"):
            # Check if it's in-memory
            is_memory_db = url.database in (None, "", ":memory:")
            engine = create_engine(
                str(url),
                echo=False,
                # For in-memory sqlite, we can use StaticPool so that the async and sync engines share state
                poolclass=StaticPool if is_memory_db else None,
                connect_args={"check_same_thread": False} if is_memory_db else {},
            )
        elif url.drivername.startswith("postgresql"):
            # Postgres engine (sync via psycopg2 or psycopg)
            # Typically you'd have just "postgresql://", which defaults to psycopg2
            # or "postgresql+psycopg://", if you explicitly want psycopg.
            engine = create_engine(
                str(url.set(drivername="postgresql")),  # ensure the sync driver name
                echo=False,
            )
        else:
            # Fallback / other databases
            engine = create_engine(str(url), echo=False)

        _engine_cache["default"] = engine

    return _engine_cache["default"]


def get_async_engine() -> AsyncEngine:
    """Get the SQLAlchemy engine for async operations."""
    if "default" not in _async_engine_cache:
        url = make_url(settings.database_url)

        if url.drivername.startswith("sqlite"):
            # Check if it's in-memory
            is_memory_db = url.database in (None, "", ":memory:")
            if is_memory_db:
                # For in-memory databases, share the sync engine’s connection
                sync_engine = get_engine()
                engine = create_async_engine(
                    "sqlite+aiosqlite://",
                    echo=False,
                    poolclass=StaticPool,
                    creator=lambda: sync_engine.raw_connection(),
                )
            else:
                # file-based sqlite
                engine = create_async_engine(
                    str(url.set(drivername="sqlite+aiosqlite")),
                    echo=False,
                )

        elif url.drivername.startswith("postgresql"):
            # For async PostgreSQL, the typical driver is "postgresql+asyncpg"
            engine = create_async_engine(
                str(url.set(drivername="postgresql+asyncpg")),
                echo=False,
            )
        else:
            # Fallback / other databases
            engine = create_async_engine(str(url), echo=False)

        _async_engine_cache["default"] = engine

    return _async_engine_cache["default"]


def set_engine(engine: Engine):
    """Set the SQLAlchemy engine for sync operations."""
    _engine_cache["default"] = engine


def set_async_engine(engine: AsyncEngine):
    """Set the SQLAlchemy engine for async operations."""
    _async_engine_cache["default"] = engine


def utc_now() -> datetime:
    """Get the current UTC timestamp."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class DBThread(Base):
    __tablename__ = "threads"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    parent_thread_id: Mapped[str | None] = mapped_column(ForeignKey("threads.id"))
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    messages: Mapped[list["DBMessage"]] = relationship(back_populates="thread")
    llm_calls: Mapped[list["DBLLMCall"]] = relationship(back_populates="thread")

    @classmethod
    async def create(
        cls,
        session: AsyncSession,
        id: str | None = None,
        parent_thread_id: str | None = None,
    ) -> "DBThread":
        """Create a new thread record.

        Args:
            session: Database session to use
            id: Optional ID to use for the thread. If not provided, a UUID will be generated.
            parent_thread_id: Optional ID of the parent thread

        Returns:
            The created DBThread instance
        """
        thread = cls(
            id=id or str(uuid.uuid4()),
            parent_thread_id=parent_thread_id,
        )
        session.add(thread)
        await session.commit()
        await session.refresh(thread)
        return thread


class DBMessage(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.id"), index=True)
    llm_call_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("llm_calls.id"),
        default=None,
    )
    message: Mapped[dict[str, Any]] = mapped_column(JSON)
    timestamp: Mapped[datetime] = mapped_column(default=utc_now)

    thread: Mapped[DBThread] = relationship(back_populates="messages")
    llm_call: Mapped[Optional["DBLLMCall"]] = relationship(back_populates="messages")

    @classmethod
    def from_message(
        cls,
        thread_id: str,
        message: Message,
        llm_call_id: uuid.UUID | None = None,
    ) -> "DBMessage":
        return cls(
            thread_id=thread_id,
            message=serialize_message(message),
            llm_call_id=llm_call_id,
        )


class UsageType(TypeDecorator[Usage]):
    """Custom type for Usage objects that stores them as JSON in the database."""

    impl = JSON
    cache_ok = True

    def process_bind_param(
        self, value: Usage | None, dialect: Any
    ) -> dict[str, Any] | None:
        """Convert Usage to JSON before storing in DB."""
        if value is None:
            return None
        return usage_adapter.dump_python(value, mode="json")

    def process_result_value(
        self, value: dict[str, Any] | None, dialect: Any
    ) -> Usage | None:
        """Convert JSON back to Usage when loading from DB."""
        if value is None:
            return None
        return usage_adapter.validate_python(value)


class DBLLMCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.id"), index=True)
    usage: Mapped[Usage] = mapped_column(UsageType)
    timestamp: Mapped[datetime] = mapped_column(default=utc_now)

    messages: Mapped[list[DBMessage]] = relationship(back_populates="llm_call")
    thread: Mapped[DBThread] = relationship(back_populates="llm_calls")

    @classmethod
    async def create(
        cls,
        thread_id: str,
        usage: Usage,
        session: AsyncSession | None = None,
    ) -> "DBLLMCall":
        """Create a new LLM call record.

        Args:
            thread_id: ID of the thread this call belongs to
            usage: Usage information from the model
            session: Optional database session. If not provided, a new one will be created.

        Returns:
            The created DBLLMCall instance
        """
        llm_call = cls(thread_id=thread_id, usage=usage)

        if session is None:
            async with get_async_session() as session:
                session.add(llm_call)
                await session.commit()
                await session.refresh(llm_call)
                return llm_call
        else:
            session.add(llm_call)
            await session.commit()
            await session.refresh(llm_call)
            return llm_call


def ensure_tables_exist():
    engine = get_engine()
    if engine.dialect.name == "sqlite":
        # Check if it’s in-memory or if the database is empty
        if (
            engine.url.database in (None, "", ":memory:")
            or not inspect(engine).get_table_names()
        ):
            Base.metadata.create_all(engine)
    else:
        # For Postgres or other DBs, only create if no tables exist
        # or always create, depending on your preference
        if not inspect(engine).get_table_names():
            Base.metadata.create_all(engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get a database session."""
    session = Session(get_engine())
    try:
        yield session
    finally:
        session.close()


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    session = AsyncSession(get_async_engine())
    try:
        yield session
    finally:
        await session.close()


def create_db_and_tables(*, force: bool = False):
    """Create all database tables.

    Args:
        force: If True, drops all existing tables before creating new ones.
    """
    engine = get_engine()

    if force:
        Base.metadata.drop_all(engine)
        print("Database tables dropped.")

    Base.metadata.create_all(engine)
    print("Database tables created.")
