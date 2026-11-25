"""Database configuration and session management"""
import os
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

from .config import get_data_dir

DATABASE_DIR = get_data_dir()
os.makedirs(DATABASE_DIR, exist_ok=True)

DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_DIR}/digital_archeology.db"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    """Tune SQLite for concurrent readers alongside a writing worker.

    The analysis worker commits continuously while the HTMX poll reads every
    couple of seconds. Under the default rollback journal those readers block
    on every write; WAL lets them proceed. busy_timeout stops a transient lock
    from surfacing as a 500 mid-poll.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=15000")
        cursor.execute("PRAGMA synchronous=NORMAL")
    finally:
        cursor.close()


async def get_db():
    """Dependency for getting database sessions"""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def _ensure_column(conn, table: str, column: str, ddl_type: str) -> None:
    """Add a column if it is missing.

    create_all() only creates whole tables, so columns added to an existing
    model need this. Kept alongside the index below rather than pulling in a
    migration tool for a single-file SQLite database.
    """
    result = await conn.exec_driver_sql(f"PRAGMA table_info({table})")
    existing = {row[1] for row in result.fetchall()}
    if column not in existing:
        await conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # create_all adds indexes only for tables it creates; ensure the
        # performance index exists on pre-existing databases too (idempotent).
        await conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_archived_files_project_status "
            "ON archived_files (project_id, status)"
        )
        # Dedupe picks one representative capture per URL; without this index
        # that lookup is a full scan per candidate row.
        await conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_archived_files_project_url "
            "ON archived_files (project_id, original_url)"
        )
        await _ensure_column(conn, "file_analyses", "source_file_id", "INTEGER")
        await _ensure_column(conn, "archived_files", "last_raw_response", "TEXT")
        # Captures the archive recorded as 4xx/5xx used to be fetched and marked
        # ERROR ("HTTP 404"), which counted crawl-time losses as our failures.
        # The downloader now skips them; relabel rows from before. Idempotent.
        await conn.exec_driver_sql(
            "UPDATE archived_files "
            "SET status = 'SKIPPED', "
            "    error_message = 'Archive only holds an error page (HTTP ' || status_code || ')' "
            "WHERE status = 'ERROR' AND error_message LIKE 'HTTP %' "
            "  AND status_code GLOB '[45][0-9][0-9]'"
        )
        await conn.exec_driver_sql(
            "UPDATE archived_files "
            "SET status = 'SKIPPED', "
            "    error_message = 'Redirect target was never archived (capture HTTP ' || status_code || ')' "
            "WHERE status = 'ERROR' AND error_message IN ('HTTP 403', 'HTTP 404') "
            "  AND status_code GLOB '3[0-9][0-9]'"
        )
        # Analysis scope (see Project.scope_start). Existing rows default to
        # in-scope, so adding this changes nothing until a window is set.
        await _ensure_column(conn, "projects", "scope_start", "VARCHAR(14)")
        await _ensure_column(conn, "projects", "scope_end", "VARCHAR(14)")
        await _ensure_column(conn, "projects", "scope_reason", "TEXT")
        await _ensure_column(conn, "projects", "exclude_patterns", "TEXT")
        await _ensure_column(conn, "archival_cards", "whois_domain", "VARCHAR(255)")
        await _ensure_column(
            conn, "archived_files", "in_scope", "BOOLEAN NOT NULL DEFAULT 1"
        )
        await conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_archived_files_project_scope "
            "ON archived_files (project_id, in_scope, status)"
        )
        # Evidence-backed findings on the archival card (link_targets and
        # page_links are whole new tables, so create_all handles them).
        await _ensure_column(conn, "archival_cards", "findings", "JSON")
        await _ensure_column(conn, "archival_cards", "findings_model", "VARCHAR(100)")
        await _ensure_column(conn, "archival_cards", "findings_generated_at", "DATETIME")
