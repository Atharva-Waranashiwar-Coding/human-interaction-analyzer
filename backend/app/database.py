import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
DB_PATH = Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "human_interaction.db")))

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def initialize_database(target_engine=None):
    """Idempotent additive migration for existing local SQLite files."""
    from sqlalchemy import inspect, text
    target_engine = target_engine or engine
    Base.metadata.create_all(bind=target_engine)
    additions = {
        "annotations": {"objects": "TEXT NOT NULL DEFAULT '[]'", "parent_activity_id": "VARCHAR(64)",
                        "annotation_kind": "VARCHAR(32) NOT NULL DEFAULT 'behavior'", "details": "TEXT NOT NULL DEFAULT '{}'",
                        "original_suggestion": "TEXT NOT NULL DEFAULT '{}'", "review_history": "TEXT NOT NULL DEFAULT '[]'",
                        "reviewer_feedback": "TEXT NOT NULL DEFAULT ''", "rule_version": "VARCHAR(64)",
                        "evaluation_role": "VARCHAR(32) NOT NULL DEFAULT 'unspecified'"},
        "videos": {"detections": "TEXT NOT NULL DEFAULT '[]'", "analysis_warnings": "TEXT NOT NULL DEFAULT '[]'",
                   "scene_regions": "TEXT NOT NULL DEFAULT '[]'", "review_log": "TEXT NOT NULL DEFAULT '[]'", "audio_metadata": "TEXT NOT NULL DEFAULT '{}'"},
    }
    inspector = inspect(target_engine)
    missing = any(name not in {column["name"] for column in inspector.get_columns(table)}
                  for table, columns in additions.items() for name in columns)
    database = target_engine.url.database
    if missing and database and database != ":memory:":
        import sqlite3
        backup = Path(database).with_suffix(".pre-object-backup.sqlite3")
        if backup.exists():
            backup = Path(database).with_suffix(".pre-understanding-backup.sqlite3")
        if not backup.exists():
            with sqlite3.connect(database) as source, sqlite3.connect(backup) as destination:
                source.backup(destination)
    with target_engine.begin() as connection:
        for table, columns in additions.items():
            existing = {column["name"] for column in inspect(connection).get_columns(table)}
            for name, definition in columns.items():
                if name not in existing:
                    connection.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {definition}'))
