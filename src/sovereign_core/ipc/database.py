"""SQLite database layer for IPC communication."""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional

from .models import Command, CommandResult, CommandStatus

logger = logging.getLogger(__name__)


@contextmanager
def get_connection(db_path: str):
    """Context manager for database connections."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def init_database(db_path: str) -> None:
    """Initialize the database with required tables.
    
    Args:
        db_path: Path to the SQLite database file
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Create commands table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                speaker_id TEXT,
                action TEXT NOT NULL,
                params TEXT NOT NULL,
                status TEXT NOT NULL
            )
        """)
        
        # Create results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command_id INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                error TEXT,
                completed_at TEXT NOT NULL,
                FOREIGN KEY (command_id) REFERENCES commands (id)
            )
        """)
        
        # Create index on command_id for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_command_id 
            ON results (command_id)
        """)
        
        logger.info(f"Database initialized at {db_path}")


def create_command(
    db_path: str,
    action: str,
    params: dict[str, Any],
    speaker_id: Optional[str] = None,
) -> int:
    """Create a new command in the database.
    
    Args:
        db_path: Path to the SQLite database file
        action: Action to be performed (e.g., "spotify.play_query")
        params: Parameters for the action as a dictionary
        speaker_id: Optional speaker identifier for future use
        
    Returns:
        The ID of the created command
    """
    command = Command.create(action=action, params=params, speaker_id=speaker_id)
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO commands (timestamp, speaker_id, action, params, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                command.timestamp,
                command.speaker_id,
                command.action,
                json.dumps(command.params),
                command.status.value,
            ),
        )
        command_id = cursor.lastrowid
        logger.debug(f"Created command {command_id}: {action}")
        return command_id


def get_command_result(db_path: str, command_id: int) -> Optional[CommandResult]:
    """Get the result for a command if available.
    
    Args:
        db_path: Path to the SQLite database file
        command_id: The ID of the command to get results for
        
    Returns:
        CommandResult if available, None otherwise
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, command_id, outcome, error, completed_at
            FROM results
            WHERE command_id = ?
            ORDER BY completed_at DESC
            LIMIT 1
            """,
            (command_id,),
        )
        row = cursor.fetchone()
        
        if row is None:
            logger.debug(f"No result found for command {command_id}")
            return None
        
        result = CommandResult(
            id=row["id"],
            command_id=row["command_id"],
            outcome=row["outcome"],
            error=row["error"],
            completed_at=row["completed_at"],
        )
        logger.debug(f"Retrieved result for command {command_id}")
        return result