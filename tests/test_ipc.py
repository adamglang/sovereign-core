"""Tests for IPC module."""

import json
import os
import sqlite3
import tempfile
from datetime import datetime

import pytest

from sovereign_core.ipc import (
    Command,
    CommandResult,
    CommandStatus,
    create_command,
    get_command_result,
    init_database,
)


@pytest.fixture
def test_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_database(path)
    yield path
    os.unlink(path)


def test_init_database(test_db):
    """Test database initialization creates tables."""
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    
    # Check commands table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='commands'"
    )
    assert cursor.fetchone() is not None
    
    # Check results table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='results'"
    )
    assert cursor.fetchone() is not None
    
    conn.close()


def test_create_command(test_db):
    """Test creating a command."""
    params = {"query": "play some jazz", "volume": 0.8}
    command_id = create_command(
        test_db,
        action="spotify.play_query",
        params=params,
        speaker_id="user_123",
    )
    
    assert command_id > 0
    
    # Verify in database
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM commands WHERE id = ?", (command_id,))
    row = cursor.fetchone()
    conn.close()
    
    assert row is not None
    assert row[3] == "spotify.play_query"
    assert json.loads(row[4]) == params
    assert row[5] == "queued"
    assert row[2] == "user_123"


def test_create_command_without_speaker(test_db):
    """Test creating a command without speaker_id."""
    params = {"query": "turn on the lights"}
    command_id = create_command(
        test_db,
        action="homeassistant.turn_on",
        params=params,
    )
    
    assert command_id > 0
    
    # Verify speaker_id is None
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT speaker_id FROM commands WHERE id = ?", (command_id,))
    row = cursor.fetchone()
    conn.close()
    
    assert row[0] is None


def test_get_command_result_none(test_db):
    """Test getting result for command with no result."""
    params = {"query": "test"}
    command_id = create_command(test_db, action="test.action", params=params)
    
    result = get_command_result(test_db, command_id)
    assert result is None


def test_get_command_result_with_result(test_db):
    """Test getting result for command with result."""
    # Create a command
    params = {"query": "test"}
    command_id = create_command(test_db, action="test.action", params=params)
    
    # Manually insert a result
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    completed_at = datetime.utcnow().isoformat()
    cursor.execute(
        """
        INSERT INTO results (command_id, outcome, error, completed_at)
        VALUES (?, ?, ?, ?)
        """,
        (command_id, "success", None, completed_at),
    )
    conn.commit()
    conn.close()
    
    # Get the result
    result = get_command_result(test_db, command_id)
    
    assert result is not None
    assert result.command_id == command_id
    assert result.outcome == "success"
    assert result.error is None
    assert result.completed_at == completed_at


def test_command_model_create():
    """Test Command model creation."""
    command = Command.create(
        action="test.action",
        params={"key": "value"},
        speaker_id="user_1",
    )
    
    assert command.action == "test.action"
    assert command.params == {"key": "value"}
    assert command.speaker_id == "user_1"
    assert command.status == CommandStatus.QUEUED
    assert command.timestamp is not None


def test_command_status_enum():
    """Test CommandStatus enum values."""
    assert CommandStatus.QUEUED.value == "queued"
    assert CommandStatus.PROCESSING.value == "processing"
    assert CommandStatus.DONE.value == "done"
    assert CommandStatus.FAILED.value == "failed"