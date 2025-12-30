"""Pydantic models for IPC communication."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class CommandStatus(str, Enum):
    """Command status enum."""

    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class Command(BaseModel):
    """Represents a command in the IPC queue."""

    id: Optional[int] = None
    timestamp: str
    speaker_id: Optional[str] = None
    action: str
    params: dict[str, Any]
    status: CommandStatus = CommandStatus.QUEUED

    @classmethod
    def create(
        cls, action: str, params: dict[str, Any], speaker_id: Optional[str] = None
    ) -> "Command":
        """Create a new command with current timestamp."""
        return cls(
            timestamp=datetime.utcnow().isoformat(),
            speaker_id=speaker_id,
            action=action,
            params=params,
            status=CommandStatus.QUEUED,
        )


class CommandResult(BaseModel):
    """Represents a command result."""

    id: Optional[int] = None
    command_id: int
    outcome: str
    error: Optional[str] = None
    completed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())