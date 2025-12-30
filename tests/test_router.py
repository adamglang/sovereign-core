"""Test Router functionality."""

import os
import tempfile
from pathlib import Path

import pytest

from sovereign_core.brain.llm_factory import get_llm_provider
from sovereign_core.ipc.database import get_command_result, init_database
from sovereign_core.router import IntentType, Router, RouterResponse


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    init_database(db_path)
    yield db_path
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def router(test_db):
    """Create a Router instance for testing."""
    # Skip tests if no OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    llm_provider = get_llm_provider(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000,
    )
    
    action_keywords = ["play", "pause", "stop", "resume", "next", "previous"]
    
    return Router(
        llm_provider=llm_provider,
        db_path=test_db,
        action_keywords=action_keywords,
    )


def test_conversational_intent(router):
    """Test that conversational questions are handled correctly."""
    response = router.route("Why is the sky blue?")
    
    assert response.type == IntentType.CONVERSATIONAL
    assert response.conversational_response is not None
    assert len(response.conversational_response) > 0
    assert response.action_intent is None
    assert response.command_id is None


def test_action_intent(router, test_db):
    """Test that action commands create IPC commands."""
    response = router.route("Play 46 & 2 by Tool")
    
    assert response.type == IntentType.ACTION
    assert response.action_intent is not None
    assert response.command_id is not None
    assert response.conversational_response is None
    
    # Verify action format
    action_data = response.action_intent
    assert "action" in action_data
    assert "params" in action_data
    assert action_data["action"] == "spotify.play_query"
    assert "query" in action_data["params"]
    assert "46 & 2" in action_data["params"]["query"].lower()


def test_clarification_needed(router):
    """Test that ambiguous requests trigger clarification."""
    response = router.route("Play something")
    
    assert response.type == IntentType.CLARIFICATION_NEEDED
    assert response.clarification_question is not None
    assert len(response.clarification_question) > 0
    assert response.action_intent is None
    assert response.command_id is None


def test_pause_action(router):
    """Test pause command handling."""
    response = router.route("Pause the music")
    
    assert response.type == IntentType.ACTION
    assert response.action_intent is not None
    
    action_data = response.action_intent
    assert action_data["action"] == "spotify.pause"
    assert action_data["params"] == {}


def test_with_conversation_history(router):
    """Test conversational responses with history."""
    history = [
        {"role": "user", "content": "What are stars?"},
        {"role": "assistant", "content": "Stars are massive balls of hot gas..."},
    ]
    
    response = router.route("Why do they twinkle?", conversation_history=history)
    
    assert response.type == IntentType.CONVERSATIONAL
    assert response.conversational_response is not None


def test_router_never_executes_actions(router, test_db):
    """Verify Router only writes to IPC, never executes."""
    response = router.route("Play some jazz")
    
    # Router should only have created a command, not executed it
    assert response.type == IntentType.ACTION
    assert response.command_id is not None
    
    # Check that command is in database with QUEUED status
    result = get_command_result(test_db, response.command_id)
    # Result should be None because executor hasn't processed it yet
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])