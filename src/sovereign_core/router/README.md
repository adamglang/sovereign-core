# Router Module

The Router is the **MOST CRITICAL** component in sovereign-core. It enforces the safety boundary between conversational and action modes.

## Architecture

The Router analyzes user utterances and decides between THREE possible outcomes:

1. **CONVERSATIONAL** → Natural language response (no actions)
2. **ACTION** → Structured JSON written to IPC database (for executor to process)
3. **CLARIFICATION_NEEDED** → Ask a clarifying question

## Safety Through Separation

**CRITICAL PRINCIPLE:** Safety comes from SEPARATION, not cleverness.

- **Conversational mode:** NO execution, NO structured output, NO JSON
- **Action mode:** NO prose, only JSON written to IPC
- **Ambiguous intent:** Ask clarification instead of guessing

The Router NEVER executes actions directly. It only writes commands to the IPC database.

## Components

### 1. Models (`models.py`)

- **`IntentType`** enum: CONVERSATIONAL, ACTION, CLARIFICATION_NEEDED
- **`RouterResponse`** model: Discriminated union based on intent type
  - Only one payload field is populated per response type
  - Runtime validation ensures correct structure

### 2. Prompts (`prompts.py`)

System prompts for different routing tasks:

- **`get_classification_prompt()`**: Intent classification with examples
- **`get_action_extraction_prompt()`**: Structured action extraction
- **`get_conversational_prompt()`**: Natural conversational responses
- **`get_clarification_prompt()`**: Clarification question generation

### 3. Intent Classifier (`intent_classifier.py`)

- **`classify_intent()`**: Uses LLM to classify user utterances
- LLM-based (not regex) to handle natural language variations
- Low temperature (0.1) for consistent classification
- Safe fallback to CONVERSATIONAL on errors

### 4. Router (`router.py`)

Main orchestration class:

```python
router = Router(
    llm_provider=llm_provider,
    db_path="./sovereign.db",
    action_keywords=["play", "pause", "stop", "resume"]
)

response = router.route("Play 46 & 2 by Tool")
# Returns RouterResponse with type=ACTION, command_id set
```

## Usage Example

```python
from sovereign_core.router import Router, IntentType
from sovereign_core.brain.llm_factory import create_llm_provider

# Initialize
llm_provider = create_llm_provider(provider="openai", model="gpt-4o-mini")
router = Router(
    llm_provider=llm_provider,
    db_path="./sovereign.db",
    action_keywords=["play", "pause", "stop"]
)

# Route conversational query
response = router.route("Why is the sky blue?")
assert response.type == IntentType.CONVERSATIONAL
print(response.conversational_response)

# Route action command
response = router.route("Play 46 & 2 by Tool")
assert response.type == IntentType.ACTION
assert response.command_id is not None
print(f"Command {response.command_id} queued")

# Route ambiguous request
response = router.route("Play something")
assert response.type == IntentType.CLARIFICATION_NEEDED
print(response.clarification_question)
```

## Supported Actions (POC)

Music control via Spotify:
- `spotify.play_query` - Play specific song/artist/album
- `spotify.pause` - Pause playback
- `spotify.resume` - Resume playback
- `spotify.next_track` - Skip to next
- `spotify.previous_track` - Go to previous

Action format:
```json
{
  "action": "spotify.play_query",
  "params": {"query": "46 & 2 by Tool"}
}
```

## Testing

Run tests with:
```bash
cd sovereign-core
pytest tests/test_router.py -v
```

Tests verify:
- Conversational intents return natural language
- Action intents create IPC commands with correct format
- Ambiguous requests trigger clarification
- Router never executes actions (only writes to IPC)

## Integration

The Router is designed to be orchestrated by the main application:

```python
# In main application loop
while True:
    utterance = get_user_input()  # From speech recognition
    
    response = router.route(utterance, conversation_history)
    
    if response.type == IntentType.CONVERSATIONAL:
        speak(response.conversational_response)
    elif response.type == IntentType.ACTION:
        # Command written to IPC, executor will handle
        speak("OK, working on it")
    elif response.type == IntentType.CLARIFICATION_NEEDED:
        speak(response.clarification_question)
```

## Configuration

Action keywords are configured in `config.yaml`:

```yaml
router:
  action_keywords:
    - "execute"
    - "run"
    - "play"
    - "pause"
    - "stop"
```

## Safety Guarantees

1. **No Direct Execution**: Router ONLY writes to IPC database
2. **Separation of Concerns**: Clear boundaries between conversation and action
3. **Default to Safe**: On errors, defaults to conversational (not action)
4. **Validation**: Action format validated before writing to IPC
5. **Clarification Over Guessing**: Ambiguous requests trigger questions

## Future Enhancements

- Multi-step action planning
- Context-aware clarification
- Speaker identification integration
- More action categories (system, apps, etc.)
- Confidence scoring for classification