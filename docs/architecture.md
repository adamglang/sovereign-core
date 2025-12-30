# Sovereign Core — Architecture & Design

## Role of Sovereign Core

Sovereign Core is the **intelligence and interaction layer** of the system.

It is responsible for:
- Listening
- Understanding
- Reasoning
- Deciding
- Speaking

It is **never allowed to execute real-world actions**.

Mental model:
> Sovereign Core is the assistant you interact with.  
> Sovereign Executor is invisible.

---

## Responsibilities (Strict)

### 1. Ears

- Always-on microphone capture
- Wake-word detection ("Hey Sovereign")
- Optional fallback activation:
  - Hotkey
  - Click-to-talk
- Audio stream management

Wake-word detection must be **local and offline**.

---

### 2. Brain

- Speech-to-text (STT)
- Conversational intelligence
- Follow-up handling
- Curiosity chains
- Age-appropriate response shaping (child vs adult)
- Planning and explanation

The brain may:
- Ask clarification questions
- Handle multi-turn conversation

The brain may **not**:
- Execute actions
- Assume device state
- Claim execution occurred

---

### 3. Router (Critical Safety Boundary)

The router decides how each utterance is handled.

Possible outcomes:
1. Conversational response
2. Action intent
3. Clarification request

Rules:
- Conversational mode → natural language output
- Action mode → structured intent output **only**
- Ambiguous intent → ask one clarifying question

The router enforces:
- No execution in conversation mode
- No prose in action mode

---

### 4. Mouth

- Text-to-speech (TTS)
- Spoken answers
- Spoken confirmations (optional)
- Tone shaping (especially for children)

The mouth:
- Speaks what the brain decides
- Never executes anything
- Never talks about internal mechanics

---

## Language & Runtime

- **Language:** Python
- **Runtime:** Native Windows (not WSL)

Reasoning:
- Reliable microphone access
- Reliable wake-word handling
- Easier GPU access for STT
- Lower friction for always-on services

---

## Core Libraries

### Wake Word

- Picovoice Porcupine

Requirements:
- Low CPU usage
- Offline
- Continuous listening

---

### Audio Capture

- sounddevice or pyaudio

Requirements:
- Low latency
- Stable device selection
- Continuous streaming

---

### Speech-to-Text

- faster-whisper (local)

Requirements:
- Offline
- Acceptable CPU performance
- GPU acceleration if available
- Streaming or near-streaming behavior preferred

---

### LLM (Brain)

**POC Implementation:**
- OpenAI API (GPT-5)

**CRITICAL DESIGN PRINCIPLE: Model Swappability**

The LLM backend MUST be architected for **zero-friction model swapping**.

Swapping from OpenAI to another hosted LLM (Anthropic, Gemini, etc.) or to a local 72B model should require:
- **One configuration change only:** point to the new model endpoint/path
- **Zero code changes**
- **Zero additional configuration**

This is achieved through:
1. **Abstraction layer:** A unified interface that all LLM backends implement
2. **Provider adapters:** Thin wrappers that translate the standard interface to provider-specific APIs
3. **Runtime selection:** Model backend chosen at startup via single config parameter

The abstraction must handle:
- Message formatting (system/user/assistant)
- Streaming responses (if provider supports)
- Temperature and generation parameters
- Error handling and retry logic

Future model targets include:
- Local 72B models (post-RTX 5090 upgrade)
- Other cloud providers (Anthropic Claude, Google Gemini)
- Self-hosted inference servers (vLLM, TGI)

**Functional Requirements:**
- Multi-turn conversation
- Ability to follow strict output instructions
- Ability to emit structured JSON when explicitly requested
- Consistent behavior across providers (within model capability limits)

---

### Text-to-Speech

- Piper (preferred)
- Windows native voices acceptable for early POC

Requirements:
- Low latency
- Always available
- No cloud dependency required for POC

---

## Structured Action Intent (Output Contract)

When the router selects **Action Mode**, output must be:

- JSON only
- One action per utterance
- No prose
- No assumptions of success

Example:
```json
{
  "action": "spotify.play_query",
  "params": {
    "query": "46 & 2 by Tool"
  }
}
```

This output is written to the shared SQLite queue.

---

## IPC with Sovereign Executor

- Communication via SQLite database
- Core inserts rows into commands
- Core does not execute or validate actions
- Core may read results later to speak confirmations

Core must assume:
- Executor may fail
- Executor may be delayed
- No execution guarantees

---

## POC Responsibilities (Core)

For POC, Sovereign Core must:
- Detect wake word
- Capture speech
- Convert speech to text
- Answer conversational questions with follow-ups
- Route action intent correctly
- Emit valid structured intent for Spotify playback
- Speak responses aloud

---

## Explicit Non-Responsibilities

Sovereign Core must NOT:
- Control Spotify directly
- Control the OS
- Call Home Assistant
- Run scripts
- Make assumptions about execution success
- Enforce permissions (later phase)

---

## Upgrade Path (Designed In)

Future additions:
- Voice identification
- Per-speaker memory
- Profile persistence
- Permission awareness
- Visual UI overlays

None of these require architectural changes.

---

## Summary

- Sovereign Core listens, thinks, decides, and speaks
- It never acts directly
- Python is chosen for ML, audio, and iteration speed
- The router is the most critical component
- Safety comes from separation, not cleverness

This document is authoritative for the Sovereign Core repository.