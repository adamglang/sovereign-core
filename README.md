# Sovereign Core

The **intelligence and interaction layer** of Sovereign — our family's local-first, always-available voice assistant.

## What It Is

Sovereign Core is the brain, ears, and mouth. It:
- **Listens** via wake-word detection ("Hey Sovereign")
- **Understands** through speech-to-text and conversational AI
- **Reasons** to decide between conversation and action
- **Speaks** responses through text-to-speech

**Critical:** Sovereign Core never executes real-world actions. It only thinks and talks.

---

## Responsibilities

### 🎤 Ears
- Always-on microphone capture
- Wake-word detection (Picovoice Porcupine)
- Hotkey/click-to-talk fallback
- Audio stream management

### 🧠 Brain
- Speech-to-text (faster-whisper)
- Conversational intelligence (LLM)
- Follow-up question handling
- Curiosity chains
- Age-appropriate response shaping

### 🚦 Router (Critical Safety Boundary)
- Decides: Conversational mode vs Action mode
- Enforces: No execution in conversation, no prose in action
- Outputs structured JSON intents when actions are needed

### 🗣️ Mouth
- Text-to-speech (Piper or Windows native)
- Spoken responses
- Kid-friendly tone shaping

---

## What It Does NOT Do

- ❌ Execute system commands
- ❌ Control applications directly
- ❌ Make assumptions about execution success
- ❌ Validate or enforce permissions

Those are the responsibilities of **Sovereign Executor**.

---

## Technology Stack

- **Language:** Python
- **Runtime:** Native Windows (not WSL)
- **Wake Word:** Picovoice Porcupine
- **Audio:** sounddevice or pyaudio
- **STT:** faster-whisper (local, offline)
- **LLM:** Pluggable (cloud or local)
- **TTS:** Piper (preferred) or Windows native

**Why Python on Windows?**
- Reliable microphone access
- Better wake-word support
- Easier GPU access for STT
- Lower friction for always-on services

---

## IPC with Executor

Communication happens via **SQLite database**:
- Core writes action intents to `commands` table
- Core reads results from `results` table
- No direct execution or validation

---

## POC Goals

For the proof-of-concept, Sovereign Core must:
1. Detect "Hey Sovereign" wake word
2. Capture and transcribe speech
3. Answer conversational questions with follow-ups
4. Route action intents correctly
5. Emit valid structured JSON for Spotify playback
6. Speak responses aloud

---

## Architecture

For complete architectural details, see [`docs/architecture.md`](docs/architecture.md).

For overall project context, see [`../docs/sovereign.md`](../docs/sovereign.md).

---

## Mental Model

> Sovereign Core is the assistant you interact with.  
> Sovereign Executor is invisible.

Core listens, thinks, decides, and speaks.  
It never acts.
