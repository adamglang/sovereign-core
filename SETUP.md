# Sovereign Core - Setup Guide

Complete setup instructions to get the voice assistant POC running.

## Prerequisites

### System Requirements
- **Operating System:** Windows 11 (native, not WSL)
- **Python:** 3.11 or 3.12 (3.13 not yet supported)
- **Microphone:** Working audio input device
- **Internet:** Required for OpenAI API calls and initial model downloads

### API Keys Required
You must obtain and configure the following API keys:

1. **OpenAI API Key**
   - Sign up at: https://platform.openai.com/
   - Create an API key at: https://platform.openai.com/api-keys
   - Used for: Conversational AI and intent classification
   
2. **Porcupine Access Key** (for wake word detection)
   - Sign up at: https://console.picovoice.ai/
   - Create an access key in your account dashboard
   - Free tier available with usage limits
   - Used for: "Hey Sovereign" wake word detection

## Installation

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd sovereign-core
```

### Step 2: Create Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Install the package with all dependencies
pip install -e .

# For development (includes testing tools)
pip install -e ".[dev]"
```

### Step 4: Configure Environment Variables
Create a `.env` file in the project root or set environment variables:

**Required:**
```bash
# OpenAI API Key
set OPENAI_API_KEY=sk-your-openai-api-key-here

# Porcupine Access Key
set PORCUPINE_ACCESS_KEY=your-porcupine-access-key-here
```

**Optional:**
```bash
# Override default config file location
set SOVEREIGN_CONFIG_PATH=path/to/your/config.yaml
```

### Step 5: Verify Configuration
The `config.yaml` file should already exist at the project root. Review and adjust settings if needed:

```yaml
# Key sections to verify:
wake_word:
  access_key: ${PORCUPINE_ACCESS_KEY}  # Uses environment variable
  sensitivity: 0.5  # Adjust 0.0-1.0 for wake word sensitivity

stt:
  model_size: "base"  # Options: tiny, base, small, medium, large-v2, large-v3
  device: "cpu"       # Use "cuda" if you have NVIDIA GPU

llm:
  model: "gpt-4o-mini"  # Cost-efficient model for POC
  temperature: 0.7

logging:
  level: "INFO"        # Change to "DEBUG" for troubleshooting
```

## First Run

### Quick Test
```bash
# Ensure virtual environment is activated
python -m sovereign_core.main
```

### Expected Behavior
1. **Initialization:**
   - Logs appear showing component initialization
   - Models download on first run (may take a few minutes)
   - Console shows: `🎤 Listening for 'Hey Sovereign'...`

2. **Wake Word Detection:**
   - Say "Hey Sovereign" clearly into your microphone
   - Console shows: `✅ Wake word detected! Speak now...`

3. **Voice Interaction:**
   - Speak your question or command (5 seconds of audio capture)
   - System transcribes your speech
   - Shows: `💬 You: [your transcribed text]`
   - Processes request and speaks response

4. **Shutdown:**
   - Press `Ctrl+C` to stop gracefully
   - All components cleanup properly

### Testing Commands

**Conversational Examples:**
- "Hey Sovereign... What is the weather today?"
- "Hey Sovereign... Why is the sky blue?"
- "Hey Sovereign... Tell me a joke"

**Action Examples (requires Sovereign Executor):**
- "Hey Sovereign... Play 46 and 2 by Tool"
- "Hey Sovereign... Pause the music"
- "Hey Sovereign... Resume playback"

## Troubleshooting

### Wake Word Not Detected
**Problem:** System doesn't respond to "Hey Sovereign"

**Solutions:**
1. Check microphone is working and selected as default
2. Adjust `sensitivity` in config.yaml (try 0.7)
3. Speak clearly and at normal volume
4. Verify PORCUPINE_ACCESS_KEY is set correctly
5. Check logs for Porcupine initialization errors

### Speech Not Transcribed
**Problem:** Wake word detected but transcription fails

**Solutions:**
1. Check internet connection (models download on first use)
2. Verify microphone permissions
3. Check `logs/sovereign.log` for errors
4. Try smaller model: set `model_size: "tiny"` in config.yaml
5. Speak after the beep, within 5 seconds

### OpenAI API Errors
**Problem:** "API key not found" or rate limit errors

**Solutions:**
1. Verify OPENAI_API_KEY is set: `echo %OPENAI_API_KEY%`
2. Check API key is valid at https://platform.openai.com/api-keys
3. Verify you have API credits available
4. Check rate limits if using free tier

### Audio Device Issues
**Problem:** "Failed to access microphone" error

**Solutions:**
1. Run as administrator (may be needed for microphone access)
2. Check Windows privacy settings → Microphone access
3. Test microphone in Windows Sound settings
4. List available devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`
5. Set specific device in config.yaml: `device_index: 0`

### Model Download Failures
**Problem:** First run fails downloading models

**Solutions:**
1. Check internet connection
2. Verify firewall isn't blocking downloads
3. Check available disk space (models ~500MB total)
4. Try manual download to `./models/whisper/` directory

### High CPU Usage
**Problem:** Computer becomes slow during use

**Solutions:**
1. Use smaller STT model: `model_size: "tiny"` or `"base"`
2. Disable GPU if causing issues: `device: "cpu"`
3. Close other applications
4. Reduce `temperature` in LLM config for faster responses

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_config.py -v
```

### Run Tests with Coverage
```bash
pytest --cov=sovereign_core --cov-report=html
```

### Skip Integration Tests (require API keys)
```bash
pytest -m "not integration"
```

## Development

### Code Quality
```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy sovereign_core/
```

### Project Structure
```
sovereign-core/
├── config.yaml           # Main configuration
├── sovereign_core/       # Source code
│   ├── ears/            # Wake word + audio capture
│   ├── brain/           # STT + LLM processing
│   ├── mouth/           # TTS output
│   ├── router/          # Intent classification
│   ├── ipc/             # Database communication
│   ├── config.py        # Configuration loader
│   └── main.py          # Main orchestrator
├── tests/               # Test suite
├── logs/                # Runtime logs
└── models/              # Downloaded AI models
```

## Performance Expectations (POC)

### Resource Usage
- **RAM:** ~2-4GB (depends on STT model size)
- **CPU:** Moderate during transcription, low when idle
- **Disk:** ~500MB for models
- **Network:** API calls only (no continuous streaming)

### Response Times
- **Wake word detection:** <100ms
- **Audio capture:** 5 seconds (configurable)
- **Transcription:** 1-3 seconds (base model, CPU)
- **LLM response:** 1-5 seconds (depends on OpenAI API)
- **TTS playback:** Real-time (depends on response length)

### Model Download Times (first run)
- **Whisper base:** ~150MB, 2-5 minutes
- **Porcupine:** Included in package
- Total: ~5-10 minutes on first run

## Next Steps

### After POC Works
1. **Integrate Sovereign Executor** (for actual command execution)
2. **Test Spotify Integration** (play music commands)
3. **Optimize Model Selection** (balance speed vs accuracy)
4. **Configure Auto-start** (Windows service or startup task)
5. **Add Voice Profiles** (multi-user support)

### Production Considerations
- Replace cloud LLM with local (e.g., Ollama)
- Add authentication/authorization
- Implement conversation context persistence
- Add more action handlers (smart home, calendar, etc.)
- Optimize wake word sensitivity for environment
- Add conversation history limits and cleanup

## Support

### Logs Location
- **Main log:** `./logs/sovereign.log`
- Check this first for any errors or issues

### Common Log Messages
- `"Listening for wake word..."` - System ready
- `"Wake word detected!"` - Successfully heard "Hey Sovereign"
- `"Transcription completed"` - Speech-to-text successful
- `"Action queued"` - Command sent to IPC database

### Getting Help
1. Check logs in `./logs/sovereign.log`
2. Enable DEBUG logging in config.yaml
3. Review error messages carefully
4. Verify all environment variables are set
5. Test components individually with test suite

## Security Notes

### API Key Security
- ⚠️ Never commit API keys to version control
- ⚠️ Use environment variables or .env files (gitignored)
- ⚠️ Rotate keys regularly
- ⚠️ Monitor API usage for unexpected activity

### Privacy Considerations
- Audio data is processed locally (STT runs on device)
- Transcribed text sent to OpenAI API for processing
- No audio stored or transmitted except for processing
- Review OpenAI's data usage policy

## License

MIT License - See LICENSE file for details