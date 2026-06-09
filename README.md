# Persona Engine — Decompose, Compose & Vibe-Code Personalities

<div align="center">

**A voice is not just *what* someone says. It's *how* they say it.**

[![I2I Protocol](https://img.shields.io/badge/protocol-I2I%20v2.1-blue)](#i2i-integration)
[![Pipeline](https://img.shields.io/badge/pipeline-Fleet--MIDI--Pulse-ff69b4)](https://github.com/SuperInstance/fleet-midi-pulse)
[![Status](https://img.shields.io/badge/status-alpha-yellow)]()

</div>

---

## What This Is

The Persona Engine decomposes human speech into **personality vectors** — cadence profiles, prosody envelopes, lexical fingerprints, and groove parameters — then composes *any* content through that persona's voice. It's how you build a system that sounds like a specific scientist explaining a paper they never actually read, because the *way* they think has been decomposed and recomposed.

**In Casey's words:**

> *"Create vector databases for specific people that then complement the repo-agent that thinks like that specific personality. For generative content, create a fictional character and dial in aspects of the persona as vibe-coding character-building."*

---

## Architecture

```
Source Audio                              Content + Persona ID
      │                                         │
      ▼                                         ▼
┌──────────────┐                      ┌───────────────────┐
│  Decompose    │                      │  Compose          │
│               │                      │                   │
│  Audio  ─────►│  OpenSMILE eGeMAPS   │  Content ────────►│  Rhythmic Adapter
│  Preprocessor │  (25 features)       │  Persona Vector   │  (cadence → timing)
│               │                      │                   │
│  Cadence ────►│  Pause Analysis      │  Prosody ────────►│  SSML Generator
│  Analyzer     │  Turn Boundaries     │  Shaper           │  (F0, rate, energy)
│               │  Speaking Rate       │                   │
│  Prosody ────►│  F0 Contour          │  Groove ─────────►│  Piper TTS
│  Analyzer     │  Energy Envelope     │  Layer            │  (voice rendering)
│               │                      │                   │
│  Lexical ────►│  STT + Analysis      │  Phone ──────────►│  Latency Shaper
│  Fingerprint  │  Phrase Patterns     │  Feel             │  (0.5s delay ≈ real)
└──────┬───────┘                      └────────┬──────────┘
       │                                       │
       ▼                                       ▼
┌──────────────────────────────────────────────────────┐
│                PERSONA VECTOR DB                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐             │
│  │ Cadence  │ │ Prosody  │ │  Groove  │             │
│  │ Profile  │ │Envelope  │ │  Params  │             │
│  └──────────┘ └──────────┘ └──────────┘             │
│  ┌──────────┐ ┌──────────┐                           │
│  │  Lexical │ │  Persona │  ← semantic embedding     │
│  │ Print    │ │  Vector  │    for similarity search   │
│  └──────────┘ └──────────┘                           │
└──────────────────────────────────────────────────────┘
```

---

## The Groove Insight

**Conversation is rhythmic, not transactional.** The latency in a phone call isn't a bug — it's the pocket. Humans learn each other's timing within milliseconds and start laughing before the punchline.

The groove engine maps musical timing to conversation:

| Musical Concept | Conversational Equivalent |
|----------------|--------------------------|
| BPM | Turns per minute (60 ≥ leisurely chat, 120 ≥ intense debate) |
| Swing | Turn imbalance (0.0 = polite alternation, 0.3 = overlapping enthusiasm) |
| Fermata | Pregnant pause before a big thought |
| Accent | Punchline / key insight |
| Anticipation | Laughing before the punchline (human timing prediction) |
| Call & Response | Q&A rhythm |
| Walking Bass | Steady conversational flow (always someone "walking") |

This is the key insight: **a good conversation has a groove, just like a good jazz performance.** The same persona that speaks with 45bpm and heavy swing (a storyteller) is fundamentally different from one at 80bpm with straight eighths (a scientist), even when saying the same words.

---

## Quick Start

### Decompose a Persona from Audio

```bash
# Extract a scientist's voice signature from a podcast
python -m persona_engine.cli.character decompose feynman_podcast.mp3 \
    --speaker "Richard Feynman"
# → Extracts cadence, prosody, groove → stores as persona vector
```

### Create a Vibe-Coded Character

```bash
# Build from a preset
python -m persona_engine.cli.character create jazz_vocalist

# Dial in parameters like a mixing board
python -m persona_engine.cli.character dial feynman \
    --speed 1.2 --swing 0.15 --enthusiasm 0.8

# List all presets
python -m persona_engine.cli.character list --presets
```

### Render Content Through a Persona

```bash
# Have the persona explain something (in their voice)
python -m persona_engine.cli.character render feynman \
    "The conservation theorem shows that symmetry implies conservation"
# → Audio file with Feynman's cadence, prosody, and groove
```

### Interactive Scientist Podcast Demo

```bash
# Step 1: Build a persona from a podcast
python examples/scientist_podcast.py \
    --build-profile feynman_interview.mp3 \
    --speaker "Feynman"

# Step 2: Ask a question through their voice
python examples/scientist_podcast.py \
    --ask "Explain the conservation theorem" \
    --speaker "Feynman"

# Step 3: Interactive mode (feels like a phone call)
python examples/scientist_podcast.py --interactive
```

---

## Character Presets

| Preset | BPM | Swing | Enthusiasm | Vibe |
|--------|-----|-------|------------|------|
| **scientist** | 70 | 0.1 | 0.4 | Precise, technical, moderate pace |
| **jazz_vocalist** | 55 | 0.35 | 0.7 | Laid back, swinging, warm |
| **storyteller** | 45 | 0.25 | 0.9 | Expansive, rubato, animated |
| **feynman** | 80 | 0.15 | 0.8 | Quick, playful, explanatory |
| **noir_detective** | 35 | 0.1 | 0.1 | Slow, gravelly, deliberate |
| **bard** | 50 | 0.2 | 0.6 | Lyrical, formal, melodic |
| **mickey** | 100 | 0.3 | 1.0 | High energy, bouncy, gag-timing |

Each preset is a starting point. Dial parameters to find your character.

---

## I2I Integration

Personas are discoverable fleet agents. Any agent can send bottles:

```json
{
  "type": "PERSONA_DECOMPOSE",
  "payload": {
    "audio_path": "/tmp/feynman.wav",
    "speaker": "Richard Feynman"
  }
}
```

| Bottle Type | Function |
|-------------|----------|
| `PERSONA_DECOMPOSE` | Decompose audio → persona vector |
| `PERSONA_COMPOSE` | Render content through a persona |
| `PERSONA_QUERY` | Search persona database |
| `PERSONA_COMPARE` | Compare two personas (similarity score) |
| `PERSONA_CREATE_CHARACTER` | Create a fictional character persona |

---

## The Scientific Telephone

The end state is a system where:

1. You feed it podcast interviews with a scientist
2. It decomposes their **cadence** (pause patterns, turn-taking, thought duration), **prosody** (F0 contour, energy shapes), **lexical fingerprint** (sentence structure, signature phrases), and **groove** (conversational rhythm, swing, anticipation)
3. You ask questions about a paper they never discussed — total rabbit trail from anything in their recordings
4. The system synthesizes an answer using the paper's content AND renders it **through the scientist's voice** — same cadence, same timing, same feel
5. The latency feels like a phone call (not a robot) — because the groove engine composes with delay, not against it

---

## Related

- [fleet-midi-pulse](https://github.com/SuperInstance/fleet-midi-pulse) — BPM/swing/fermata timing layer (groove backbone)
- [fleet-i2i-protocol](https://github.com/SuperInstance/fleet-i2i-protocol) — I2I wire format (bottle transport)
- [A2A-native-notebookLM](https://github.com/SuperInstance/A2A-native-notebookLM) — Fleet cognitive command center (surround research/synthesis)
- [OpenSMILE](https://audeering.com/technology/opensmile/) — Feature extraction (eGeMAPS)
- [Piper TTS](https://github.com/rhasspy/piper) — Voice rendering (SSML prosody)

---

## License

MIT
