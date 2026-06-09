#!/usr/bin/env python3
"""
Vibe-Coding Character Builder — dial in a persona like a jazz musician
picks their feel on a standard.

Usage:
    python -m persona_engine.cli.character create jazz_vocalist
    python -m persona_engine.cli.character list
    python -m persona_engine.cli.character render jazz_vocalist "Take the A Train"
    python -m persona_engine.cli.character dial jazz_vocalist --speed 1.3 --swing 0.4
    python -m persona_engine.cli.character decompose podcast.mp3 --speaker "Miles Davis"
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from persona_engine.schemas.persona import CharacterParameters, Persona, PersonaManifest

logger = logging.getLogger(__name__)

# Built-in character presets
CHARACTER_PRESETS: Dict[str, Dict[str, Any]] = {
    "jazz_vocalist": {
        "name": "Jazz Vocalist",
        "pitch_baseline": 180.0,
        "pitch_variability": 0.8,
        "speed": 0.85,
        "pause_length": 1.2,
        "pause_variability": 0.7,
        "conversational_bpm": 55.0,
        "swing": 0.35,
        "anticipation": 0.5,
        "call_response": 0.7,
        "formality": 0.3,
        "enthusiasm": 0.7,
        "humor_frequency": 0.2,
        "technical_density": 0.1,
        "verbosity": 0.6,
        "assertiveness": 0.6,
        "art_style": "noir_jazz",
        "composition_style": "call_and_response",
        "rhythm_section": "swing",
    },
    "scientist": {
        "name": "Research Scientist",
        "pitch_baseline": 120.0,
        "pitch_variability": 0.3,
        "speed": 1.1,
        "pause_length": 0.8,
        "pause_variability": 0.4,
        "conversational_bpm": 70.0,
        "swing": 0.1,
        "anticipation": 0.2,
        "call_response": 1.0,
        "formality": 0.7,
        "enthusiasm": 0.4,
        "humor_frequency": 0.1,
        "technical_density": 0.85,
        "verbosity": 0.8,
        "assertiveness": 0.7,
        "art_style": "diagrammatic",
        "composition_style": "expository",
        "rhythm_section": "straight",
    },
    "storyteller": {
        "name": "Campfire Storyteller",
        "pitch_baseline": 150.0,
        "pitch_variability": 0.9,
        "speed": 0.9,
        "pause_length": 1.5,
        "pause_variability": 0.8,
        "conversational_bpm": 45.0,
        "swing": 0.25,
        "anticipation": 0.6,
        "call_response": 0.8,
        "formality": 0.1,
        "enthusiasm": 0.9,
        "humor_frequency": 0.4,
        "technical_density": 0.1,
        "verbosity": 0.9,
        "assertiveness": 0.8,
        "art_style": "watercolor",
        "composition_style": "narrative_arch",
        "rhythm_section": "rubato",
    },
    "feynman": {
        "name": "Richard Feynman",
        "pitch_baseline": 135.0,
        "pitch_variability": 0.7,
        "speed": 1.2,
        "pause_length": 0.6,
        "pause_variability": 0.5,
        "conversational_bpm": 80.0,
        "swing": 0.15,
        "anticipation": 0.25,
        "call_response": 0.9,
        "formality": 0.3,
        "enthusiasm": 0.8,
        "humor_frequency": 0.3,
        "technical_density": 0.6,
        "verbosity": 0.7,
        "assertiveness": 0.9,
        "art_style": "chalkboard_doodle",
        "composition_style": "exploratory",
        "rhythm_section": "bebop",
    },
    "noir_detective": {
        "name": "Noir Detective",
        "pitch_baseline": 110.0,
        "pitch_variability": 0.2,
        "speed": 0.7,
        "pause_length": 1.8,
        "pause_variability": 0.6,
        "conversational_bpm": 35.0,
        "swing": 0.1,
        "anticipation": 0.4,
        "call_response": 1.2,
        "formality": 0.6,
        "enthusiasm": 0.1,
        "humor_frequency": 0.1,
        "technical_density": 0.2,
        "verbosity": 0.5,
        "assertiveness": 0.4,
        "art_style": "film_noir",
        "composition_style": "monologue",
        "rhythm_section": "walking_bass",
    },
    "bard": {
        "name": "The Bard",
        "pitch_baseline": 140.0,
        "pitch_variability": 0.9,
        "speed": 0.8,
        "pause_length": 1.3,
        "pause_variability": 0.7,
        "conversational_bpm": 50.0,
        "swing": 0.2,
        "anticipation": 0.5,
        "call_response": 0.6,
        "formality": 0.8,
        "enthusiasm": 0.6,
        "humor_frequency": 0.2,
        "technical_density": 0.3,
        "verbosity": 0.9,
        "assertiveness": 0.5,
        "art_style": "illuminated_manuscript",
        "composition_style": "verse_chorus",
        "rhythm_section": "lute_strum",
    },
    "mickey": {
        "name": "Mickey Mouse (Storyteller)",
        "pitch_baseline": 200.0,
        "pitch_variability": 0.95,
        "speed": 1.3,
        "pause_length": 0.4,
        "pause_variability": 0.8,
        "conversational_bpm": 100.0,
        "swing": 0.3,
        "anticipation": 0.3,
        "call_response": 0.5,
        "formality": 0.0,
        "enthusiasm": 1.0,
        "humor_frequency": 0.6,
        "technical_density": 0.0,
        "verbosity": 0.4,
        "assertiveness": 0.7,
        "art_style": "rubber_hose_animation",
        "composition_style": "gag_sequence",
        "rhythm_section": "ragtime",
    },
}

# Where character files are stored
CHARACTER_DIR = Path(os.environ.get("PERSONA_CHARACTER_DIR", str(Path.cwd() / "characters")))


def main():
    parser = argparse.ArgumentParser(
        description="Vibe-Coding Character Builder — dial in a persona",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m persona_engine.cli.character create jazz_vocalist
  python -m persona_engine.cli.character list --presets
  python -m persona_engine.cli.character dial feynman --speed 1.5 --swing 0.2
  python -m persona_engine.cli.character render scientist "The conservation theorem states" --output feynman.wav
  python -m persona_engine.cli.character decompose podcast.mp3 --speaker "Miles Davis"
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    create_p = sub.add_parser("create", help="Create a character from preset or scratch")
    create_p.add_argument("name", help="Character name or preset name")
    create_p.add_argument("--from", dest="from_preset", help="Base preset", default=None)
    create_p.add_argument("--file", help="Output file", default=None)

    # list
    list_p = sub.add_parser("list", help="List characters or presets")
    list_p.add_argument("--presets", action="store_true", help="Show built-in presets")
    list_p.add_argument("--detail", type=str, help="Show details for a specific character")

    # dial
    dial_p = sub.add_parser("dial", help="Adjust character parameters")
    dial_p.add_argument("name", help="Character name")
    dial_p.add_argument("--speed", type=float)
    dial_p.add_argument("--swing", type=float)
    dial_p.add_argument("--pitch", type=float, dest="pitch_baseline")
    dial_p.add_argument("--bpm", type=float, dest="conversational_bpm")
    dial_p.add_argument("--enthusiasm", type=float)
    dial_p.add_argument("--formality", type=float)
    dial_p.add_argument("--technical", type=float, dest="technical_density")
    dial_p.add_argument("--verbosity", type=float)
    dial_p.add_argument("--pause", type=float, dest="pause_length")
    dial_p.add_argument("--anticipation", type=float)
    dial_p.add_argument("--art", type=str, dest="art_style")

    # render
    render_p = sub.add_parser("render", help="Render content through a character voice")
    render_p.add_argument("name", help="Character name")
    render_p.add_argument("text", help="Content to render")
    render_p.add_argument("--output", default="output.wav", help="Output audio file")

    # decompose
    decomp_p = sub.add_parser("decompose", help="Extract persona from audio")
    decomp_p.add_argument("audio", help="Audio file path")
    decomp_p.add_argument("--speaker", default="unknown", help="Speaker name")
    decomp_p.add_argument("--output", help="Persona output file")

    args = parser.parse_args()
    CHARACTER_DIR.mkdir(parents=True, exist_ok=True)

    if args.command == "create":
        cmd_create(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "dial":
        cmd_dial(args)
    elif args.command == "render":
        cmd_render(args)
    elif args.command == "decompose":
        cmd_decompose(args)


def cmd_create(args: argparse.Namespace):
    """Create a new character."""
    name = args.name

    # Check if it's a preset
    if name.lower() in CHARACTER_PRESETS:
        preset = CHARACTER_PRESETS[name.lower()]
        params = CharacterParameters(**preset)
        print(f"Created '{preset['name']}' from preset '{name}'")
    elif args.from_preset and args.from_preset.lower() in CHARACTER_PRESETS:
        preset = CHARACTER_PRESETS[args.from_preset.lower()]
        params = CharacterParameters(**{**preset, "name": name})
        print(f"Created '{name}' from preset '{args.from_preset}'")
    else:
        params = CharacterParameters(name=name)
        print(f"Created new character '{name}' with default parameters")

    _save_character(params)
    _print_character(params)


def cmd_list(args: argparse.Namespace):
    """List available characters and presets."""
    if args.presets:
        print("\n=== Built-in Presets ===\n")
        for name, preset in CHARACTER_PRESETS.items():
            print(f"  {name:20s} — {preset['name']}")
        print()

    if args.detail:
        char_file = CHARACTER_DIR / f"{args.detail}.json"
        if char_file.exists():
            params = CharacterParameters(**json.loads(char_file.read_text()))
            _print_character(params)
        elif args.detail.lower() in CHARACTER_PRESETS:
            params = CharacterParameters(**CHARACTER_PRESETS[args.detail.lower()])
            _print_character(params)
        else:
            print(f"Character '{args.detail}' not found")
        return

    files = list(CHARACTER_DIR.glob("*.json"))
    if files:
        print(f"\n=== Saved Characters ({len(files)} ===\n")
        for f in files:
            params = CharacterParameters(**json.loads(f.read_text()))
            print(f"  {f.stem:20s} → {params.name}")
        print()
    else:
        print("No saved characters. Use `create` to make one.\n")


def cmd_dial(args: argparse.Namespace):
    """Adjust character parameters — the vibe-coding interface."""
    char_file = _resolve_char_file(args.name)

    if not char_file.exists():
        print(f"Character '{args.name}' not found. Create it first.")
        return

    params = CharacterParameters(**json.loads(char_file.read_text()))

    # Apply any dial adjustments
    for key, val in vars(args).items():
        if val is not None and hasattr(params, key):
            old = getattr(params, key)
            setattr(params, key, val)
            if old != val:
                print(f"  {key}: {old} → {val}")

    _save_character(params)
    _print_summary(params)


def _resolve_char_file(name: str) -> Path:
    """Resolve a character file by name (case-insensitive)."""
    name_lower = name.lower().replace(' ', '_')
    char_file = CHARACTER_DIR / f"{name_lower}.json"
    if char_file.exists():
        return char_file
    for f in CHARACTER_DIR.glob("*.json"):
        if f.stem.lower() == name_lower:
            return f
    return char_file  # return the expected path even if missing (caller handles)


def cmd_render(args: argparse.Namespace):
    """Render content through a character's voice."""
    char_file = _resolve_char_file(args.name)
    if not char_file.exists():
        print(f"Character '{args.name}' not found.")
        return

    params = CharacterParameters(**json.loads(char_file.read_text()))
    persona = params.to_persona()

    from persona_engine.compose.engine import CompositionEngine

    engine = CompositionEngine()
    result = engine.compose(args.text, persona, output_path=args.output)

    print(f"\nRendered through '{persona.name}':")
    print(f"  Audio:     {result['audio_path']}")
    print(f"  Duration:  {result['duration']:.1f}s")
    print(f"  Personality at play:")
    print(f"    Rate:    {persona.cadence.mean_wpm:.0f} wpm")
    print(f"    Pauses:  {persona.cadence.mean_pause_duration:.1f}s avg")
    print(f"    Groove:  {persona.groove.conversational_bpm:.0f} bpm, "
          f"swing {persona.groove.swing_factor:.2f}")
    print(f"    Voice:   {persona.prosody.mean_f0:.0f} Hz "
          f"(±{persona.prosody.f0_std:.0f})")


def cmd_decompose(args: argparse.Namespace):
    """Extract personality from source audio."""
    from persona_engine.decompose.pipeline import DecompositionPipeline

    import asyncio
    pipeline = DecompositionPipeline()
    persona = asyncio.run(pipeline.decompose(args.audio, speaker=args.speaker))

    default_path = None
    if args.output:
        pipeline.store(persona, store_path=args.output)
        default_path = args.output
    else:
        default_path = pipeline.store(persona)

    print(f"\nDecomposed '{args.speaker}' from {args.audio}:")
    print(f"  Confidence:     {persona.confidence:.0%}")
    print(f"  Duration:       {persona.source_duration_seconds:.0f}s")
    print(f"  Speaking rate:  {persona.cadence.mean_wpm:.0f} wpm")
    print(f"  Pause duration: {persona.cadence.mean_pause_duration:.1f}s avg")
    print(f"  Thought length: {persona.cadence.thought_duration_mean:.1f}s avg")
    print(f"  Voice:          {persona.prosody.mean_f0:.0f} Hz")
    print(f"  BPM (groove):   {persona.groove.conversational_bpm:.0f}")
    print(f"  Stored to:      {default_path}")
    print(f"\n  To create a dialable character from this persona:")
    print(f"    python -m persona_engine.cli.character dial {args.speaker} --speed X")


def _save_character(params: CharacterParameters):
    """Save character to JSON file."""
    char_file = CHARACTER_DIR / f"{params.name.lower().replace(' ', '_')}.json"
    char_file.write_text(params.model_dump_json(indent=2))
    print(f"  Saved to {char_file}")


def _print_character(params: CharacterParameters):
    """Pretty-print full character."""
    print(f"\n{'='*60}")
    print(f"  {params.name}")
    print(f"{'='*60}")
    print(f"  Voice:     {params.pitch_baseline:.0f}Hz base, "
          f"{params.pitch_variability:.0%} variability")
    print(f"  Speed:     {params.speed:.1f}x")
    print(f"  Pauses:    {params.pause_length:.1f}s avg, "
          f"{params.pause_variability:.0%} variability")
    print(f"  Groove:    {params.conversational_bpm:.0f} bpm, "
          f"{params.swing:.0%} swing")
    print(f"  Anticipate:{params.anticipation:.1f}s window")
    print(f"  Persona:   formality={params.formality:.0%}, "
          f"enthusiasm={params.enthusiasm:.0%}, "
          f"humor={params.humor_frequency:.0%}")
    print(f"  Intellect: technical={params.technical_density:.0%}, "
          f"verbosity={params.verbosity:.0%}, "
          f"assertiveness={params.assertiveness:.0%}")
    print(f"  Art:       {params.art_style}, {params.composition_style}, "
          f"{params.rhythm_section}")
    print()


def _print_summary(params: CharacterParameters):
    """Print a quick summary after dialing."""
    print(f"\n  '{params.name}' — "
          f"speaks at {params.speed:.1f}x "
          f"with {params.swing:.0%} swing, "
          f"{params.enthusiasm:.0%} enthusiasm")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
