#!/usr/bin/env python3
"""
Scientist Interactive Podcast Demo — persona-decomposed Q&A.

This is the vision Casey described:
    1. Feed the system podcast audio of a scientist
    2. It extracts their cadence, prosody, groove
    3. You ask questions about papers they never discussed
    4. The system answers *in their voice* — cadence, timing, feel
    5. The latency feels like a phone call, not a robot

Run:
    python examples/scientist_podcast.py --build-profile podcast.mp3 --speaker "Feynman"
    python examples/scientist_podcast.py --ask "Explain the conservation theorem"
    python examples/scientist_podcast.py --interactive
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persona_engine.schemas.persona import Persona, CharacterParameters, PersonaManifest
from persona_engine.decompose.pipeline import DecompositionPipeline
from persona_engine.compose.engine import CompositionEngine
from persona_engine.groove.engine import GrooveEngine, ConversationGrooveMapper, TickType


def build_profile(args):
    """Step 1: Decompose audio → persona vector."""
    print(f"\n{'='*60}")
    print("  STEP 1: PERSONA DECOMPOSITION")
    print(f"{'='*60}")
    print(f"  Source:     {args.build_profile}")
    print(f"  Speaker:    {args.speaker}")
    print()

    pipeline = DecompositionPipeline()
    persona = asyncio.run(pipeline.decompose(args.build_profile, speaker=args.speaker))
    path = pipeline.store(persona)

    print(f"\n  ✅ Decomposed '{args.speaker}':")
    print(f"     Speaking rate:  {persona.cadence.mean_wpm:.0f} wpm")
    print(f"     Pause duration: {persona.cadence.mean_pause_duration:.1f}s avg")
    print(f"     Thought length: {persona.cadence.thought_duration_mean:.1f}s avg")
    print(f"     Voice:          {persona.prosody.mean_f0:.0f} Hz "
          f"(±{persona.prosody.f0_std:.0f})")
    print(f"     BPM (groove):   {persona.groove.conversational_bpm:.0f}")
    print(f"     Swing:          {persona.groove.swing_factor:.2f}")
    print(f"     Confidence:     {persona.confidence:.0%}")
    print(f"     Stored to:      {path}")
    print()

    # Also create a dialable character version
    char_params = CharacterParameters(
        name=args.speaker,
        speed=persona.cadence.mean_wpm / 150.0,
        pause_length=persona.cadence.mean_pause_duration,
        conversational_bpm=persona.groove.conversational_bpm,
        swing=persona.groove.swing_factor,
        anticipation=persona.groove.anticipation_window,
        pitch_baseline=persona.prosody.mean_f0,
        pitch_variability=persona.prosody.f0_std / 50.0,
        technical_density=persona.lexical.technical_density if persona.lexical else 0.5,
    )
    char_dir = Path.cwd() / "characters"
    char_dir.mkdir(exist_ok=True)
    char_path = char_dir / f"{args.speaker.lower().replace(' ', '_')}.json"
    char_path.write_text(char_params.model_dump_json(indent=2))
    print(f"  🎭 Dialable character also saved to: {char_path}")
    print()

    return persona


def ask_question(args, persona: Persona = None):
    """Step 2: Ask a question and get a persona-rendered answer."""
    if persona is None:
        # Load from store
        store_path = Path.cwd() / "memory"
        persona_files = list(store_path.glob("*.json"))
        if not persona_files:
            print("No decomposed personas found. Run --build-profile first.")
            return
        latest = sorted(persona_files, key=lambda p: p.stat().st_mtime)[-1]
        persona = Persona(**json.loads(latest.read_text()))

    print(f"\n{'='*60}")
    print(f"  STEP 2: PERSONA-COMPOSED ANSWER")
    print(f"{'='*60}")
    print(f"  Persona:    {persona.name}")
    print(f"  Question:   {args.ask}")
    print()

    # Generate the answer content
    answer = _generate_answer(args.ask, persona)

    # Set up the groove engine for conversational timing
    bpm, swing, anticipation = ConversationGrooveMapper.persona_to_groove(persona.cadence)
    groove = GrooveEngine(bpm=bpm, swing=swing, anticipation_ms=anticipation)
    groove.start_turn(speaker=persona.name, anticipated_delay=0.4)

    print(f"  🎵 Groove: {bpm:.0f} bpm, {swing:.0%} swing, "
          f"{anticipation:.0f}ms anticipation")
    print()

    # Render through the composition engine
    engine = CompositionEngine()
    result = engine.compose(answer, persona, output_path=f"{args.speaker}_answer.wav")

    print(f"  ✅ Composed audio: {result['audio_path']}")
    print(f"     Duration:       {result['duration']:.1f}s")
    print(f"     SSML:           {result['ssml'][:100]}...")
    print()
    print(f"  Answer (rendered in {persona.name}'s voice):")
    print(f"  {'─'*60}")
    print(f"  {answer}")
    print(f"  {'─'*60}")
    print()

    return result


def interactive_mode(args):
    """Step 3: Interactive back-and-forth with persona-aware timing."""
    print(f"\n{'='*60}")
    print("  STEP 3: INTERACTIVE MODE")
    print("  Type questions. The persona answers in its voice.")
    print("  Type 'quit' to exit.")
    print(f"{'='*60}\n")

    # Load persona
    store_path = Path.cwd() / "memory"
    persona_files = list(store_path.glob("*.json"))
    if not persona_files:
        print("No personas found. Run --build-profile first.")
        return
    latest = sorted(persona_files, key=lambda p: p.stat().st_mtime)[-1]
    persona = Persona(**json.loads(latest.read_text()))

    # Set up groove
    bpm, swing, anticipation = ConversationGrooveMapper.persona_to_groove(persona.cadence)
    groove = GrooveEngine(bpm=bpm, swing=swing, anticipation_ms=anticipation)

    engine = CompositionEngine()
    turn_count = 0

    print(f"  Talking to: {persona.name}")
    print(f"  Feel: {bpm:.0f} bpm, {swing:.0%} swing "
          f"(like {'a relaxed chat' if bpm < 60 else 'an energetic conversation'})")
    print(f"  Anticipation: {anticipation:.0f}ms "
          f"(they{' ' if anticipation > 200 else ' don\'t '}lean into punchlines)")
    print()

    while True:
        user_input = input("  You > ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        turn_count += 1
        answer = _generate_answer(user_input, persona)

        # Conversational timing
        turn_delay = groove.start_turn(
            speaker=persona.name,
            anticipated_delay=persona.groove.anticipation_window,
        )

        result = engine.compose_interactive(
            answer, persona, context=user_input, is_turn_end=True,
        )

        duration = result.get("duration", 0)
        print(f"\n  {persona.name} > {answer}")
        print(f"     (rendered in {duration:.1f}s — "
              f"{'feels like a phone call' if duration > 0.5 else 'quick response'})")

        if turn_count % 3 == 0:
            groove.breathe()
            print("     (they pause to reflect...)")

        print()

    print(f"\n  Conversation ended. {turn_count} turns exchanged.")


def _generate_answer(question: str, persona: Persona) -> str:
    """
    Generate a persona-aware answer.

    In a real implementation, this would:
    1. Search the persona's domain vector DB (papers, source material)
    2. Synthesize a response using the persona's lexical fingerprint
    3. Return text ready for TTS rendering

    For the demo, we generate a stylized answer based on persona type.
    """
    is_scientist = persona.lexical.technical_density > 0.5 if persona.lexical else False
    is_storyteller = not is_scientist

    if is_scientist:
        return (
            f"That's an interesting question. Let me think about it for a moment. "
            f"The key insight is that the conservation theorem isn't just a mathematical constraint — "
            f"it's a statement about what quantities are preserved under transformation. "
            f"If you look at the symmetry group of the system, you'll find that each symmetry "
            f"corresponds to a conserved quantity. This is Noether's theorem, and it's one of the "
            f"most beautiful results in physics. Now, the specific case you're asking about involves "
            f"a broken symmetry, which means the conserved quantity becomes approximate. "
            f"But here's the thing — even approximate conservation tells you something deep about "
            f"the system's dynamics. The timescale over which the quantity changes tells you "
            f"about the coupling strength. So yes, the answer is yes, but the real question is "
            f"how fast and under what conditions."
        )
    elif "story" in question.lower() or "tell" in question.lower():
        return (
            f"Ah, now that's a story worth telling. You see, it started in a small workshop "
            f"tucked away behind the old oak tree. The kind of place where the light filters through "
            f"dusty windows and falls on workbenches covered in half-finished projects. "
            f"The inventor — and I should say, every good story needs an inventor — was working on "
            f"something nobody had ever seen before. They didn't know if it would work. "
            f"They didn't know if it should work. But they knew they had to build it. "
            f"And that, right there, is where every great story begins: with someone who doesn't "
            f"know if they'll succeed, but knows they have to try."
        )
    else:
        return (
            f"Well, let me put it this way. Everything connects to everything else — "
            f"that's the first thing you need to understand. The second thing is that these "
            f"connections aren't random. They form patterns, like the veins in a leaf or "
            f"the branches of a river delta. And once you start seeing those patterns, "
            f"you can't unsee them. They're everywhere. In the way people talk, in the way "
            f"markets move, in the way a jazz band finds the pocket. So when you ask me "
            f"about that specific question, what you're really asking about is which pattern "
            f"it belongs to. And that's the most interesting thing to explore."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Scientist Interactive Podcast Demo — persona-decomposed Q&A"
    )
    parser.add_argument("--build-profile", help="Audio file to decompose into persona")
    parser.add_argument("--speaker", default="Scientist", help="Speaker name")
    parser.add_argument("--ask", help="Ask a question through the persona")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive Q&A with persona-aware timing")
    args = parser.parse_args()

    persona = None
    if args.build_profile:
        persona = build_profile(args)

    if args.ask:
        ask_question(args, persona=persona)

    if args.interactive:
        interactive_mode(args)

    if not any([args.build_profile, args.ask, args.interactive]):
        parser.print_help()


if __name__ == "__main__":
    main()
