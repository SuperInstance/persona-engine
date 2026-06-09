#!/usr/bin/env python3
"""
Scientific Telephone — Casey's live demo vision.

Load a scientist's voice from podcast audio → decompose their persona →
search the ternary research paper → answer IN THEIR VOICE.

The presentation feels like you're ON THE PHONE with that scientist,
not listening to a recording. Phone-bandpassed audio, natural turn delays,
groove-aware conversational timing, and persona-adapted text.

Modes:
    --build-profile podcast.mp3 --speaker "Feynman"
        Decompose audio → extract full persona → store for Q&A

    --ask "What is the conservation theorem?"
        Ask about the paper → find best section → render in scientist's voice

    --interactive
        Back-and-forth: you ask, they answer, phone-call feel

Run:
    python examples/scientific_telephone.py --build-profile podcast.mp3 --speaker "Feynman"
    python examples/scientific_telephone.py --ask "What is avoidance dominance?"
    python examples/scientific_telephone.py --interactive
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persona_engine.schemas.persona import (
    Persona,
    PersonaManifest,
    CharacterParameters,
    TurnStyle,
    SpeakingRate,
)
from persona_engine.decompose.pipeline import DecompositionPipeline
from persona_engine.compose.engine import CompositionEngine
from persona_engine.groove.engine import (
    GrooveEngine,
    ConversationGrooveMapper,
    TickType,
)

# ---------------------------------------------------------------------------
# Corpus — SCIENCE-PAPER.md
# ---------------------------------------------------------------------------

SCIENCE_PAPER_PATH = Path(__file__).resolve().parent.parent.parent / "construct-coordination" / "SCIENCE-PAPER.md"


@dataclass
class CorpusChunk:
    """A chunk of the paper with metadata for keyword retrieval."""
    title: str
    section: str  # e.g. "abstract", "law_1", "law_2", "conclusion"
    text: str
    keywords: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.keywords:
            self.keywords = self._extract_keywords()

    def _extract_keywords(self) -> List[str]:
        """Extract important keywords from the text."""
        words = self.text.lower().split()
        # Technical terms + capitalized multi-word phrases + frequent nouns
        technical = {
            "ternary", "binary", "avoidance", "selection", "conservation",
            "population", "agent", "species", "fitness", "entropy",
            "feedback", "negative", "positive", "lotka", "volterra",
            "ecosystem", "resilience", "convergent", "evolution",
            "proof", "theorem", "lemma", "corollary", "law",
            "ratio", "scale", "invariant", "symmetry", "noether",
            "compiled", "policy", "esp32", "microcontroller", "nanosecond",
            "bare", "metal", "edge", "ai", "embedded", "inference",
            "explorer", "diplomat", "marksman", "climber", "prospector",
            "niche", "win", "rate", "signature", "negative-space",
            "decision", "alpha", "beta", "gamma", "delta", "epsilon",
            "cuda", "gpu", "rtx", "wasm", "cross-language",
            "blake2b", "python", "rust", "binding",
            "intelligence", "learning", "reinforcement",
        }
        return list({w for w in words if w in technical or len(w) > 7} | {w.strip("():,.") for w in self.text.split() if w.strip("():,.").isupper() and len(w) > 3})


def load_corpus(path: Optional[Path] = None) -> List[CorpusChunk]:
    """
    Load and chunk SCIENCE-PAPER.md into searchable sections.

    Returns chunks for: abstract, each law, strategy species table,
    scaling section, cross-language validation, bare-metal proof,
    discussion/conclusion.
    """
    if path is None:
        path = SCIENCE_PAPER_PATH

    if not path.exists():
        print(f"  ⚠ Paper not found at {path}")
        return []

    text = path.read_text(encoding="utf-8")
    chunks = []

    # --- Abstract ---
    m = re.search(r"## Abstract\s*\n(.*?)(?=\n## \d)", text, re.DOTALL)
    if m:
        chunks.append(CorpusChunk("Abstract", "abstract", m.group(1).strip()))

    # --- Introduction ---
    m = re.search(r"## 1\. Introduction.*?\n(.*?)(?=\n## 2\.)", text, re.DOTALL)
    if m:
        intro_text = re.sub(r"^## .*$", "", m.group(0), flags=re.MULTILINE).strip()
        chunks.append(CorpusChunk("Introduction — Why Ternary", "introduction", intro_text))

    # --- The Five Laws (section 2) ---
    law_patterns = [
        ("Law 1: Negative Space Discovery", r"### Law 1:.*?(?=### Law 2:)"),
        ("Law 2: Avoidance Dominance", r"### Law 2:.*?(?=### Law 3:)"),
        ("Law 3: Strategy Species Coexistence", r"### Law 3:.*?(?=### Law 4:)"),
        ("Law 4: Population > Individual", r"### Law 4:.*?(?=### Law 5:)"),
        ("Law 5: Conservation of the Avoidance Ratio", r"### Law 5:.*?(?=\n## 3\.)"),
    ]
    for i, (title, pattern) in enumerate(law_patterns):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            cleaned = re.sub(r"^### .*$", "", m.group(0), flags=re.MULTILINE).strip()
            # Include formal statement + experimental evidence
            law_text = cleaned
            chunks.append(CorpusChunk(title, f"law_{i+1}", law_text))

    # --- Strategy Species (section 3) ---
    m = re.search(r"## 3\. The Five Universal Strategy Species\n(.*?)(?=\n## 4\.)", text, re.DOTALL)
    if m:
        chunks.append(CorpusChunk(
            "The Five Universal Strategy Species",
            "strategy_species",
            m.group(1).strip(),
        ))

    # --- Scaling (section 4) ---
    m = re.search(r"## 4\. Scaling Behavior\n(.*?)(?=\n## 5\.)", text, re.DOTALL)
    if m:
        chunks.append(CorpusChunk(
            "Scaling Behavior",
            "scaling",
            m.group(1).strip(),
        ))

    # --- Cross-Language Validation (section 5) ---
    m = re.search(r"## 5\. Cross-Language Validation\n(.*?)(?=\n## 6\.)", text, re.DOTALL)
    if m:
        chunks.append(CorpusChunk(
            "Cross-Language Validation",
            "cross_language",
            m.group(1).strip(),
        ))

    # --- Bare Metal (section 6) ---
    m = re.search(r"## 6\. Bare Metal Proof.*?\n(.*?)(?=\n## 7\.)", text, re.DOTALL)
    if m:
        chunks.append(CorpusChunk(
            "Bare Metal Proof — 279 Bytes, 8 Nanoseconds",
            "bare_metal",
            m.group(1).strip(),
        ))

    # --- Discussion / Conclusion (section 7) ---
    m = re.search(r"## 7\. Discussion\n(.*?)(?=\n## References)", text, re.DOTALL)
    if m:
        discussion_text = m.group(1).strip()
        # Split into sub-sections
        sub_sections = re.split(r"\n### ", discussion_text)
        for i, sub in enumerate(sub_sections):
            title_line = sub.split("\n")[0].strip("# ")
            body_lines = sub.split("\n")[1:]
            body = "\n".join(line for line in body_lines if line.strip())
            if body.strip():
                chunks.append(CorpusChunk(
                    title_line or f"Discussion Part {i+1}",
                    f"discussion_{i+1}",
                    body.strip(),
                ))

    return chunks


# ---------------------------------------------------------------------------
# Corpus Search
# ---------------------------------------------------------------------------

def search_corpus(question: str, chunks: List[CorpusChunk]) -> Tuple[CorpusChunk, float]:
    """
    Find the most relevant chunk for a question using keyword overlap scoring.

    Returns (best_chunk, score).
    """
    q_words = set(question.lower().split())
    q_terms = set()

    # Extract meaningful terms (nouns, compounds, technical words)
    for w in q_words:
        w_clean = w.strip("?:!.,;'\"")
        if len(w_clean) > 2:
            q_terms.add(w_clean)

    # Also extract bi-grams
    words_list = [w.strip("?:!.,;'\"") for w in q_words if len(w.strip("?:!.,;'\"")) > 2]
    for i in range(len(words_list) - 1):
        q_terms.add(f"{words_list[i]}_{words_list[i+1]}")

    best_score = -1.0
    best_chunk = chunks[0] if chunks else None

    for chunk in chunks:
        # Keyword overlap score
        chunk_keywords = set(chunk.keywords)
        common = q_terms & chunk_keywords
        overlap_score = len(common) / max(len(q_terms), 1)

        # Title match bonus (section title contains question terms)
        title_words = set(chunk.title.lower().split())
        title_common = q_terms & title_words
        title_bonus = len(title_common) * 0.3

        # Section type heuristic bonuses
        section_bonus = 0.0
        q_lower = question.lower()
        if "law" in q_lower and chunk.section.startswith("law_"):
            section_bonus = 0.2
        if "species" in q_lower and chunk.section == "strategy_species":
            section_bonus = 0.2
        if "scale" in q_lower and chunk.section == "scaling":
            section_bonus = 0.2
        if "esp32" in q_lower or "microcontroller" in q_lower or "bare" in q_lower or "279" in q_lower:
            if chunk.section == "bare_metal":
                section_bonus = 0.3
        if "abstract" in q_lower and chunk.section == "abstract":
            section_bonus = 0.2
        if "conclusion" in q_lower or "discussion" in q_lower or "implication" in q_lower:
            if chunk.section.startswith("discussion_"):
                section_bonus = 0.2

        score = overlap_score + title_bonus + section_bonus

        # Text-length normalization: prefer chunks that aren't too short
        if len(chunk.text) < 50:
            score -= 0.3

        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk, best_score


# ---------------------------------------------------------------------------
# Answer Synthesis — persona-adapted text from corpus
# ---------------------------------------------------------------------------

def adapt_text_for_persona(text: str, persona: Persona) -> str:
    """
    Adapt corpus text using the persona's speaking patterns.

    For fast speakers: shorter sentences, more punchy
    For pensive types: more pauses, rhetorical questions
    For rhythmic styles: repeating structures
    For technical types: preserve jargon, add framing
    """
    cadence = persona.cadence
    lexical = persona.lexical

    # Clean the text: remove markdown formatting
    clean = text
    clean = re.sub(r"\*\*(.*?)\*\*", r"\1", clean)
    clean = re.sub(r"\*(.*?)\*", r"\1", clean)
    clean = re.sub(r"`(.*?)`", r"\1", clean)
    clean = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", clean)
    clean = re.sub(r"\|.*?\|", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"[-–—]\s*", ", ", clean)
    clean = re.sub(r"\n+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    # Remove table rows
    clean = re.sub(r"\s{2,}", " ", clean)

    # Split into sentences
    raw_sentences = re.split(r"(?<=[.!?])\s+", clean)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]

    if not sentences:
        return clean

    # Adapt sentence length based on speaking speed
    is_fast = cadence.mean_wpm > 160
    is_slow = cadence.mean_wpm < 130
    is_pensive = cadence.mean_pause_duration > 0.8
    is_rhythmic = cadence.turn_style == TurnStyle.RHYTHMIC
    is_exploratory = cadence.turn_style == TurnStyle.EXPLORATORY

    adapted = []

    # Opening phrase based on persona
    if cadence.turn_style == TurnStyle.PATIENT or is_pensive:
        adapted.append("Let me think about that for a moment.")
    elif is_fast:
        adapted.append("Great question — let me walk you through it.")
    elif is_exploratory:
        adapted.append("Well, you know, that's really interesting when you look at it closely.")
    else:
        adapted.append("Let me explain.")

    for i, sent in enumerate(sentences[:5]):  # Use up to 5 sentences
        words = sent.split()
        word_count = len(words)

        # Fast speakers: break long sentences into shorter chunks
        if is_fast and word_count > 15:
            mid = word_count // 2
            first_half = " ".join(words[:mid])
            second_half = " ".join(words[mid:])
            adapted.append(first_half + ",")
            adapted.append(second_half.rstrip(".,") + ".")
        elif is_slow and word_count < 8:
            # Slow speakers: add explanatory framing to short sentences
            adapted.append(sent)
            adapted.append("And the reason for that is worth understanding.")
        else:
            adapted.append(sent)

        # Add persona-appropriate connectors
        if i < len(sentences) - 1:
            if is_pensive:
                # Pensive speakers: long pauses, reflective connectors
                if i % 2 == 0:
                    adapted.append("Actually, let me rephrase that.")
                elif i % 2 == 1:
                    adapted.append("Now — and this is the key point —")
            elif is_rhythmic:
                # Rhythmic speakers: parallel structures
                if i % 2 == 0:
                    adapted.append("But here's the thing:")
                elif i % 2 == 1:
                    adapted.append("And this matters because:")
            elif is_exploratory:
                # Exploratory speakers: trailing thoughts
                if i % 2 == 0:
                    adapted.append("Or at least, that's one way to think about it.")
                elif i % 2 == 1:
                    adapted.append("But let me back up a bit.")

    # Closing based on persona
    if is_pensive:
        adapted.append("Does that answer your question?")
    elif is_fast:
        adapted.append("So that's the short version. Happy to go deeper if you want.")
    elif is_rhythmic:
        adapted.append("And that's the beat of it. Clear as a bell.")
    elif is_exploratory:
        adapted.append("Anyway, that's how I see it.")
    else:
        adapted.append("That's the essence of it.")

    return " ".join(adapted)


# ---------------------------------------------------------------------------
# Phone-call audio processing
# ---------------------------------------------------------------------------

def apply_phone_effect(input_path: str, output_path: str) -> str:
    """
    Apply a telephone-bandpass effect to make speech sound like a phone call.

    Simulates:
    - 300-3400Hz bandpass (POTS telephone bandwidth)
    - Slight compression (phone dynamics)
    - 8kHz sample rate (phone line sample rate)
    - A little noise for authenticity
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af",
            "bandpass=frequency=1200:width_type=q:width=1.5,"
            "equalizer=f=300:t=q:w=1:g=-3,"
            "equalizer=f=3400:t=q:w=1:g=-3,"
            "compand=0.1:0.3:-90/-90|-30/-20|-10/-5|0/0:6:0:0:0",
            "-ar", "8000",
            "-ac", "1",
            "-b:a", "32k",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        pass

    # Fallback: just copy
    if os.path.exists(input_path) and input_path != output_path:
        import shutil
        shutil.copy2(input_path, output_path)
    return output_path


def ensure_output_dir(path: str) -> str:
    """Ensure directory exists for output path and return the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Persona loading / storage helpers
# ---------------------------------------------------------------------------

MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory"


def load_latest_persona() -> Optional[Persona]:
    """Load the most recently decomposed persona."""
    persona_files = sorted(MEMORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not persona_files:
        print("  ⚠ No decomposed personas found. Run --build-profile first.")
        return None
    latest = persona_files[-1]
    return Persona(**json.loads(latest.read_text()))


# ---------------------------------------------------------------------------
# Mode 1: Build Profile
# ---------------------------------------------------------------------------

def build_profile(args):
    """Decompose audio → extract full persona."""
    print(f"\n{'='*60}")
    print("  📞 SCIENTIFIC TELEPHONE — BUILD PROFILE")
    print(f"{'='*60}")
    print(f"  Source:     {args.build_profile}")
    print(f"  Speaker:    {args.speaker}")
    print()

    pipeline = DecompositionPipeline()
    persona = asyncio.run(pipeline.decompose(args.build_profile, speaker=args.speaker))
    path = pipeline.store(persona)

    print(f"  ✅ Persona decomposed for '{args.speaker}':")
    print(f"     ┌─────────────────────────────────────────┐")
    print(f"     │  Speaking rate:  {persona.cadence.mean_wpm:>5.0f} wpm", end="")
    if persona.cadence.mean_wpm > 160:
        print("   🏃 fast talker     │")
    elif persona.cadence.mean_wpm < 130:
        print("   🐢 slow talker     │")
    else:
        print("   → moderate       │")
    print(f"     │  Pause duration: {persona.cadence.mean_pause_duration:>5.1f}s avg", end="")
    if persona.cadence.mean_pause_duration > 0.8:
        print("  🧘 pensive        │")
    else:
        print("  → crisp          │")
    print(f"     │  Thought length: {persona.cadence.thought_duration_mean:>5.1f}s avg   │")
    print(f"     │  Voice pitch:    {persona.prosody.mean_f0:>5.0f} Hz", end="")
    print(f" (±{persona.prosody.f0_std:.0f})           │")
    print(f"     │  Groove BPM:     {persona.groove.conversational_bpm:>5.0f} bpm", end="")
    if persona.groove.swing_factor > 0.15:
        print(f"  🎵 swing:{persona.groove.swing_factor:.0%}  │")
    else:
        print("  → straight      │")
    print(f"     │  Turn style:     {persona.cadence.turn_style.value:>18s}  │")
    print(f"     └─────────────────────────────────────────┘")
    print(f"     Confidence:     {persona.confidence:.0%}")
    print(f"     Stored to:      {path}")

    # Save dialable CharacterParameters for re-use
    char_params = CharacterParameters(
        name=args.speaker,
        speed=persona.cadence.mean_wpm / 150.0,
        pause_length=persona.cadence.mean_pause_duration,
        thought_length=persona.cadence.thought_duration_mean,
        conversational_bpm=persona.groove.conversational_bpm,
        swing=persona.groove.swing_factor,
        anticipation=persona.groove.anticipation_window,
        pitch_baseline=persona.prosody.mean_f0,
        pitch_variability=(persona.prosody.f0_std / 50.0) if persona.prosody.f0_std else 0.5,
        technical_density=persona.lexical.technical_density if persona.lexical else 0.5,
    )
    char_dir = Path.cwd() / "characters"
    char_dir.mkdir(exist_ok=True)
    char_path = char_dir / f"{args.speaker.lower().replace(' ', '_')}.json"
    char_path.write_text(char_params.model_dump_json(indent=2))
    print(f"  🎭 Character profile: {char_path}")
    print()

    return persona


# ---------------------------------------------------------------------------
# Mode 2: Ask a Question
# ---------------------------------------------------------------------------

def ask_question(args, persona: Optional[Persona] = None, corpus: Optional[List[CorpusChunk]] = None):
    """Ask a question → search corpus → adapt with persona → render."""
    if persona is None:
        persona = load_latest_persona()
        if persona is None:
            return

    if corpus is None:
        corpus = load_corpus()
        if not corpus:
            print("  ⚠ No corpus available.")
            return

    print(f"\n{'='*60}")
    print(f"  📞 SCIENTIFIC TELEPHONE — ASK")
    print(f"{'='*60}")
    print(f"  On the line: {persona.name}")
    print(f"  You asked:   {args.ask}")
    print(f"  {'─'*60}")

    # 1. Search corpus
    best_chunk, score = search_corpus(args.ask, corpus)
    if not best_chunk:
        print("  ⚠ No relevant section found.")
        return

    print(f"  📖 Found in:  {best_chunk.title}  (relevance: {score:.0%})")
    print()

    # 2. Adapt text through persona
    answer = adapt_text_for_persona(best_chunk.text, persona)

    # 3. Set up groove for conversational timing
    bpm, swing, anticipation_ms = ConversationGrooveMapper.persona_to_groove(persona.cadence)
    groove = GrooveEngine(bpm=bpm, swing=swing, anticipation_ms=anticipation_ms)
    groove.start_turn(speaker=persona.name, anticipated_delay=0.4)

    # 4. Compose through the engine
    engine = CompositionEngine()
    phone_name = f"{persona.name.lower().replace(' ', '_')}_answer_{hash(args.ask) & 0xFFFF}.wav"
    phone_path = ensure_output_dir(str(Path.cwd() / "output" / phone_name))

    result = engine.compose(answer, persona, output_path=phone_path)

    # 5. Apply phone effect
    phone_output = phone_path.replace(".wav", "_phone.wav")
    apply_phone_effect(result["audio_path"], phone_output)

    # 6. Show the answer
    print(f"  📞 {persona.name} answers (phone-filtered):")
    print(f"  {'─'*60}")
    print(f"  {answer}")
    print(f"  {'─'*60}")
    print()
    print(f"     Groove:     {bpm:.0f} bpm, {swing:.0%} swing")
    print(f"     Audio:      {phone_output}")
    print(f"     Duration:   {result.get('duration', 0):.1f}s")
    print()

    return result


# ---------------------------------------------------------------------------
# Mode 3: Interactive — Phone Call
# ---------------------------------------------------------------------------

def interactive_mode(args):
    """Full interactive Q&A — feels like a phone call."""
    print(f"\n{'='*60}")
    print("  📞 SCIENTIFIC TELEPHONE — INTERACTIVE MODE")
    print(f"{'='*60}")
    print("  You're on the phone with a scientist.")
    print("  Ask about the ternary research paper.")
    print("  They'll answer in their own voice.")
    print()
    print("  Commands:")
    print("    quit / exit / q   — Hang up")
    print("    help / ?          — Show commands")
    print("    who               — Show current persona")
    print("    source            — Show corpus details")
    print()

    # Load persona
    persona = load_latest_persona()
    if persona is None:
        return

    # Load corpus
    corpus = load_corpus()
    if not corpus:
        return

    # Set up groove
    bpm, swing, anticipation_ms = ConversationGrooveMapper.persona_to_groove(persona.cadence)
    groove = GrooveEngine(bpm=bpm, swing=swing, anticipation_ms=anticipation_ms)
    engine = CompositionEngine()

    # Call greeting
    greeting_map = {
        TurnStyle.PATIENT: "Hello. I'm glad you called. What would you like to discuss?",
        TurnStyle.INTERRUPTIBLE: "Hey — yeah, I'm here. Go ahead, ask me anything.",
        TurnStyle.RHYTHMIC: "Hello there. I was just thinking about your call. Fire away.",
        TurnStyle.EXPLORATORY: "Oh hey! You know, I was just reading about this paper. What's on your mind?",
    }
    greeting = greeting_map.get(persona.cadence.turn_style, "Hello. What would you like to ask?")
    greeting_groove = engine.compose_interactive(greeting, persona)

    print(f"  📞 Ring ring...")
    print(f"  ─────────────────────────────────────────────")
    print(f"  {persona.name} > {greeting}")
    print(f"  ─────────────────────────────────────────────")
    print()

    turn_count = 0
    context = []  # conversation context

    while True:
        try:
            user_input = input("  You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  📞 ...click.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() in ("help", "?"):
            print("  Commands: quit, who, source, or just ask anything about the paper.")
            continue
        if user_input.lower() == "who":
            print(f"  You're talking to {persona.name}.")
            print(f"  Style: {persona.cadence.turn_style.value}, "
                  f"{persona.cadence.mean_wpm:.0f} wpm, "
                  f"groove at {persona.groove.conversational_bpm:.0f} bpm")
            continue
        if user_input.lower() == "source":
            print(f"  Paper: Intelligence as Negative Space")
            print(f"  Sections loaded: {len(corpus)}")
            for c in corpus:
                print(f"    ─ {c.title}  ({len(c.text)} chars, {len(c.keywords)} keywords)")
            continue

        turn_count += 1
        context.append(f"User: {user_input}")

        # Search corpus
        best_chunk, score = search_corpus(user_input, corpus)
        if not best_chunk:
            answer = "I'm not sure I have anything relevant on that from the paper. Could you rephrase?"
        else:
            answer = adapt_text_for_persona(best_chunk.text, persona)

        context.append(f"{persona.name}: {answer[:100]}...")

        # Conversational timing — phone-call feel
        turn_delay = groove.start_turn(
            speaker=persona.name,
            anticipated_delay=persona.groove.anticipation_window,
        )

        phone_name = f"interactive_turn_{turn_count}.wav"
        phone_path = ensure_output_dir(str(Path.cwd() / "output" / phone_name))

        result = engine.compose_interactive(
            answer, persona,
            context=user_input,
            is_turn_end=True,
        )

        # Apply phone effect
        phone_output = phone_path.replace(".wav", "_phone.wav")
        apply_phone_effect(result["audio_path"], phone_output)

        duration = result.get("duration", 0)

        # Display with conversation flow
        print(f"  ─────────────────────────────────────────────")
        print(f"  📖 ({best_chunk.title if best_chunk else 'conversation'})")
        print(f"  {persona.name} > {answer}")
        print(f"  ─────────────────────────────────────────────")
        print(f"     (took {duration:.1f}s to say — "
              f"{'like a quick remark' if duration < 3 else 'a considered response'})")

        # Natural conversation rhythm: breathe after every 3 turns
        if turn_count % 3 == 0:
            groove.breathe()
            print(f"     ({persona.name} pauses, you hear a breath...)")

        # After 5 turns, the scientist gets more relaxed
        if turn_count == 5:
            groove.modulate_bpm(bpm * 0.9, ramp_seconds=1.0)
            print(f"     (the conversation settles into a natural rhythm...)")

        print()

    # Farewell
    farewell_map = {
        TurnStyle.PATIENT: "Thank you for the thoughtful questions. Goodbye.",
        TurnStyle.INTERRUPTIBLE: "Great talking. Call anytime, seriously.",
        TurnStyle.RHYTHMIC: "Always a pleasure. Until next time.",
        TurnStyle.EXPLORATORY: "Oh — we're done already? Well, this was fun. Talk soon!",
    }
    farewell = farewell_map.get(
        persona.cadence.turn_style,
        "Goodbye. It was a pleasure discussing this with you.",
    )
    print(f"  📞 {persona.name} > {farewell}")
    print(f"  ...click.")
    print(f"\n  Call summary: {turn_count} turns with {persona.name}.")
    print(f"  Audio saved to: output/")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="📞 Scientific Telephone — Live Q&A through a scientist's voice persona",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --build-profile audio.mp3 --speaker "Name"    Decompose a voice
  --ask "your question"                           Ask about the paper
  --interactive                                   Full phone-call Q&A

Examples:
  python examples/scientific_telephone.py --build-profile feynman.mp3 --speaker "Feynman"
  python examples/scientific_telephone.py --ask "What is avoidance dominance?"
  python examples/scientific_telephone.py --interactive
        """,
    )
    parser.add_argument(
        "--build-profile",
        help="Audio file (MP3/WAV) to decompose into a persona",
    )
    parser.add_argument(
        "--speaker",
        default="Scientist",
        help="Name of the speaker (default: Scientist)",
    )
    parser.add_argument(
        "--ask",
        help="Ask a question about the ternary paper through the persona",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive phone-call mode with persona-aware timing",
    )
    args = parser.parse_args()

    # Validate
    if args.build_profile and not os.path.exists(args.build_profile):
        print(f"  ❌ Audio file not found: {args.build_profile}")
        sys.exit(1)
    if args.build_profile and not args.speaker:
        print("  ❌ --speaker required with --build-profile")
        sys.exit(1)

    # Check paper exists
    if args.ask or args.interactive:
        if not SCIENCE_PAPER_PATH.exists():
            print(f"  ⚠ Paper not found at {SCIENCE_PAPER_PATH}")
            print(f"     Q&A will not have a corpus to search.")
            print(f"     Continue anyway? (y/N) ", end="")
            try:
                resp = input().strip().lower()
                if resp != "y":
                    return
            except EOFError:
                return
        # Pre-load corpus (shared across ask and interactive)
        corpus = load_corpus()
    else:
        corpus = None

    # Track persona between modes
    persona = None

    if args.build_profile:
        persona = build_profile(args)

    if args.ask:
        ask_question(args, persona=persona, corpus=corpus)

    if args.interactive:
        interactive_mode(args)

    if not any([args.build_profile, args.ask, args.interactive]):
        parser.print_help()


if __name__ == "__main__":
    main()
