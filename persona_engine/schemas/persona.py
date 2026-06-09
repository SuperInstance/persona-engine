"""
Persona Vector Schema — the cognitive fingerprint of a voice.

A persona is a multi-vector embedding that captures not just *what* someone
says, but *how* they say it: cadence, pause allocation, prosody, lexical
patterns, thinking rhythm, and conversational turn-taking style.

This is the lead sheet for a voice. The composition engine reads this,
and renders any content through it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


class SpeakingRate(Enum):
    SLOW = "slow"
    MODERATE = "moderate"
    FAST = "fast"
    VARIABLE = "variable"


class TurnStyle(Enum):
    PATIENT = "patient"        # long pauses, waits for completion
    INTERRUPTIBLE = "interruptible"  # short pauses, allows overlap
    RHYTHMIC = "rhythmic"      # clock-like turn boundaries
    EXPLORATORY = "exploratory"  # filled pauses, trailing off


class CadenceProfile(BaseModel):
    """
    The temporal fingerprint of how someone speaks.

    Captured from source audio via OpenSMILE feature extraction.
    """
    model_config = {"extra": "allow"}

    # Speaking rate (words per minute)
    mean_wpm: float = 0.0
    wpm_std: float = 0.0

    # Pause patterns (seconds)
    mean_pause_duration: float = 0.0
    pause_duration_std: float = 0.0
    pause_frequency: float = 0.0  # pauses per minute
    filled_pause_ratio: float = 0.0  # "um", "uh", "like" as fraction of pauses

    # Turn boundaries (seconds)
    mean_turn_delay: float = 0.0  # avg time between speaker turns
    turn_delay_std: float = 0.0
    interruption_recovery: float = 0.0  # avg ms to recover from overlap

    # Thought allocation
    thought_duration_mean: float = 0.0  # avg length of uninterrupted thought
    thought_duration_std: float = 0.0
    thought_complexity: float = 0.0  # avg number of clauses per thought

    # Anticipation (how early they react to humor/punchlines)
    anticipation_offset: float = 0.0  # ms before punchline they laugh
    laughter_onset: float = 0.0  # typical laughter duration
    laughter_frequency: float = 0.0  # laughs per minute

    # Classified style
    speaking_rate: SpeakingRate = SpeakingRate.MODERATE
    turn_style: TurnStyle = TurnStyle.RHYTHMIC


class ProsodyEnvelope(BaseModel):
    """
    The melodic shape of someone's voice.

    Pitch contour, energy shapes, and rate variability over time.
    """
    model_config = {"extra": "allow"}

    # Fundamental frequency (Hz)
    mean_f0: float = 120.0
    f0_std: float = 20.0
    f0_range: Tuple[float, float] = (80.0, 200.0)
    f0_contour: List[float] = Field(default_factory=list)

    # Energy / loudness
    mean_energy: float = 0.0
    energy_std: float = 0.0
    energy_contour: List[float] = Field(default_factory=list)

    # Rate variability
    rate_contour: List[float] = Field(default_factory=list)

    # eGeMAPS features (25-dimensional vector)
    egemaps_vector: Optional[List[float]] = None

    # Speaker embedding (if available)
    speaker_embedding: Optional[List[float]] = None


class LexicalFingerprint(BaseModel):
    """
    The lexical/semantic signature — not just *what* words, but *how*
    they're organized.
    """
    model_config = {"extra": "allow"}

    # Sentence length distribution
    mean_sentence_length: float = 0.0
    sentence_length_std: float = 0.0

    # Signature phrases
    signature_phrases: List[str] = Field(default_factory=list)
    transition_phrases: List[str] = Field(default_factory=list)
    hedge_phrases: List[str] = Field(default_factory=list)

    # Question patterns
    question_rate: float = 0.0  # questions per minute
    rhetorical_question_rate: float = 0.0

    # Narrative constructs
    analogy_rate: float = 0.0  # analogies per minute
    anecdote_rate: float = 0.0  # anecdotes per minute
    technical_density: float = 0.0  # jargon/terms per sentence

    # Topic affinity (through semantic embedding)
    topic_embedding: Optional[List[float]] = None


class GrooveParameters(BaseModel):
    """
    Conversational groove — the rhythmic pocket.

    This maps to fleet-midi-pulse concepts for conversational timing.
    """
    model_config = {"extra": "allow"}

    # Conversational BPM (turns per minute)
    conversational_bpm: float = 60.0

    # Swing factor (0.0 = straight, 0.5 = heavy swing)
    swing_factor: float = 0.0

    # Fermata threshold (how long a pause before it becomes "significant")
    fermata_threshold: float = 1.5  # seconds

    # Call-and-response ratio
    call_response_ratio: float = 1.0  # 1.0 = equal, >1.0 = they talk more

    # Anticipation window (how far ahead they predict turn boundaries)
    anticipation_window: float = 0.3  # seconds

    # Groove vector (complex, can be composed)
    groove_vector: Optional[List[float]] = None


class Persona(BaseModel):
    """
    A complete persona — the cognitive fingerprint of a speaker.

    This is the core data structure. It captures everything needed to
    render content *through* someone's voice.
    """
    model_config = {"extra": "allow"}

    # Identity
    id: str = Field(default_factory=lambda: f"persona-{uuid.uuid4().hex[:12]}")
    name: str = "unknown"
    source: str = ""  # where this persona was derived from
    created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Core profiles
    cadence: CadenceProfile = Field(default_factory=CadenceProfile)
    prosody: ProsodyEnvelope = Field(default_factory=ProsodyEnvelope)
    lexical: LexicalFingerprint = Field(default_factory=LexicalFingerprint)
    groove: GrooveParameters = Field(default_factory=GrooveParameters)

    # Semantic persona embedding (combined vector for similarity search)
    persona_vector: Optional[List[float]] = None

    # Metadata
    source_duration_seconds: float = 0.0
    source_file_count: int = 0
    confidence: float = 0.0  # how confident we are in this persona

    # SMP seed for deterministic identity
    smp_seed: Optional[str] = None

    # Tags for discoverability
    tags: List[str] = Field(default_factory=list)


class PersonaManifest(BaseModel):
    """
    Lightweight manifest for fleet discovery.
    """
    id: str
    name: str
    tags: List[str]
    capabilities: List[str] = Field(default_factory=list)
    vector_dim: int = 0
    confidence: float = 0.0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CharacterParameters(BaseModel):
    """
    Dialable character parameters for vibe-coding.

    These are the "sliders" you adjust to build a character persona.
    Maps to how a jazz musician picks their feel on a standard.
    """
    model_config = {"extra": "allow"}

    name: str = "New Character"
    base_persona_id: Optional[str] = None  # start from a real person

    # Voice character
    pitch_baseline: float = 120.0  # Hz
    pitch_variability: float = 0.5  # 0-1, how melodic
    speed: float = 1.0  # 0.5-2.0 playback rate
    breathiness: float = 0.0  # 0-1

    # Cadence character
    pause_length: float = 1.0  # 0-2 seconds
    pause_variability: float = 0.5  # 0-1
    thought_length: float = 1.0  # 0-2 seconds per thought
    interruption_tolerance: float = 0.5  # 0-1

    # Groove character
    conversational_bpm: float = 60.0
    swing: float = 0.0  # 0-0.5
    anticipation: float = 0.3  # seconds
    call_response: float = 1.0  # ratio

    # Personality proxies
    formality: float = 0.5  # 0=casual, 1=formal
    enthusiasm: float = 0.5  # 0=flat, 1=exuberant
    humor_frequency: float = 0.3  # 0-1
    technical_density: float = 0.5  # 0=casual vocab, 1=dense jargon
    verbosity: float = 0.5  # 0=terse, 1=expansive
    assertiveness: float = 0.5  # 0=hedging, 1=confident

    # Visual/artistic character (for generative rendering)
    art_style: str = "realistic"
    color_palette: List[str] = Field(default_factory=list)
    composition_style: str = "balanced"
    rhythm_section: str = "swing"

    def to_persona(self) -> Persona:
        """Convert character parameters to a Persona for the composition engine."""
        p = Persona(name=self.name)
        p.cadence.mean_pause_duration = self.pause_length
        p.cadence.mean_wpm = self.speed * 150.0
        p.cadence.thought_duration_mean = self.thought_length
        p.cadence.speaking_rate = (
            SpeakingRate.FAST if self.speed > 1.3
            else SpeakingRate.SLOW if self.speed < 0.7
            else SpeakingRate.MODERATE
        )
        p.groove.conversational_bpm = self.conversational_bpm
        p.groove.swing_factor = self.swing
        p.groove.anticipation_window = self.anticipation
        p.groove.call_response_ratio = self.call_response
        p.prosody.mean_f0 = self.pitch_baseline
        p.prosody.f0_std = self.pitch_variability * 50.0
        p.lexical.technical_density = self.technical_density
        p.tags = ["character", "synthetic"]
        if self.art_style:
            p.tags.append(f"art:{self.art_style}")
        return p
