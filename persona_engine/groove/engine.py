"""
Groove Engine — conversational rhythm as musical timing.

Maps fleet-midi-pulse concepts to conversational timing:
- BPM = conversational turn rate
- Swing = imbalance between speaker turns
- Fermata = pregnant pause before a big thought
- TickEvents = turn boundary signals

The groove engine turns conversation into a rhythmic performance.
You feel the beat, even when nobody's playing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TickType(Enum):
    QUARTER = "quarter"          # normal turn beat
    EIGHTH_SWUNG = "eighth_swung"  # swung subdivision
    FERMATA = "fermata"          # held pause / pregnant silence
    BREATH = "breath"            # breath point
    ACCENT = "accent"            # emphasis / punchline
    ANTICIPATION = "anticipation"  # pre-emptive laugh or reaction


@dataclass
class TickEvent:
    """A rhythmic event in conversational time."""
    tick_type: TickType
    timestamp: float  # seconds from start
    duration: float   # event duration
    energy: float     # 0-1 emphasis
    label: str = ""


@dataclass
class GrooveState:
    """
    The current groove pocket — who's got the beat.
    """
    bpm: float = 120.0
    swing_ratio: float = 0.0  # 0.0 = even, 0.5 = heavy swing
    time_signature: Tuple[int, int] = (4, 4)
    current_bar: int = 0
    current_beat: int = 0
    is_leader: bool = True  # who's setting the groove

    # Anticipation window (how far ahead we predict/react)
    anticipation_ms: float = 300.0

    # State
    ticks: List[TickEvent] = field(default_factory=list)

    @property
    def beat_interval(self) -> float:
        """Time between quarter-note beats in seconds."""
        return 60.0 / self.bpm

    @property
    def bar_duration(self) -> float:
        """Duration of one bar in seconds."""
        return self.beat_interval * self.time_signature[0]


class GrooveEngine:
    """
    Orchestrates conversational timing like a jazz rhythm section.

    Usage:
        groove = GrooveEngine(bpm=60, swing=0.15)
        groove.start_turn(speaker="A", anticipated_delay=0.3)
        groove.emit_punchline(timestamp=2.5)
        groove.end_turn()
    """

    def __init__(
        self,
        bpm: float = 60.0,
        swing: float = 0.0,
        anticipation_ms: float = 300.0,
    ):
        self.state = GrooveState(
            bpm=bpm,
            swing_ratio=swing,
            anticipation_ms=anticipation_ms,
        )
        self.on_tick: Optional[Callable[[TickEvent], None]] = None

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    def start_turn(
        self,
        speaker: str,
        anticipated_delay: float = 0.3,
        is_interruption: bool = False,
    ) -> float:
        """
        Start a conversational turn.

        Returns the timestamp when speech should begin (accounts for
        turn-taking delay and anticipation).
        """
        delay = anticipated_delay

        if is_interruption:
            # Interruption adjusts the groove — shorter delay, higher energy
            self.state.is_leader = not self.state.is_leader
            delay *= 0.3

        self._emit_tick(TickType.QUARTER, 0.0, delay, energy=0.3 if is_interruption else 0.1)

        # If we have swing, the turn start shifts
        if self.state.swing_ratio > 0 and not is_interruption:
            swing_offset = self.state.beat_interval * self.state.swing_ratio * 0.5
            delay += swing_offset

        return delay

    def end_turn(self, wait_for_reaction: bool = True) -> float:
        """
        End a conversational turn.

        Returns the pause duration before the next speaker.
        """
        pause = self.state.beat_interval * 1.5  # ~1.5 beats of rest
        if wait_for_reaction:
            self._emit_tick(TickType.FERMATA, 0.0, pause, energy=0.0)
        return pause

    def emit_punchline(self, timestamp: float) -> TickEvent:
        """
        Emit a punchline event — listeners anticipate this.

        Returns the TickEvent so the anticipation reactor can trigger
        early laughter.
        """
        event = TickEvent(
            tick_type=TickType.ACCENT,
            timestamp=timestamp,
            duration=self.state.anticipation_ms / 1000 * 1.5,
            energy=0.9,
            label="punchline",
        )
        self.state.ticks.append(event)
        self._emit_tick(event.tick_type, event.timestamp, event.duration, event.energy)
        return event

    def anticipate(
        self,
        reference_tick: TickEvent,
        offset_ms: float = 0.0,
    ) -> TickEvent:
        """
        Anticipate an event — laugh before the punchline.

        This is the key insight: humans react BEFORE events they expect.
        The anticipation window defines how far ahead they predict.
        """
        anticipation_time = max(0, reference_tick.timestamp - self.state.anticipation_ms / 1000 + offset_ms / 1000)
        event = TickEvent(
            tick_type=TickType.ANTICIPATION,
            timestamp=anticipation_time,
            duration=self.state.anticipation_ms / 1000,
            energy=reference_tick.energy * 0.7,
            label=f"anticipation:{reference_tick.label}",
        )
        self.state.ticks.append(event)
        self._emit_tick(event.tick_type, event.timestamp, event.duration, event.energy)
        return event

    def breathe(self, depth: float = 0.5) -> TickEvent:
        """Insert a breath point (natural pause for reflection)."""
        duration = 0.3 + depth * 0.5
        event = TickEvent(
            tick_type=TickType.BREATH,
            timestamp=self._current_time(),
            duration=duration,
            energy=0.2,
            label="breath",
        )
        self.state.ticks.append(event)
        self._emit_tick(event.tick_type, event.timestamp, event.duration, event.energy)
        return event

    # ------------------------------------------------------------------
    # Groove modulation
    # ------------------------------------------------------------------

    def modulate_bpm(self, target_bpm: float, ramp_seconds: float = 2.0) -> None:
        """
        Gradually change conversational tempo.

        Like a jazz group picking up the tempo for a solo.
        """
        old_bpm = self.state.bpm
        steps = int(ramp_seconds * 10)
        for i in range(1, steps + 1):
            t = i / steps
            self.state.bpm = old_bpm + (target_bpm - old_bpm) * t
        logger.info(f"Groove BPM modulated: {old_bpm:.1f} → {target_bpm:.1f}")

    def set_swing(self, swing: float) -> None:
        """Set swing ratio for the groove."""
        self.state.swing_ratio = max(0.0, min(0.5, swing))
        logger.info(f"Groove swing set to {self.state.swing_ratio:.2f}")

    def leader_follow(self, is_now_leader: bool) -> None:
        """Swap who's leading the conversational groove."""
        self.state.is_leader = is_now_leader
        logger.info(f"Groove leader: {'self' if is_now_leader else 'other'}")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_timeline(self, duration_seconds: float) -> List[TickEvent]:
        """
        Render the groove timeline for a conversation segment.

        Produces all tick events within the time window, including
        quarter-note pulses for conversational BPM.
        """
        events = []
        beat_interval = self.state.beat_interval
        total_beats = int(duration_seconds / beat_interval)

        for beat in range(total_beats):
            t = beat * beat_interval
            is_swung = (beat % 2 == 1 and self.state.swing_ratio > 0)

            # Swing delays every other beat
            if is_swung:
                t += beat_interval * self.state.swing_ratio * 0.5

            tick_type = TickType.EIGHTH_SWUNG if is_swung else TickType.QUARTER
            energy = 0.3 if beat % 4 == 0 else 0.1  # downbeats hit harder

            event = TickEvent(
                tick_type=tick_type,
                timestamp=t,
                duration=beat_interval * 0.8,
                energy=energy,
                label=f"beat_{beat % self.state.time_signature[0] + 1}",
            )
            events.append(event)

        # Add anticipation events around any existing accented ticks
        for tick in self.state.ticks:
            if tick.tick_type == TickType.ACCENT:
                events.append(self.anticipate(tick))

        events.sort(key=lambda e: e.timestamp)
        return events

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit_tick(
        self,
        tick_type: TickType,
        timestamp: float,
        duration: float,
        energy: float,
        label: str = "",
    ) -> None:
        """Fire the on_tick callback if set."""
        if self.on_tick:
            event = TickEvent(tick_type, timestamp, duration, energy, label)
            self.on_tick(event)

    def _current_time(self) -> float:
        """Get current timeline position (sum of all tick durations)."""
        return sum(t.duration for t in self.state.ticks)


class ConversationGrooveMapper:
    """
    Maps a persona's cadence profile to groove engine parameters.

    This is the bridge between decomposed persona vectors and
    real-time conversational timing.
    """

    @staticmethod
    def persona_to_groove(
        cadence_profile: "CadenceProfile",
    ) -> Tuple[float, float, float]:
        """
        Convert a cadence profile to groove parameters.

        Returns (bpm, swing, anticipation_ms)
        """
        from persona_engine.schemas.persona import TurnStyle

        # BPM = turns per minute from thought duration
        bpm = 60.0 / max(cadence_profile.thought_duration_mean, 0.5)
        bpm = max(30.0, min(200.0, bpm))

        # Swing from turn style
        swing_map = {
            TurnStyle.PATIENT: 0.0,
            TurnStyle.INTERRUPTIBLE: 0.1,
            TurnStyle.RHYTHMIC: 0.2,
            TurnStyle.EXPLORATORY: 0.05,
        }
        swing = swing_map.get(cadence_profile.turn_style, 0.0)

        # Anticipation from pause patterns
        anticipation_ms = cadence_profile.mean_pause_duration * 1000 * 0.6
        anticipation_ms = max(100.0, min(1000.0, anticipation_ms))

        return (bpm, swing, anticipation_ms)
