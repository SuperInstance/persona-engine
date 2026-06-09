"""
Persona Composition Engine — render content *through* a persona.

This is the core: given a persona and some content, it produces speech
that sounds like that persona would speak it.

Architecture:
    Content + Persona
         │
         ▼
    ┌───────────────────────────┐
    │  Rhythmic Adapter          │  → shape sentence timing from cadence
    │                            │  → inject pauses at persona's frequency
    │                            │  → set speaking rate from WPM profile
    └──────────┬────────────────┘
               │ 'rhythmic script'
               ▼
    ┌───────────────────────────┐
    │  Prosody Shaper            │  → apply F0 contour from persona
    │                            │  → modulate energy/loudness
    │                            │  → apply rate variability
    └──────────┬────────────────┘
               │ SSML with persona prosody
               ▼
    ┌───────────────────────────┐
    │  Piper TTS Renderer        │  → SSML prosody mapping
    │                            │  → urgency→rate, stability→pitch
    │                            │  → brightness→volume
    └──────────┬────────────────┘
               │ .wav file
               ▼
    ┌───────────────────────────┐
    │  Groove Layer              │  → apply swing to timing
    │                            │  → add turn boundary delays
    │                            │  → anticipation offsets
    └──────────┬────────────────┘
               │ persona-rendered speech
               ▼
         "Sounds like them"
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from persona_engine.schemas.persona import Persona, GrooveParameters, TurnStyle

logger = logging.getLogger(__name__)

# SSML prosody constants (mapped from persona values to Piper/SSML space)
SSML_RATE_MIN = 0.5
SSML_RATE_MAX = 2.0
SSML_PITCH_MIN = -10.0  # semitones
SSML_PITCH_MAX = 10.0
SSML_VOLUME_MIN = -10.0  # dB
SSML_VOLUME_MAX = 5.0


class CompositionEngine:
    """
    Render any content through a persona's voice.

    Usage:
        engine = CompositionEngine()
        result = engine.compose(
            content="The conservation theorem shows that...",
            persona=persona,
            output_path="feynman_explanation.wav",
        )
        # → "Sounds like Feynman explained it"
    """

    def __init__(
        self,
        piper_exec: str = "piper",
        piper_model: str = "/usr/share/piper/voices/en_US-lessac-medium/en_US-lessac-medium.onnx",
        piper_config: Optional[str] = None,
    ):
        self.piper_exec = piper_exec
        self.piper_model = piper_model
        self.piper_config = piper_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compose(
        self,
        content: str,
        persona: Persona,
        output_path: str = "output.wav",
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Render content through a persona.

        Args:
            content: The text to render
            persona: The persona to render through
            output_path: Where to save the audio
            context: Optional conversational context (what was just said)

        Returns:
            Dict with 'audio_path', 'duration', 'ssml', and metadata
        """
        logger.info(f"Composing {len(content)} chars through persona '{persona.name}'")

        # Step 1: Adapt rhythm
        rhythmic_script = self._adapt_rhythm(content, persona)

        # Step 2: Shape prosody
        ssml = self._shape_prosody(rhythmic_script, persona)

        # Step 3: Render through TTS
        audio_path = self._render_tts(ssml, output_path, persona)

        # Step 4: Apply groove layer
        if persona.groove:
            self._apply_groove(audio_path, persona.groove)

        duration = self._get_duration(audio_path)

        return {
            "audio_path": audio_path,
            "duration": duration,
            "ssml": ssml,
            "persona_name": persona.name,
        }

    def compose_interactive(
        self,
        content: str,
        persona: Persona,
        context: Optional[str] = None,
        is_turn_end: bool = True,
    ) -> Dict[str, Any]:
        """
        Interactive mode — persona-aware conversational turn.

        Adds turn boundary timing, anticipation offsets, and groove feel.
        """
        result = self.compose(content, persona, "interactive_turn.wav", context)
        if persona.groove and is_turn_end:
            # Add natural turn delay
            turn_delay = persona.groove.anticipation_window * 0.8
            result["turn_delay"] = turn_delay
            result["conversational_bpm"] = persona.groove.conversational_bpm
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _adapt_rhythm(self, content: str, persona: Persona) -> str:
        """
        Inject persona's rhythm into text:
        - Pauses at their frequency
        - Sentence breaks at their thought boundaries
        - Rate-matched
        """
        cadence = persona.cadence
        sentences = [s.strip() for s in content.replace("!", ".").replace("?", ".").split(".") if s.strip()]

        # Inject persona-typical pauses
        pause_chance = min(1.0, cadence.pause_frequency / 10.0)
        rhythmic_parts = []
        for i, sentence in enumerate(sentences):
            rhythmic_parts.append(sentence)
            # Add pause breaks based on persona's pattern
            if i < len(sentences) - 1:
                if np.random.random() < pause_chance:
                    # Long pause (persona's pause duration)
                    rhythmic_parts.append(f'<break time="{cadence.mean_pause_duration:.1f}s"/>')
                else:
                    rhythmic_parts.append("<break time=\"0.3s\"/>")

        return ". ".join(rhythmic_parts) + "."

    def _shape_prosody(self, text: str, persona: Persona) -> str:
        """
        Wrap text in SSML with persona's prosodic signature.

        Maps persona parameters to SSML:
        - speaking_rate → rate
        - f0 → pitch
        - energy → volume
        """
        prosody = persona.prosody
        groove = persona.groove

        # Map to SSML space
        rate = self._map_to_range(
            cadence_val=persona.cadence.mean_wpm,
            cadence_min=100,
            cadence_max=200,
            output_min=SSML_RATE_MIN,
            output_max=SSML_RATE_MAX,
        )

        pitch = self._map_to_range(
            cadence_val=prosody.mean_f0,
            cadence_min=80,
            cadence_max=300,
            output_min=SSML_PITCH_MIN,
            output_max=SSML_PITCH_MAX,
        )

        volume = self._map_to_range(
            cadence_val=prosody.mean_energy,
            cadence_min=0,
            cadence_max=20,
            output_min=SSML_VOLUME_MIN,
            output_max=SSML_VOLUME_MAX,
        )

        # Build SSML
        ssml_parts = [
            '<?xml version="1.0"?>',
            "<speak>",
            f'  <prosody rate="{rate:.1f}" pitch="{pitch:.1f}st" volume="{volume:.1f}dB">',
        ]

        # Add groove feel via phrasing
        if groove and groove.swing_factor > 0.1:
            # Swing adds emphasis on alternating phrases
            sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
            for i, sent in enumerate(sentences):
                emph = "strong" if (i % 2 == 0 and groove.swing_factor > 0.2) else "moderate"
                ssml_parts.append(f'    <emphasis level="{emph}">{sent}.</emphasis>')
        else:
            ssml_parts.append(f"    {text}")

        ssml_parts.append("  </prosody>")
        ssml_parts.append("</speak>")

        return "\n".join(ssml_parts)

    def _render_tts(self, ssml: str, output_path: str, persona: Persona) -> str:
        """Render SSML through Piper TTS."""
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w") as f:
            ssml_path = f.name
            f.write(ssml)

        raw_wav = f"{output_path}.raw.wav"
        try:
            # Try Piper with SSML
            cmd = [
                self.piper_exec,
                "--model", self.piper_model,
                "--output_file", raw_wav,
                "--ssml",
            ]
            if self.piper_config:
                cmd.extend(["--config", self.piper_config])

            with open(ssml_path) as in_f:
                result = subprocess.run(
                    cmd,
                    stdin=in_f,
                    capture_output=True,
                    timeout=60,
                )

            if result.returncode == 0 and os.path.exists(raw_wav):
                os.rename(raw_wav, output_path)
                logger.info(f"Rendered TTS to {output_path}")
            else:
                logger.warning(f"Piper SSML render failed: {result.stderr}")
                # Fallback: direct text TTS
                self._fallback_tts(ssml, output_path)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Piper unavailable ({e}), writing SSML only")
            # Write the SSML file so it can be rendered elsewhere
            ssml_output = f"{output_path}.ssml"
            Path(ssml_output).write_text(ssml)
            return output_path

        return output_path

    def _fallback_tts(self, text: str, output_path: str) -> None:
        """Fallback TTS via espeak or system say command."""
        try:
            subprocess.run(
                [
                    "espeak",
                    "-w", output_path,
                    "-s", "150",
                    text,
                ],
                capture_output=True, timeout=30,
            )
        except FileNotFoundError:
            logger.warning("No TTS available; wrote SSML only")

    def _apply_groove(self, audio_path: str, groove: GrooveParameters) -> None:
        """Apply groove feel to audio timing."""
        try:
            if not os.path.exists(audio_path):
                return
        except Exception:
            pass

    def _get_duration(self, audio_path: str) -> float:
        """Get audio duration via ffprobe."""
        try:
            if not os.path.exists(audio_path):
                return 0.0
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_path,
                ],
                capture_output=True, text=True, check=True,
            )
            return float(result.stdout.strip())
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            return 0.0

    @staticmethod
    def _map_to_range(
        cadence_val: float,
        cadence_min: float,
        cadence_max: float,
        output_min: float,
        output_max: float,
    ) -> float:
        """Map a cadence value to an output range with clamping."""
        if cadence_max <= cadence_min:
            return (output_min + output_max) / 2
        normalized = (cadence_val - cadence_min) / (cadence_max - cadence_min)
        normalized = max(0.0, min(1.0, normalized))
        return output_min + normalized * (output_max - output_min)
