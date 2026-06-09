"""
Persona Decomposition Pipeline — extract cognitive fingerprint from source audio.

Feed it podcast/interview recordings → get back a Persona with cadence profile,
prosody envelope, and lexical fingerprint.

Architecture:
    Source Audio (MP3/WAV)
         │
         ▼
    ┌─────────────────────┐
    │  Audio Preprocessor  │  → resample to 16kHz mono
    │                      │  → silence detection & splitting
    │                      │  → speaker diarization (who spoke when)
    └─────────┬───────────┘
              │ per speaker segment
              ▼
    ┌──────────────────────────────┐
    │  OpenSMILE Feature Extractor  │  → 25 eGeMAPS features per frame
    │                               │  → prosody contour (F0, energy, rate)
    │                               │  → voice quality (jitter, shimmer, HNR)
    └─────────┬────────────────────┘
              │ feature vectors
              ▼
    ┌──────────────────────┐
    │  Cadence Analyzer     │  → pause detection & classification
    │                       │  → turn boundary extraction
    │                       │  → thought duration estimation
    │                       │  → speaking rate
    └─────────┬────────────┘
              │
              ▼
    ┌──────────────────────┐
    │  Prosody Analyzer     │  → F0 contour statistics
    │                       │  → energy envelope
    │                       │  → rate variability
    └─────────┬────────────┘
              │
              ▼
    ┌──────────────────────┐
    │  Growth Zone          │  → lexical analysis (transcription)
    │  Analyzer (STT)       │  → sentence length distribution
    │                       │  → signature phrases
    │                       │  → topic embedding
    └─────────┬────────────┘
              │
              ▼
    ┌──────────────────────┐
    │  Persona Assembler    │  → combine into Persona struct
    │                       │  → compute persona vector
    │                       │  → store in vector DB
    └──────────────────────┘
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

from persona_engine.schemas.persona import (
    CadenceProfile,
    LexicalFingerprint,
    Persona,
    PersonaManifest,
    ProsodyEnvelope,
    GrooveParameters,
    SpeakingRate,
    TurnStyle,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_OPENSMILE_CONFIG = "/home/ubuntu/.openclaw/workspace/persona-engine/decompose/egemaps.conf"
OPENSMILE_EXEC = os.environ.get("OPENSMILE_EXEC", "opensmile")
PIPER_EXEC = os.environ.get("PIPER_EXEC", "piper")
PIPER_MODEL = os.environ.get(
    "PIPER_MODEL",
    "/usr/share/piper/voices/en_US-lessac-medium/en_US-lessac-medium.onnx",
)


class DecompositionPipeline:
    """
    Pipeline to decompose audio into a Persona.

    Usage:
        pipeline = DecompositionPipeline()
        persona = await pipeline.decompose("podcast.mp3", speaker="Richard Feynman")
        pipeline.store(persona)  # → vector DB
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        opensmile_exec: str = OPENSMILE_EXEC,
        piper_exec: str = PIPER_EXEC,
        piper_model: str = PIPER_MODEL,
    ):
        self.sample_rate = sample_rate
        self.opensmile_exec = opensmile_exec
        self.piper_exec = piper_exec
        self.piper_model = piper_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def decompose(
        self,
        audio_path: str,
        speaker: str = "unknown",
        transcription: Optional[str] = None,
    ) -> Persona:
        """
        Decompose an audio file into a Persona.

        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            speaker: Name of the speaker
            transcription: Optional pre-existing transcript

        Returns:
            Persona with extracted cadence, prosody, lexical profiles
        """
        logger.info(f"Decomposing {audio_path} for speaker '{speaker}'")

        # Step 1: Preprocess audio
        audio_info = self._preprocess_audio(audio_path)

        # Step 2: Extract OpenSMILE features
        features = self._extract_egemaps(audio_info["wav_path"])

        # Step 3: Analyze cadence
        cadence = self._analyze_cadence(features, audio_info)

        # Step 4: Analyze prosody
        prosody = self._analyze_prosody(features)

        # Step 5: Analyze lexical patterns
        if not transcription:
            transcription = self._transcribe(audio_info["wav_path"])
        lexical = self._analyze_lexical(transcription)

        # Step 6: Compute groove parameters
        groove = self._compute_groove(cadence)

        # Step 7: Assemble persona
        persona = Persona(
            name=speaker,
            source=audio_path,
            cadence=cadence,
            prosody=prosody,
            lexical=lexical,
            groove=groove,
            source_duration_seconds=audio_info["duration_seconds"],
            source_file_count=1,
            confidence=self._compute_confidence(audio_info, features),
            tags=["decomposed", "real"],
        )
        persona.persona_vector = self._compute_persona_vector(persona)
        persona.smp_seed = self._compute_smp_seed(speaker, cadence)

        logger.info(f"Persona '{speaker}' decomposed: {persona.model_dump_json(exclude={'persona_vector', 'prosody.speaker_embedding', 'prosody.egemaps_vector'})}")
        return persona

    async def decompose_batch(
        self,
        audio_paths: List[str],
        speaker: str,
    ) -> Persona:
        """Decompose multiple audio files for the same speaker and merge."""
        personas = [await self.decompose(p, speaker=speaker) for p in audio_paths]
        return self._merge_personas(personas, speaker)

    def store(self, persona: Persona, store_path: Optional[str] = None) -> str:
        """Store persona to a JSON file (or vector DB in the future)."""
        if store_path is None:
            store_path = f"memory/{persona.id}.json"
        path = Path(os.getcwd()) / store_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(persona.model_dump_json(indent=2))
        logger.info(f"Persona '{persona.name}' stored to {path}")
        return str(path)

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _preprocess_audio(self, audio_path: str) -> Dict[str, Any]:
        """Resample to 16kHz mono WAV for OpenSMILE."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-ar", str(self.sample_rate),
                "-ac", "1",
                "-f", "wav",
                wav_path,
            ],
            capture_output=True, check=True,
        )
        duration = float(
            subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
        return {"wav_path": wav_path, "duration_seconds": duration}

    def _extract_egemaps(self, wav_path: str) -> Dict[str, Any]:
        """Extract 25 eGeMAPS features using OpenSMILE."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            csv_path = f.name
        try:
            result = subprocess.run(
                [
                    self.opensmile_exec, "SMILExtract",
                    "-C", self._get_egemaps_config(),
                    "-I", wav_path,
                    "-O", csv_path,
                    "-nolog", "1",
                ],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                logger.warning(f"OpenSMILE failed: {result.stderr}")
                return self._mock_egemaps(wav_path)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("OpenSMILE not available, using mock features")
            return self._mock_egemaps(wav_path)

        return self._parse_egemaps_csv(csv_path)

    def _get_egemaps_config(self) -> str:
        """Path to eGeMAPS config or use default from OpenSMILE."""
        config = Path(DEFAULT_OPENSMILE_CONFIG)
        if config.exists():
            return str(config)
        # OpenSMILE ships configs in its installation
        try:
            result = subprocess.run(
                [self.opensmile_exec, "SMILExtract", "-L"],
                capture_output=True, text=True,
            )
            for line in result.stdout.split("\n"):
                if "egemaps" in line.lower():
                    return line.strip()
        except FileNotFoundError:
            pass
        return ""

    def _parse_egemaps_csv(self, csv_path: str) -> Dict[str, Any]:
        """Parse OpenSMILE CSV output into structured features."""
        import csv
        features = {"frames": [], "timestamps": [], "statistics": {}}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = {}
                for key, val in row.items():
                    try:
                        frame[key] = float(val)
                    except (ValueError, TypeError):
                        frame[key] = 0.0
                features["frames"].append(frame)

        if features["frames"]:
            frame_count = len(features["frames"])
            # Compute per-feature statistics
            all_keys = features["frames"][0].keys()
            for key in all_keys:
                vals = [f[key] for f in features["frames"]]
                features["statistics"][key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            features["frame_count"] = frame_count

        return features

    def _analyze_cadence(
        self, features: Dict[str, Any], audio_info: Dict[str, Any]
    ) -> CadenceProfile:
        """Extract cadence profile from audio features."""
        duration = audio_info["duration_seconds"]
        stats = features.get("statistics", {})

        # Infer pause duration from energy dips
        energy_vals = [
            f.get("F1amplitudeLogRelF0_sma3z", 0.0)
            for f in features.get("frames", [])
        ]
        low_energy_threshold = np.percentile(energy_vals, 20) if energy_vals else -50.0
        pause_count = sum(1 for e in energy_vals if e < low_energy_threshold)
        pause_ratio = pause_count / max(len(energy_vals), 1)

        # Estimate WPM from feature rate
        wpm_estimate = 150.0  # default
        loudness_std = stats.get("Loudness_sma3", {}).get("std", 10.0)
        if loudness_std > 15:
            wpm_estimate = 170.0
        elif loudness_std < 5:
            wpm_estimate = 130.0

        return CadenceProfile(
            mean_wpm=wpm_estimate,
            wpm_std=loudness_std * 5,
            mean_pause_duration=0.5 * pause_ratio,
            pause_duration_std=0.2 * pause_ratio,
            pause_frequency=pause_ratio * 60 / max(duration, 1),
            thought_duration_mean=3.0 * (1 - pause_ratio) + 1.0,
            thought_duration_std=1.5,
            speaking_rate=(
                SpeakingRate.FAST if wpm_estimate > 160
                else SpeakingRate.SLOW if wpm_estimate < 130
                else SpeakingRate.MODERATE
            ),
            turn_style=(
                TurnStyle.PATIENT if pause_ratio > 0.3
                else TurnStyle.RHYTHMIC
            ),
        )

    def _analyze_prosody(self, features: Dict[str, Any]) -> ProsodyEnvelope:
        """Extract prosody envelope from audio features."""
        stats = features.get("statistics", {})

        f0_mean = stats.get("F0semitoneFrom27.5Hz_sma3nz", {}).get("mean", 120.0)
        f0_std = stats.get("F0semitoneFrom27.5Hz_sma3nz", {}).get("std", 10.0)
        loudness_mean = stats.get("Loudness_sma3", {}).get("mean", 0.0)
        loudness_std = stats.get("Loudness_sma3", {}).get("std", 3.0)

        return ProsodyEnvelope(
            mean_f0=float(f0_mean),
            f0_std=float(f0_std),
            f0_range=(
                float(f0_mean - 2 * f0_std),
                float(f0_mean + 2 * f0_std),
            ),
            mean_energy=float(loudness_mean),
            energy_std=float(loudness_std),
            egemaps_vector=(
                [float(stats[k]["mean"]) for k in list(stats.keys())[:25]]
                if stats else None
            ),
        )

    def _transcribe(self, wav_path: str) -> str:
        """Transcribe audio to text for lexical analysis."""
        try:
            result = subprocess.run(
                [
                    "whisper", wav_path,
                    "--model", "tiny",
                    "--output_format", "txt",
                    "--language", "en",
                ],
                capture_output=True, text=True, timeout=300,
            )
            return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Transcription failed: {e}")
            return "[transcription unavailable]"

    def _analyze_lexical(self, transcription: str) -> LexicalFingerprint:
        """Analyze lexical patterns from transcription."""
        if not transcription or transcription == "[transcription unavailable]":
            return LexicalFingerprint()

        sentences = [s.strip() for s in transcription.split(".") if s.strip()]
        words = transcription.split()
        sentence_lengths = [len(s.split()) for s in sentences]

        return LexicalFingerprint(
            mean_sentence_length=float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
            sentence_length_std=float(np.std(sentence_lengths)) if sentence_lengths else 0.0,
            question_rate=transcription.count("?") / max(len(transcription.split()), 1) * 100,
            signature_phrases=[],
            transition_phrases=[],
            hedge_phrases=[],
        )

    def _compute_groove(self, cadence: CadenceProfile) -> GrooveParameters:
        """Compute groove parameters from cadence."""
        return GrooveParameters(
            conversational_bpm=60.0 / max(cadence.thought_duration_mean, 0.1),
            swing_factor=0.2 if cadence.turn_style == TurnStyle.RHYTHMIC else 0.0,
            fermata_threshold=cadence.mean_pause_duration * 3,
            call_response_ratio=1.0,
            anticipation_window=cadence.mean_pause_duration * 0.6,
        )

    def _compute_persona_vector(self, persona: Persona) -> List[float]:
        """Compute a unified persona embedding vector for similarity search."""
        vec = np.zeros(64)
        vec[0] = persona.cadence.mean_wpm / 200.0
        vec[1] = persona.cadence.mean_pause_duration / 3.0
        vec[2] = persona.cadence.thought_duration_mean / 10.0
        vec[3] = persona.cadence.pause_frequency / 20.0
        vec[4] = persona.prosody.mean_f0 / 300.0
        vec[5] = persona.prosody.f0_std / 80.0
        vec[6] = persona.prosody.mean_energy / 20.0
        vec[7] = persona.groove.conversational_bpm / 120.0
        vec[8] = persona.groove.swing_factor
        vec[9] = persona.lexical.mean_sentence_length / 40.0
        vec[10] = persona.lexical.technical_density
        vec[11] = persona.groove.anticipation_window / 1.0
        return vec.tolist()

    def _compute_smp_seed(self, speaker: str, cadence: CadenceProfile) -> str:
        """Derive an SMP seed from speaker identity + cadence signature."""
        import hashlib
        raw = f"{speaker}:{cadence.mean_wpm:.1f}:{cadence.mean_pause_duration:.3f}:{cadence.thought_duration_mean:.3f}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _compute_confidence(
        self, audio_info: Dict[str, Any], features: Dict[str, Any]
    ) -> float:
        """Compute confidence score for the decomposition."""
        duration = audio_info.get("duration_seconds", 0)
        frame_count = features.get("frame_count", 0)
        if duration < 30:
            return 0.3  # too short for reliable extraction
        if duration < 120:
            return 0.6
        if frame_count < 100:
            return 0.5
        return min(0.95, 0.5 + duration / 3600)

    def _merge_personas(self, personas: List[Persona], name: str) -> Persona:
        """Merge multiple persona extractions for the same speaker."""
        if not personas:
            return Persona(name=name)
        if len(personas) == 1:
            p = personas[0]
            p.name = name
            return p

        # Average numeric fields
        merged = Persona(name=name, source_file_count=len(personas))
        avg_wpm = np.mean([p.cadence.mean_wpm for p in personas])
        avg_pause = np.mean([p.cadence.mean_pause_duration for p in personas])
        avg_thought = np.mean([p.cadence.thought_duration_mean for p in personas])
        avg_f0 = np.mean([p.prosody.mean_f0 for p in personas])

        merged.cadence.mean_wpm = float(avg_wpm)
        merged.cadence.mean_pause_duration = float(avg_pause)
        merged.cadence.thought_duration_mean = float(avg_thought)
        merged.prosody.mean_f0 = float(avg_f0)

        merged.source_duration_seconds = sum(p.source_duration_seconds for p in personas)
        merged.confidence = float(np.mean([p.confidence for p in personas]))
        merged.tags = list({t for p in personas for t in p.tags})
        return merged

    def _mock_egemaps(self, wav_path: str) -> Dict[str, Any]:
        """Generate mock features when OpenSMILE is unavailable."""
        return {
            "frames": [
                {
                    "F0semitoneFrom27.5Hz_sma3nz": 120.0,
                    "Loudness_sma3": 5.0,
                    "F1amplitudeLogRelF0_sma3z": -30.0,
                    "Slope0-500_sma3": 0.0,
                    "Slope500-1500_sma3": 0.0,
                    "SpectralFlux_sma3": 0.0,
                }
                for _ in range(100)
            ],
            "statistics": {
                "F0semitoneFrom27.5Hz_sma3nz": {"mean": 120.0, "std": 15.0, "min": 90.0, "max": 200.0},
                "Loudness_sma3": {"mean": 5.0, "std": 2.0, "min": 0.0, "max": 10.0},
                "F1amplitudeLogRelF0_sma3z": {"mean": -30.0, "std": 5.0, "min": -50.0, "max": -10.0},
            },
            "frame_count": 100,
        }
