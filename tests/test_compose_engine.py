"""Real behavioral tests for the pure-text parts of CompositionEngine.

_adapt_rhythm and _shape_prosody don't touch audio/TTS, so they're
testable without piper/opensmile. Full compose()/compose_interactive()
require a real piper binary and are out of scope here.
"""
import numpy as np
import pytest

from persona_engine.compose.engine import CompositionEngine
from persona_engine.schemas.persona import Persona


@pytest.fixture
def engine():
    return CompositionEngine()


def _persona_with_pause_frequency(freq: float) -> Persona:
    p = Persona(name="test")
    p.cadence.pause_frequency = freq
    return p


def test_adapt_rhythm_preserves_all_sentence_content(engine):
    persona = _persona_with_pause_frequency(0.0)  # pause_chance == 0
    content = "First thought here. Second thought follows. Third and final."
    result = engine._adapt_rhythm(content, persona)

    for fragment in ["First thought here", "Second thought follows", "Third and final"]:
        assert fragment in result


def test_adapt_rhythm_zero_pause_frequency_never_inserts_long_pause(engine):
    persona = _persona_with_pause_frequency(0.0)
    content = "One. Two. Three. Four. Five."
    # pause_chance = min(1.0, 0/10) = 0, so np.random.random() < 0 is always False
    result = engine._adapt_rhythm(content, persona)
    assert "\n\n" not in result


def test_adapt_rhythm_high_pause_frequency_always_inserts_long_pause(engine, monkeypatch):
    # pause_frequency=100 -> pause_chance = min(1.0, 100/10) = 1.0, always taken
    persona = _persona_with_pause_frequency(100.0)
    monkeypatch.setattr(np.random, "random", lambda: 0.5)  # any value < 1.0
    content = "One. Two. Three."
    result = engine._adapt_rhythm(content, persona)
    assert "\n\n" in result


def test_adapt_rhythm_ends_with_period(engine):
    persona = _persona_with_pause_frequency(1.0)
    result = engine._adapt_rhythm("Just one sentence", persona)
    assert result.endswith(".")


def test_adapt_rhythm_normalizes_question_and_exclamation_marks(engine):
    """! and ? are converted to . before splitting into sentences (per source)."""
    persona = _persona_with_pause_frequency(0.0)
    content = "Is this real? Yes it is! Truly."
    result = engine._adapt_rhythm(content, persona)
    # All three clauses should survive as content, regardless of original punctuation.
    assert "Is this real" in result
    assert "Yes it is" in result
    assert "Truly" in result
