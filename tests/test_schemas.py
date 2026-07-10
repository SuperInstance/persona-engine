"""Real behavioral tests for persona_engine.schemas.persona.

Covers construction/validation of the pydantic models and, critically,
loads the real committed character fixtures in characters/*.json to
catch schema drift against real data.
"""
import json
from pathlib import Path

import pytest

from persona_engine.schemas.persona import (
    CadenceProfile,
    CharacterParameters,
    Persona,
    SpeakingRate,
)

CHARACTERS_DIR = Path(__file__).resolve().parent.parent / "characters"


def test_persona_default_construction():
    p = Persona()
    assert p.name == "unknown"
    assert p.id.startswith("persona-")
    assert isinstance(p.cadence, CadenceProfile)
    assert p.cadence.speaking_rate == SpeakingRate.MODERATE


def test_persona_extra_fields_allowed():
    # model_config = {"extra": "allow"} — confirm unknown fields don't raise.
    p = Persona(name="Test", some_future_field="ok")
    assert p.name == "Test"


@pytest.mark.parametrize(
    "fixture_path",
    sorted(CHARACTERS_DIR.glob("*.json")),
    ids=lambda p: p.name,
)
def test_character_fixtures_validate_against_current_schema(fixture_path):
    """Every committed characters/*.json must still parse as CharacterParameters.

    This is the schema-drift check: if the model changes shape without
    updating these fixtures (or vice versa), this test fails loudly
    instead of silently producing a broken character at runtime.
    """
    data = json.loads(fixture_path.read_text())
    char = CharacterParameters(**data)
    assert char.name  # every fixture has a real name, not the default


def test_character_parameters_to_persona_maps_fields_correctly():
    char = CharacterParameters(
        name="Dial Test",
        speed=1.5,
        pause_length=0.4,
        thought_length=0.8,
        conversational_bpm=90.0,
        swing=0.2,
        anticipation=0.5,
        call_response=1.2,
        pitch_baseline=180.0,
        pitch_variability=0.7,
        technical_density=0.9,
        art_style="noir",
    )
    persona = char.to_persona()

    assert persona.name == "Dial Test"
    assert persona.cadence.mean_pause_duration == 0.4
    assert persona.cadence.mean_wpm == 1.5 * 150.0
    assert persona.cadence.thought_duration_mean == 0.8
    # speed=1.5 > 1.3 -> FAST per the documented threshold
    assert persona.cadence.speaking_rate == SpeakingRate.FAST
    assert persona.groove.conversational_bpm == 90.0
    assert persona.groove.swing_factor == 0.2
    assert persona.groove.anticipation_window == 0.5
    assert persona.groove.call_response_ratio == 1.2
    assert persona.prosody.mean_f0 == 180.0
    assert persona.prosody.f0_std == 0.7 * 50.0
    assert persona.lexical.technical_density == 0.9
    assert "character" in persona.tags
    assert "synthetic" in persona.tags
    assert "art:noir" in persona.tags


@pytest.mark.parametrize(
    "speed,expected_rate",
    [(0.5, SpeakingRate.SLOW), (1.0, SpeakingRate.MODERATE), (1.8, SpeakingRate.FAST)],
)
def test_speed_to_speaking_rate_thresholds(speed, expected_rate):
    """Documented thresholds in to_persona(): >1.3 FAST, <0.7 SLOW, else MODERATE."""
    char = CharacterParameters(speed=speed)
    persona = char.to_persona()
    assert persona.cadence.speaking_rate == expected_rate
