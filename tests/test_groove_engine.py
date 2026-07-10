"""Real behavioral tests for the pure math/state parts of GrooveEngine."""
import pytest

from persona_engine.groove.engine import GrooveEngine, GrooveState


def test_beat_interval_is_60_over_bpm():
    state = GrooveState(bpm=120.0)
    assert state.beat_interval == pytest.approx(0.5)

    state2 = GrooveState(bpm=60.0)
    assert state2.beat_interval == pytest.approx(1.0)


def test_bar_duration_is_beat_interval_times_beats_per_bar():
    state = GrooveState(bpm=120.0, time_signature=(4, 4))
    assert state.bar_duration == pytest.approx(state.beat_interval * 4)

    state_waltz = GrooveState(bpm=90.0, time_signature=(3, 4))
    assert state_waltz.bar_duration == pytest.approx(state_waltz.beat_interval * 3)


def test_set_swing_clamps_to_documented_range():
    engine = GrooveEngine(bpm=90.0)
    engine.set_swing(-1.0)
    assert engine.state.swing_ratio == 0.0

    engine.set_swing(10.0)
    assert engine.state.swing_ratio == 0.5

    engine.set_swing(0.25)
    assert engine.state.swing_ratio == pytest.approx(0.25)


def test_modulate_bpm_reaches_target():
    engine = GrooveEngine(bpm=60.0)
    engine.modulate_bpm(target_bpm=120.0, ramp_seconds=0.1)
    # The ramp loop's final step sets t == 1.0, landing exactly on target.
    assert engine.state.bpm == pytest.approx(120.0)


def test_leader_follow_toggles_state():
    engine = GrooveEngine()
    assert engine.state.is_leader is True
    engine.leader_follow(False)
    assert engine.state.is_leader is False
    engine.leader_follow(True)
    assert engine.state.is_leader is True


def test_render_timeline_returns_ticks_within_requested_duration():
    engine = GrooveEngine(bpm=120.0)
    ticks = engine.render_timeline(duration_seconds=4.0)
    assert isinstance(ticks, list)
    if ticks:
        # Every emitted tick's timestamp should fall within the requested window.
        for tick in ticks:
            assert 0.0 <= tick.timestamp <= 4.0 + engine.state.beat_interval
