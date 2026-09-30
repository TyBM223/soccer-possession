"""Possession spec as tests, using synthetic detections (no video).

These tests avoid the open design choices listed in src/possession.py
(distance exactly at the limit, equidistant players, window alignment, tied
votes), so any reasonable choice for those passes.
"""

import pytest

from src.possession import possession_stats, raw_call, smooth_calls

A, B = 0, 1
MAX_PX = 80


# --- raw_call ----------------------------------------------------------------

def test_raw_call_nearest_player_within_range():
    players = [((100.0, 100.0), A), ((300.0, 300.0), B)]
    assert raw_call((110.0, 105.0), players, MAX_PX) == A


def test_raw_call_nearest_wins_over_other_team_in_range():
    # Both players are within range; B is closer.
    players = [((100.0, 100.0), A), ((150.0, 100.0), B)]
    assert raw_call((140.0, 100.0), players, MAX_PX) == B


def test_raw_call_ball_missing_is_no_call():
    players = [((100.0, 100.0), A)]
    assert raw_call(None, players, MAX_PX) is None


def test_raw_call_ball_too_far_from_all_players_is_no_call():
    players = [((100.0, 100.0), A), ((300.0, 300.0), B)]
    assert raw_call((700.0, 700.0), players, MAX_PX) is None


def test_raw_call_uses_euclidean_distance():
    # dx=60, dy=60 -> ~84.9 px: each axis is within 80, but the distance is not.
    players = [((100.0, 100.0), A)]
    assert raw_call((160.0, 160.0), players, MAX_PX) is None


def test_raw_call_no_players_is_no_call():
    assert raw_call((100.0, 100.0), [], MAX_PX) is None


def test_raw_call_respects_max_distance_argument():
    players = [((100.0, 100.0), A)]
    ball = (150.0, 100.0)  # 50 px away
    assert raw_call(ball, players, 80) == A
    assert raw_call(ball, players, 40) is None


# --- smooth_calls ------------------------------------------------------------

def test_smooth_single_frame_flicker_smoothed_out():
    raw = [A, A, A, B, A, A, A]
    assert smooth_calls(raw, 5) == [A] * 7


def test_smooth_preserves_length():
    raw = [A, None, B, B, None, A, A, A, B]
    assert len(smooth_calls(raw, 5)) == len(raw)


def test_smooth_no_call_frames_do_not_vote():
    # Missing-ball frames don't turn the call into None or outvote A.
    raw = [A, None, None, None, A]
    assert smooth_calls(raw, 5) == [A] * 5


def test_smooth_majority_wins():
    raw = [B, B, A, B, A]
    # The window centered on / ending at frame 2 has votes B, B, A (+ more) -> B.
    assert smooth_calls(raw, 5)[2] == B


def test_smooth_empty_window_is_no_call():
    raw = [A] + [None] * 10
    smoothed = smooth_calls(raw, 5)
    # The last 5 frames only see None in their windows.
    assert smoothed[-5:] == [None] * 5


def test_smooth_all_none_stays_none():
    assert smooth_calls([None] * 6, 5) == [None] * 6


def test_smooth_window_one_is_identity():
    raw = [A, B, None, A, B, B, None]
    assert smooth_calls(raw, 1) == raw


# --- possession_stats --------------------------------------------------------

def test_stats_basic_split():
    stats = possession_stats([A, A, A, B])
    assert stats["possession_pct"][A] == pytest.approx(75.0)
    assert stats["possession_pct"][B] == pytest.approx(25.0)
    assert stats["coverage_pct"] == pytest.approx(100.0)
    assert stats["called_frames"] == 4
    assert stats["total_frames"] == 4


def test_stats_no_call_frames_excluded_from_possession():
    stats = possession_stats([A, A, None, None, B, B])
    assert stats["possession_pct"][A] == pytest.approx(50.0)
    assert stats["possession_pct"][B] == pytest.approx(50.0)


def test_stats_coverage_computed_correctly():
    stats = possession_stats([A, None, None, None, B, None, None, None])
    assert stats["coverage_pct"] == pytest.approx(25.0)
    assert stats["called_frames"] == 2
    assert stats["total_frames"] == 8
    # Possession uses only the 2 called frames.
    assert stats["possession_pct"][A] == pytest.approx(50.0)
    assert stats["possession_pct"][B] == pytest.approx(50.0)


def test_stats_possession_sums_to_100_when_any_call():
    stats = possession_stats([A, None, B, B, None, A, A])
    assert sum(stats["possession_pct"].values()) == pytest.approx(100.0)


def test_stats_all_no_call():
    stats = possession_stats([None, None, None])
    assert stats["possession_pct"] == {}
    assert stats["coverage_pct"] == pytest.approx(0.0)
    assert stats["called_frames"] == 0
    assert stats["total_frames"] == 3


def test_stats_empty_input():
    stats = possession_stats([])
    assert stats["possession_pct"] == {}
    assert stats["coverage_pct"] == pytest.approx(0.0)
    assert stats["total_frames"] == 0


# --- end to end on synthetic detections --------------------------------------

def test_end_to_end_ball_missing_and_far_are_excluded():
    a_player = ((100.0, 100.0), A)
    b_player = ((500.0, 100.0), B)
    players = [a_player, b_player]
    near_a = (110.0, 100.0)
    near_b = (490.0, 100.0)
    far = (300.0, 900.0)

    balls = (
        [near_a] * 6       # A has the ball
        + [None] * 6       # ball not detected
        + [far] * 6        # ball in the air, nobody close
        + [near_b] * 6     # B has the ball
    )
    raws = [raw_call(b, players, MAX_PX) for b in balls]
    assert raws == [A] * 6 + [None] * 12 + [B] * 6

    # window=1: exact numbers. The 12 no-call frames go to neither team.
    stats = possession_stats(smooth_calls(raws, 1))
    assert stats["possession_pct"][A] == pytest.approx(50.0)
    assert stats["possession_pct"][B] == pytest.approx(50.0)
    assert stats["coverage_pct"] == pytest.approx(50.0)
    assert stats["total_frames"] == 24

    # window=5: the exact split depends on window alignment, but a 12-frame
    # gap is longer than the window, so some frames stay uncalled.
    stats = possession_stats(smooth_calls(raws, 5))
    assert stats["called_frames"] < stats["total_frames"]
    assert set(stats["possession_pct"]) == {A, B}
    assert sum(stats["possession_pct"].values()) == pytest.approx(100.0)
