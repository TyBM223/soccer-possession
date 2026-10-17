"""Overlay drawing on blank frames: what's inside the bar, and that nothing
outside the panel changes."""

import numpy as np
import supervision as sv

from src import overlay
from src.overlay import BAR_H, BAR_W, BAR_X, BAR_Y, PANEL_H, PANEL_W, PANEL_X, PANEL_Y

A, B = 0, 1
COLORS = {A: sv.Color.from_hex("#FF1493"), B: sv.Color.from_hex("#00BFFF")}
ROW = BAR_Y + 3  # near the bar's top edge, clear of the centered labels


def blank():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def stats(possession_pct, coverage_pct=50.0):
    return {"possession_pct": possession_pct, "coverage_pct": coverage_pct,
            "called_frames": 0, "total_frames": 0}


def bar_pixel(frame, fraction):
    return tuple(int(v) for v in frame[ROW, BAR_X + int(BAR_W * fraction)])


def test_input_frame_not_modified():
    frame = blank()
    overlay.draw_possession_overlay(frame, stats({A: 60.0, B: 40.0}), A, COLORS)
    assert not frame.any()


def test_nothing_drawn_outside_panel():
    out = overlay.draw_possession_overlay(blank(), stats({A: 60.0, B: 40.0}), A, COLORS)
    outside = out.copy()
    outside[PANEL_Y:PANEL_Y + PANEL_H, PANEL_X:PANEL_X + PANEL_W] = 0
    assert not outside.any()
    assert out.shape == (1080, 1920, 3) and out.dtype == np.uint8


def test_bar_split_follows_possession():
    out = overlay.draw_possession_overlay(blank(), stats({A: 75.0, B: 25.0}), A, COLORS)
    assert bar_pixel(out, 0.10) == COLORS[A].as_bgr()
    assert bar_pixel(out, 0.70) == COLORS[A].as_bgr()
    assert bar_pixel(out, 0.80) == COLORS[B].as_bgr()
    assert bar_pixel(out, 0.99) == COLORS[B].as_bgr()


def test_one_team_fills_the_bar():
    out = overlay.draw_possession_overlay(blank(), stats({B: 100.0}), B, COLORS)
    assert bar_pixel(out, 0.0) == COLORS[B].as_bgr()
    assert bar_pixel(out, 0.99) == COLORS[B].as_bgr()


def test_no_calls_yet_is_gray():
    out = overlay.draw_possession_overlay(blank(), stats({}, coverage_pct=0.0), None, COLORS)
    assert bar_pixel(out, 0.05) == overlay.NO_CALL_COLOR.as_bgr()
    assert bar_pixel(out, 0.95) == overlay.NO_CALL_COLOR.as_bgr()


def test_bar_stays_inside_its_box():
    out = overlay.draw_possession_overlay(blank(), stats({A: 33.3, B: 66.7}), B, COLORS)
    # The pixel just past the bar's right edge is panel background, not team color.
    right = tuple(int(v) for v in out[ROW, BAR_X + BAR_W])
    assert right not in (COLORS[A].as_bgr(), COLORS[B].as_bgr())
    below = tuple(int(v) for v in out[BAR_Y + BAR_H, BAR_X + 10])
    assert below not in (COLORS[A].as_bgr(), COLORS[B].as_bgr())
