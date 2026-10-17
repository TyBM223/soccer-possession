"""Possession scoreboard drawn in the top-left corner of each output frame.

The panel shows:
    - a bar split by each team's possession % so far, in the same colors as
      the team ellipses under the players
    - coverage so far: the % of frames that have a possession call
    - this frame's call: a dot in the team's color, or "no call"
"""

from typing import Mapping

import cv2
import numpy as np
import supervision as sv

from src.possession import Call, TeamId

# Panel and bar geometry, in pixels.
PANEL_X, PANEL_Y, PANEL_W, PANEL_H = 40, 30, 560, 135
BAR_X, BAR_Y, BAR_W, BAR_H = PANEL_X + 20, PANEL_Y + 50, PANEL_W - 40, 40

PANEL_COLOR = (20, 20, 20)  # BGR
PANEL_OPACITY = 0.65
NO_CALL_COLOR = sv.Color(r=120, g=120, b=120)
TEXT_COLOR = (255, 255, 255)
OUTLINE_COLOR = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
MIN_SEGMENT_W_FOR_LABEL = 70  # narrower segments are left unlabeled


def draw_possession_overlay(
    frame: np.ndarray,
    stats: dict,
    current_call: Call,
    team_colors: Mapping[TeamId, sv.Color],
) -> np.ndarray:
    """Draw the possession panel onto a copy of the frame.

    Args:
        frame: BGR image, at least (PANEL_X + PANEL_W) x (PANEL_Y + PANEL_H).
            It is not modified.
        stats: possession_stats() for the frames up to and including this one.
        current_call: This frame's smoothed call, a team id or None.
        team_colors: The color for each team id.

    Returns:
        The annotated copy.
    """
    out = frame.copy()

    # Translucent background.
    panel = out[PANEL_Y:PANEL_Y + PANEL_H, PANEL_X:PANEL_X + PANEL_W]
    background = np.full_like(panel, PANEL_COLOR)
    out[PANEL_Y:PANEL_Y + PANEL_H, PANEL_X:PANEL_X + PANEL_W] = cv2.addWeighted(
        background, PANEL_OPACITY, panel, 1 - PANEL_OPACITY, 0)

    _text(out, "Possession", (BAR_X, PANEL_Y + 35), scale=0.8)
    _draw_bar(out, stats["possession_pct"], team_colors)

    bottom_y = BAR_Y + BAR_H + 32
    _text(out, f"Coverage {stats['coverage_pct']:.0f}% of frames", (BAR_X, bottom_y), scale=0.7)
    now_x = BAR_X + BAR_W - 150
    _text(out, "Now:", (now_x, bottom_y), scale=0.7)
    if current_call is None:
        _text(out, "no call", (now_x + 65, bottom_y), scale=0.7)
    else:
        color = team_colors.get(current_call, NO_CALL_COLOR).as_bgr()
        cv2.circle(out, (now_x + 85, bottom_y - 8), 12, color, -1, cv2.LINE_AA)
        cv2.circle(out, (now_x + 85, bottom_y - 8), 12, TEXT_COLOR, 2, cv2.LINE_AA)
    return out


def _draw_bar(
    out: np.ndarray,
    possession_pct: Mapping[TeamId, float],
    team_colors: Mapping[TeamId, sv.Color],
) -> None:
    """One segment per team, left to right by team id, widths by possession %."""
    if not possession_pct:
        _fill(out, BAR_X, BAR_W, NO_CALL_COLOR)
        _text_centered(out, "no call yet", BAR_X, BAR_W)
        return

    teams = sorted(possession_pct)
    x = BAR_X
    for i, team in enumerate(teams):
        # The last segment takes the remainder, so rounding never leaves a gap.
        width = BAR_X + BAR_W - x if i == len(teams) - 1 else round(BAR_W * possession_pct[team] / 100)
        _fill(out, x, width, team_colors.get(team, NO_CALL_COLOR))
        if width >= MIN_SEGMENT_W_FOR_LABEL:
            _text_centered(out, f"{possession_pct[team]:.0f}%", x, width)
        x += width


def _fill(out: np.ndarray, x: int, width: int, color: sv.Color) -> None:
    cv2.rectangle(out, (x, BAR_Y), (x + width - 1, BAR_Y + BAR_H - 1), color.as_bgr(), -1)


def _text(out: np.ndarray, text: str, origin: tuple[int, int], scale: float) -> None:
    """White text with a black outline, readable on any background."""
    cv2.putText(out, text, origin, FONT, scale, OUTLINE_COLOR, 4, cv2.LINE_AA)
    cv2.putText(out, text, origin, FONT, scale, TEXT_COLOR, 2, cv2.LINE_AA)


def _text_centered(out: np.ndarray, text: str, x: int, width: int) -> None:
    scale = 0.8
    (text_w, text_h), _ = cv2.getTextSize(text, FONT, scale, 2)
    _text(out, text, (x + (width - text_w) // 2, BAR_Y + (BAR_H + text_h) // 2), scale)
