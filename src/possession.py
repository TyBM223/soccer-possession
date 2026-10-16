"""Ball possession tracking.

The nearest player within a distance limit gets the call, calls are smoothed
by a rolling majority vote, no-call frames are excluded rather than
attributed, and a coverage stat reports how much of the video has a call.

Types used below:
    Point  = tuple[float, float]         # (x, y) in frame pixels
    TeamId = int                         # team id from Roboflow's TeamClassifier (0 or 1)
    Call   = TeamId | None               # None means "no call" for that frame

The pipeline passes one entry per frame. Referees are dropped before this
module is called, and goalkeepers are already given a team id.

Edge-case choices. The tests deliberately don't pin these down, so changing
one here won't break them:
    - Distance exactly equal to max_distance_px is a call (<=).
    - Players tied for nearest: same team -> that team; different teams ->
      the ball is contested, so no call.
    - The smoothing window is centered on the frame. Processing is offline, so
      future frames are available, and centering keeps the call lined up with
      the frame it is drawn on (a trailing window lags about half a window).
      For an even window the extra frame comes from the future. Windows are
      clipped at the start and end of the video.
    - A tied vote inside the window is no call: there is no majority, so the
      frame is treated as uncertain instead of being given to either team.
"""

import math
from collections import Counter
from typing import Optional, Sequence

Point = tuple[float, float]
TeamId = int
Call = Optional[TeamId]


def raw_call(
    ball_xy: Optional[Point],
    players: Sequence[tuple[Point, TeamId]],
    max_distance_px: float,
) -> Call:
    """Per-frame possession call before smoothing.

    Args:
        ball_xy: Center of the detected ball in pixels, or None if no ball was
            detected in this frame.
        players: One (xy, team_id) entry per tracked player in this frame.
            May be empty. xy should be the player's foot position (bottom
            center of the box), since that is where the ball is played.
        max_distance_px: The nearest player must be within this Euclidean
            distance of the ball for a call.

    Returns:
        The team id of the player nearest the ball if that player is within
        max_distance_px. Otherwise None (no ball, no players, everyone too far,
        or nearest players from both teams exactly tied).
    """
    if ball_xy is None or not players:
        return None

    distances = [(math.dist(ball_xy, xy), team) for xy, team in players]
    nearest = min(d for d, _ in distances)
    if nearest > max_distance_px:
        return None

    nearest_teams = {team for d, team in distances if d == nearest}
    if len(nearest_teams) > 1:
        return None
    return nearest_teams.pop()


def smooth_calls(raw_calls: Sequence[Call], window: int) -> list[Call]:
    """Majority vote of raw calls over a rolling window of `window` frames.

    Args:
        raw_calls: One raw call per frame, in frame order.
        window: Window size in frames (>= 1). window=1 means no smoothing.

    Returns:
        A list the same length as raw_calls. Each entry is the team with the
        most votes among the raw calls in that frame's window (centered on the
        frame). None calls do not vote. If the window contains no votes, or
        the top two teams are tied, the entry is None.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    n = len(raw_calls)
    before = (window - 1) // 2  # frames before the current one in its window
    smoothed: list[Call] = []
    for i in range(n):
        start = max(0, i - before)
        end = min(n, i - before + window)
        votes = Counter(c for c in raw_calls[start:end] if c is not None)
        smoothed.append(_majority(votes))
    return smoothed


def _majority(votes: Counter) -> Call:
    """The team with the most votes, or None if there are no votes or a tie."""
    top = votes.most_common(2)
    if not top:
        return None
    if len(top) == 2 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def possession_stats(smoothed_calls: Sequence[Call]) -> dict:
    """Possession and coverage percentages from smoothed calls.

    Args:
        smoothed_calls: One smoothed call per frame, in frame order.

    Returns:
        A dict with these keys:
            "possession_pct": dict[TeamId, float]. For each team that has at
                least one smoothed call: 100 * that team's frames / frames with
                a call. None frames are excluded, never attributed. The dict is
                empty if no frame has a call.
            "coverage_pct": float. 100 * frames with a call / total frames.
                0.0 if there are no frames.
            "called_frames": int. The number of frames with a call.
            "total_frames": int. len(smoothed_calls).
    """
    total = len(smoothed_calls)
    team_frames = Counter(c for c in smoothed_calls if c is not None)
    called = sum(team_frames.values())

    return {
        "possession_pct": {
            team: 100.0 * count / called for team, count in sorted(team_frames.items())
        },
        "coverage_pct": 100.0 * called / total if total else 0.0,
        "called_frames": called,
        "total_frames": total,
    }
