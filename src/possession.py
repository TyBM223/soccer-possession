"""Ball possession tracking. OWNER-IMPLEMENTED (Ty) - this file is a stub.

Types used below:
    Point  = tuple[float, float]         # (x, y) in frame pixels
    TeamId = int                         # team id from Roboflow's TeamClassifier (0 or 1)
    Call   = TeamId | None               # None means "no call" for that frame

The pipeline passes one entry per frame. Referees are dropped before this
module is called, and goalkeepers are already given a team id.

Open design choices for the owner. The tests deliberately avoid these cases,
so any of the options will pass:
    - Distance exactly equal to max_distance_px: is that a call or not?
    - Two players the same distance from the ball: which one wins?
    - Window alignment: trailing (the last `window` frames, usable live) or
      centered on the frame (needs future frames)?
    - A tied vote inside the window: which team, or no call?
"""

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
            May be empty.
        max_distance_px: The nearest player must be within this Euclidean
            distance of the ball for a call.

    Returns:
        The team id of the player nearest the ball if that player is within
        max_distance_px. Otherwise None (no ball, no players, or everyone too far).
    """
    raise NotImplementedError


def smooth_calls(raw_calls: Sequence[Call], window: int) -> list[Call]:
    """Majority vote of raw calls over a rolling window of `window` frames.

    Args:
        raw_calls: One raw call per frame, in frame order.
        window: Window size in frames (>= 1). window=1 means no smoothing.

    Returns:
        A list the same length as raw_calls. Each entry is the team with the
        most votes among the raw calls in that frame's window. None calls do
        not vote. If the window contains no votes, the entry is None.
    """
    raise NotImplementedError


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
    raise NotImplementedError
