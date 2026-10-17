"""End-to-end possession pipeline: match video in, annotated video and stats out.

Detection, tracking, team classification and ball tracking are Roboflow's
(the sports repo and supervision), set up exactly as in their soccer example's
TEAM_CLASSIFICATION and BALL_DETECTION modes. This module chains them, passes
the results to src/possession.py, and draws src/overlay.py on each frame.

It reads the video twice:
    1. analyze_video() runs the models on every frame and keeps the small
       per-frame results: boxes, tracker ids, teams, ball position.
    2. render_video() reads the video again and draws those results.
The smoothing window is centered, so a frame's call depends on later frames.
That's why drawing waits until every frame has been analyzed.

Usage:
    .venv/bin/python -m src.pipeline \
        --target_video_path outputs/m4/2e57b9_0_possession.mp4
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import supervision as sv
from sports.common.ball import BallAnnotator, BallTracker
from sports.common.team import TeamClassifier
from tqdm import tqdm
from ultralytics import YOLO

from src import config, overlay, possession, roboflow_example
from src.possession import Call, Point, TeamId


@dataclass
class FrameResult:
    """What pass 1 keeps for one frame."""
    people: sv.Detections            # tracked players, goalkeepers and referees
    color_lookup: np.ndarray         # per person: team id (0/1), or REFEREE_CLASS_ID
    ball: sv.Detections              # after Roboflow's BallTracker: 0 or 1 detection
    ball_xy: Optional[Point]         # ball center, or None if no ball detected
    players: list[tuple[Point, TeamId]]  # players and goalkeepers: foot position, team


def fit_team_classifier(
    source_video_path: str, player_model: YOLO, device: str, seed: int
) -> tuple[TeamClassifier, int]:
    """Fit Roboflow's TeamClassifier as their example does: on player crops
    from every STRIDE-th frame. Returns the classifier and the crop count."""
    rf = roboflow_example.load()
    crops = []
    frames = sv.get_video_frames_generator(source_path=source_video_path, stride=rf.STRIDE)
    for frame in tqdm(frames, desc="collecting crops"):
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += rf.get_crops(frame, detections[detections.class_id == rf.PLAYER_CLASS_ID])

    # TeamClassifier creates UMAP and KMeans without a random_state, so both
    # draw from NumPy's global random generator. Seeding it makes the team ids
    # (which team is 0) repeatable between runs, as far as UMAP allows.
    np.random.seed(seed)
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    return team_classifier, len(crops)


def analyze_video(
    source_video_path: str, device: str, seed: int = 0
) -> tuple[list[FrameResult], int]:
    """Pass 1: detect, track and classify on every frame.

    Returns the per-frame results and the number of crops the team classifier
    was fitted on.
    """
    rf = roboflow_example.load()
    player_model = YOLO(rf.PLAYER_DETECTION_MODEL_PATH).to(device=device)
    ball_model = YOLO(rf.BALL_DETECTION_MODEL_PATH).to(device=device)
    team_classifier, n_crops = fit_team_classifier(
        source_video_path, player_model, device, seed)

    # Ball: the same model settings, slicer and tracker as Roboflow's
    # run_ball_detection.
    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=ball_callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )
    ball_tracker = BallTracker(buffer_size=20)

    # People: the same steps as Roboflow's run_team_classification.
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    total = sv.VideoInfo.from_video_path(source_video_path).total_frames
    results = []
    for frame in tqdm(sv.get_video_frames_generator(source_path=source_video_path),
                      total=total, desc="analyzing"):
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == rf.PLAYER_CLASS_ID]
        players_team_id = team_classifier.predict(rf.get_crops(frame, players))
        goalkeepers = detections[detections.class_id == rf.GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = rf.resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)
        referees = detections[detections.class_id == rf.REFEREE_CLASS_ID]

        people = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist()
            + goalkeepers_team_id.tolist()
            + [rf.REFEREE_CLASS_ID] * len(referees),
            dtype=int,
        )

        ball = ball_tracker.update(slicer(frame).with_nms(threshold=0.1))
        ball_xy = None
        if len(ball) > 0:
            x, y = ball.get_anchors_coordinates(sv.Position.CENTER)[0]
            ball_xy = (float(x), float(y))

        # Possession uses players and goalkeepers at their feet; referees are dropped.
        feet = people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        is_player = people.class_id != rf.REFEREE_CLASS_ID
        players_for_possession = [
            ((float(x), float(y)), int(team))
            for (x, y), team in zip(feet[is_player], color_lookup[is_player])
        ]

        results.append(FrameResult(people, color_lookup, ball, ball_xy, players_for_possession))
    return results, n_crops


def compute_calls(
    frames: list[FrameResult], max_distance_px: float, window: int
) -> tuple[list[Call], list[Call]]:
    """Raw and smoothed possession calls for every frame."""
    raw = [possession.raw_call(f.ball_xy, f.players, max_distance_px) for f in frames]
    return raw, possession.smooth_calls(raw, window)


def render_video(
    source_video_path: str,
    target_video_path: str,
    frames: list[FrameResult],
    smoothed: list[Call],
) -> None:
    """Pass 2: draw players, ball and the possession panel on every frame.

    The panel shows possession and coverage for the frames so far, so the last
    frame shows the final numbers.
    """
    rf = roboflow_example.load()
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)  # as in run_ball_detection
    team_colors = {0: sv.Color.from_hex(rf.COLORS[0]), 1: sv.Color.from_hex(rf.COLORS[1])}
    video_info = sv.VideoInfo.from_video_path(source_video_path)

    written = 0
    with sv.VideoSink(target_video_path, video_info) as sink:
        source_frames = sv.get_video_frames_generator(source_path=source_video_path)
        for i, frame in enumerate(tqdm(source_frames, total=len(frames), desc="rendering")):
            if i >= len(frames):
                raise RuntimeError(f"source video has more frames than the {len(frames)} analyzed")
            result = frames[i]
            labels = [str(tracker_id) for tracker_id in _tracker_ids(result.people)]

            annotated = frame.copy()
            annotated = rf.ELLIPSE_ANNOTATOR.annotate(
                annotated, result.people, custom_color_lookup=result.color_lookup)
            annotated = rf.ELLIPSE_LABEL_ANNOTATOR.annotate(
                annotated, result.people, labels, custom_color_lookup=result.color_lookup)
            annotated = ball_annotator.annotate(annotated, result.ball)
            annotated = overlay.draw_possession_overlay(
                annotated, possession.possession_stats(smoothed[: i + 1]), smoothed[i], team_colors)
            sink.write_frame(annotated)
            written += 1

    if written != len(frames):
        raise RuntimeError(f"rendered {written} frames but analyzed {len(frames)}")


def write_details(
    path_stem: Path, frames: list[FrameResult], raw: list[Call], smoothed: list[Call]
) -> None:
    """Per-frame and per-player CSVs, for checking results and for tuning."""
    with path_stem.with_suffix(".frames.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "ball_x", "ball_y", "raw_call", "smoothed_call"])
        for i, (result, r, s) in enumerate(zip(frames, raw, smoothed)):
            bx, by = result.ball_xy if result.ball_xy else ("", "")
            writer.writerow([i, bx, by, "" if r is None else r, "" if s is None else s])

    with path_stem.with_suffix(".players.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "tracker_id", "class_id", "team_or_referee", "foot_x", "foot_y"])
        for i, result in enumerate(frames):
            feet = result.people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            for tracker_id, class_id, team, (x, y) in zip(
                    _tracker_ids(result.people), result.people.class_id, result.color_lookup, feet):
                writer.writerow([i, int(tracker_id), int(class_id), int(team), round(float(x), 1),
                                 round(float(y), 1)])


def _tracker_ids(detections: sv.Detections) -> np.ndarray:
    """Tracker ids, or an empty array when a frame has no people (merging
    only empty detections leaves tracker_id unset)."""
    if detections.tracker_id is None:
        return np.array([], dtype=int)
    return detections.tracker_id


def run(
    source_video_path: str,
    target_video_path: str,
    device: str,
    max_distance_px: float,
    window: int,
    seed: int = 0,
) -> dict:
    """Run both passes, write the video, a JSON summary and the detail CSVs."""
    target = Path(target_video_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    frames, n_crops = analyze_video(source_video_path, device, seed)
    analyze_seconds = time.perf_counter() - start

    raw, smoothed = compute_calls(frames, max_distance_px, window)

    start = time.perf_counter()
    render_video(source_video_path, str(target), frames, smoothed)
    render_seconds = time.perf_counter() - start

    stats = possession.possession_stats(smoothed)
    summary = {
        "source_video_path": str(source_video_path),
        "target_video_path": str(target),
        "device": device,
        "max_distance_px": max_distance_px,
        "window": window,
        "seed": seed,
        "team_classifier_crops": n_crops,
        "total_frames": stats["total_frames"],
        "ball_detected_frames": sum(f.ball_xy is not None for f in frames),
        "raw_called_frames": sum(c is not None for c in raw),
        "smoothed_called_frames": stats["called_frames"],
        "possession_pct": {str(team): pct for team, pct in stats["possession_pct"].items()},
        "coverage_pct": stats["coverage_pct"],
        "analyze_seconds": round(analyze_seconds, 1),
        "render_seconds": round(render_seconds, 1),
    }
    target.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    write_details(target, frames, raw, smoothed)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Possession pipeline: video in, annotated video out.")
    parser.add_argument("--source_video_path", default=str(config.DEFAULT_SOURCE_VIDEO))
    parser.add_argument("--target_video_path", required=True)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--max_distance_px", type=float, default=config.POSSESSION_MAX_DISTANCE_PX,
                        help="Override for this run only; config.py is not changed.")
    parser.add_argument("--window", type=int, default=config.POSSESSION_SMOOTHING_WINDOW,
                        help="Override for this run only; config.py is not changed.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = run(args.source_video_path, args.target_video_path, args.device,
                  args.max_distance_px, args.window, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
