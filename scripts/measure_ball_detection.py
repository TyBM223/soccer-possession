"""Run Roboflow's BALL_DETECTION mode on a video and measure the ball detection rate.

Roboflow's run_ball_detection() from third_party/sports/examples/soccer/main.py
runs unchanged. Two things differ from running main.py directly:
    - The cv2.imshow preview is skipped. Frames are written with sv.VideoSink
      the same way main() writes them.
    - BallTracker is swapped for a pass-through subclass that records how many
      ball detections went into and came out of each update() call. It returns
      exactly what BallTracker.update() returns.

Ball detection rate = frames with at least one ball detection (after Roboflow's
slicer and NMS, before its BallTracker) / total frames.

Usage:
    .venv/bin/python scripts/measure_ball_detection.py \
        --target_video_path outputs/m3/2e57b9_0_ball_detection.mp4
"""

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import supervision as sv

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402


def load_roboflow_example():
    """Import Roboflow's soccer example main.py as a module, without running it."""
    path = config.SOCCER_EXAMPLE_DIR / "main.py"
    spec = importlib.util.spec_from_file_location("roboflow_soccer_main", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source_video_path", type=Path, default=config.DEFAULT_SOURCE_VIDEO)
    parser.add_argument("--target_video_path", type=Path, required=True)
    parser.add_argument(
        "--counts_csv", type=Path,
        help="Per-frame detection counts. Default: next to the target video.")
    parser.add_argument("--device", default=config.DEVICE)
    args = parser.parse_args()
    counts_csv = args.counts_csv or args.target_video_path.with_suffix(".counts.csv")

    roboflow = load_roboflow_example()
    counts: list[tuple[int, int]] = []  # (detections in, detections out) per frame

    class CountingBallTracker(roboflow.BallTracker):
        def update(self, detections: sv.Detections) -> sv.Detections:
            result = super().update(detections)
            counts.append((len(detections), len(result)))
            return result

    # run_ball_detection() looks BallTracker up in its module when it is called.
    roboflow.BallTracker = CountingBallTracker

    args.target_video_path.parent.mkdir(parents=True, exist_ok=True)
    video_info = sv.VideoInfo.from_video_path(str(args.source_video_path))
    frames = 0
    start = time.perf_counter()
    with sv.VideoSink(str(args.target_video_path), video_info) as sink:
        for frame in roboflow.run_ball_detection(
                source_video_path=str(args.source_video_path), device=args.device):
            sink.write_frame(frame)
            frames += 1
    elapsed = time.perf_counter() - start

    if len(counts) != frames:
        raise RuntimeError(f"{len(counts)} BallTracker updates for {frames} frames")

    with counts_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "ball_detections", "after_ball_tracker"])
        writer.writerows((i, n_in, n_out) for i, (n_in, n_out) in enumerate(counts))

    with_ball = sum(1 for n_in, _ in counts if n_in > 0)
    multiple = sum(1 for n_in, _ in counts if n_in > 1)
    rate = 100.0 * with_ball / frames if frames else 0.0
    print(f"source: {args.source_video_path}")
    print(f"frames: {frames} (video reports {video_info.total_frames})")
    print(f"frames with >=1 ball detection: {with_ball} ({rate:.1f}%)")
    print(f"frames with >1 ball detection: {multiple}")
    print(f"seconds: {elapsed:.0f} ({elapsed / max(frames, 1):.2f} s/frame)")
    print(f"video: {args.target_video_path}")
    print(f"counts: {counts_csv}")


if __name__ == "__main__":
    main()
