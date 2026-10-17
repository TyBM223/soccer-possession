"""Tuning sweep: one annotated video per combination of the two possession settings.

The models (Roboflow's detection, tracking, team classification and ball
tracking) run once. Only the possession calls depend on max_distance_px and
the smoothing window, so each combination recomputes the calls and redraws
the video from the same analysis. That also keeps the team colors the same
in every video of a sweep. config.py is never changed; the values are passed
straight to the pipeline's functions.

The analysis is saved in the output folder, so running again with a different
grid on the same input skips the models.

Usage:
    # 3x3 default grid on seconds 0-10 of the default clip
    .venv/bin/python scripts/tune_sweep.py --output_dir outputs/m5/first10s \
        --start_seconds 0 --duration_seconds 10

    # A different grid on the same segment reuses the saved analysis
    .venv/bin/python scripts/tune_sweep.py --output_dir outputs/m5/first10s \
        --start_seconds 0 --duration_seconds 10 --distances 40 60 80 --windows 1 3 5

Outputs in --output_dir:
    segment.mp4          the cut segment (only with --start_seconds/--duration_seconds)
    analysis.pkl         the saved model results
    d80px_w5.mp4, ...    one video per combination
    sweep_results.csv    possession % and coverage % for each combination,
                         including combinations from earlier runs in this folder
"""

import argparse
import csv
import pickle
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import config, pipeline, possession  # noqa: E402

DEFAULT_DISTANCES = [60.0, 80.0, 100.0]
DEFAULT_WINDOWS = [3, 5, 9]


def cut_segment(source: Path, target: Path, start: float, duration: float) -> None:
    """Re-encode the segment so it starts exactly at `start`. A stream copy
    would snap the start to the nearest keyframe."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i", str(source),
         "-t", str(duration), "-c:v", "libx264", "-crf", "18", "-an", str(target)],
        check=True,
    )


def load_or_analyze(video: Path, cache_path: Path, meta: dict, fresh: bool):
    """Run pass 1 of the pipeline, or load it from cache_path if it was made
    for the same input and settings."""
    if cache_path.exists() and not fresh:
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
        if cached["meta"] != meta:
            raise SystemExit(
                f"{cache_path} was made for {cached['meta']}, not {meta}. "
                "Pass --fresh or use another --output_dir.")
        print(f"reusing analysis from {cache_path}")
        return cached["frames"], cached["n_crops"]

    start = time.perf_counter()
    frames, n_crops = pipeline.analyze_video(str(video), meta["device"], meta["seed"])
    print(f"analysis: {len(frames)} frames in {time.perf_counter() - start:.0f} s")
    with cache_path.open("wb") as f:
        pickle.dump({"meta": meta, "frames": frames, "n_crops": n_crops}, f)
    return frames, n_crops


def combo_name(distance: float, window: int) -> str:
    return f"d{distance:g}px_w{window}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source_video_path", type=Path, default=config.DEFAULT_SOURCE_VIDEO)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--start_seconds", type=float,
                        help="Cut a segment starting here (needs --duration_seconds).")
    parser.add_argument("--duration_seconds", type=float)
    parser.add_argument("--distances", type=float, nargs="+", default=DEFAULT_DISTANCES,
                        help="max_distance_px values, in pixels.")
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS,
                        help="Smoothing window sizes, in frames.")
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fresh", action="store_true",
                        help="Run the models again even if a saved analysis exists.")
    args = parser.parse_args()

    if (args.start_seconds is None) != (args.duration_seconds is None):
        parser.error("--start_seconds and --duration_seconds go together")
    if any(w < 1 for w in args.windows):
        parser.error("windows must be >= 1")
    if any(d <= 0 for d in args.distances):
        parser.error("distances must be > 0")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    meta = {"source": str(args.source_video_path), "start_seconds": args.start_seconds,
            "duration_seconds": args.duration_seconds, "device": args.device, "seed": args.seed}

    video = args.source_video_path
    if args.start_seconds is not None:
        video = out / "segment.mp4"
        cache_ok = (out / "analysis.pkl").exists() and video.exists() and not args.fresh
        if not cache_ok:
            cut_segment(args.source_video_path, video, args.start_seconds, args.duration_seconds)

    frames, n_crops = load_or_analyze(video, out / "analysis.pkl", meta, args.fresh)
    print(f"team classifier fitted on {n_crops} crops")

    rows = []
    for distance in args.distances:
        for window in args.windows:
            name = combo_name(distance, window)
            raw, smoothed = pipeline.compute_calls(frames, distance, window)
            target = out / f"{name}.mp4"
            pipeline.render_video(str(video), str(target), frames, smoothed)
            stats = possession.possession_stats(smoothed)
            pct = stats["possession_pct"]
            rows.append({
                "combo": name,
                "max_distance_px": f"{distance:g}",
                "window": window,
                # A team with no calls has 0%; with no calls at all, possession is undefined.
                "team0_possession_pct": round(pct.get(0, 0.0), 1) if pct else "",
                "team1_possession_pct": round(pct.get(1, 0.0), 1) if pct else "",
                "coverage_pct": round(stats["coverage_pct"], 1),
                "raw_called_frames": sum(c is not None for c in raw),
                "smoothed_called_frames": stats["called_frames"],
                "total_frames": stats["total_frames"],
                "video": str(target),
            })
            r = rows[-1]
            print(f"{name}: team0 {r['team0_possession_pct']}%  team1 {r['team1_possession_pct']}%  "
                  f"coverage {r['coverage_pct']}%")

    # Keep rows from earlier sweeps in this folder; a combination run again
    # replaces its old row.
    results = out / "sweep_results.csv"
    by_combo = {}
    if results.exists():
        with results.open(newline="") as f:
            by_combo = {r["combo"]: r for r in csv.DictReader(f)}
    by_combo.update({r["combo"]: r for r in rows})
    ordered = sorted(by_combo.values(),
                     key=lambda r: (float(r["max_distance_px"]), int(r["window"])))
    with results.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(ordered)
    print(f"results: {results}")


if __name__ == "__main__":
    main()
