# M3 baseline: Roboflow's soccer example, unmodified

Run on 2026-10-16. Output videos are in `outputs/m3/` (gitignored).

## Setup

- Roboflow [`sports`](https://github.com/roboflow/sports) at commit `42c80c0`,
  `examples/soccer/main.py`, unmodified, with the pretrained
  `football-player-detection.pt` and `football-ball-detection.pt` that its
  `setup.sh` downloads.
- Clip: `2e57b9_0.mp4`, one of the five sample clips. 1920x1080, 25 fps,
  750 frames (30 s).
- CPU only (no NVIDIA GPU): WSL2 Ubuntu, 16 logical cores, PyTorch 2.14.1 using
  8 threads. Python 3.14, ultralytics 8.4.170, supervision 0.23.0.

## supervision pinned to 0.23.0

M1 installed supervision 0.30.6. With it, `BALL_DETECTION` crashed before the
first frame:

```
TypeError: InferenceSlicer.__init__() got an unexpected keyword argument 'overlap_filter_strategy'
```

The example passes `overlap_filter_strategy=`, which supervision 0.24.0 removed.
0.23.0 is the newest release that still accepts it (with a deprecation
warning), so `requirements.txt` now pins it. It is installed with `--no-deps`:
0.23.0 lists `opencv-python-headless` as a dependency, and that build has no
GUI, which would break the example's `cv2.imshow` preview. `pip check`
therefore reports `opencv-python-headless` as missing. That is expected.

## How each mode was run

`PLAYER_TRACKING` through Roboflow's own command line, unmodified:

```bash
.venv/bin/python third_party/sports/examples/soccer/main.py \
  --source_video_path third_party/sports/examples/soccer/data/2e57b9_0.mp4 \
  --target_video_path outputs/m3/2e57b9_0_player_tracking.mp4 \
  --device cpu --mode PLAYER_TRACKING
```

`BALL_DETECTION` through `scripts/measure_ball_detection.py`. It runs Roboflow's
`run_ball_detection()` unchanged and writes frames the way `main()` does,
without the `cv2.imshow` preview. A pass-through `BallTracker` subclass counts
the detections in each frame and returns exactly what Roboflow's tracker
returns.

```bash
.venv/bin/python scripts/measure_ball_detection.py \
  --source_video_path third_party/sports/examples/soccer/data/2e57b9_0.mp4 \
  --target_video_path outputs/m3/2e57b9_0_ball_detection.mp4
```

On a 2-second (50-frame) cut of the same clip, this script's video was
pixel-identical to the video from `main.py --mode BALL_DETECTION`: the maximum
pixel difference across all 50 frames was 0. Ball detection is the slow mode,
so the full clip was run once, through the script.

## Results (full clip, 750 frames)

|                | PLAYER_TRACKING        | BALL_DETECTION         |
|----------------|------------------------|------------------------|
| Exit code      | 0                      | 0                      |
| Output frames  | 750                    | 750                    |
| Output size    | 68.8 MB                | 64.3 MB                |
| Time on CPU    | 1283 s (1.71 s/frame)  | 4278 s (5.70 s/frame)  |

**Ball detection rate: 687 of 750 frames (91.6%)** have at least one ball
detection, counted after Roboflow's slicer and NMS and before its `BallTracker`.

- 136 frames (18.1%) have more than one ball detection, up to 4. Roboflow's
  `BallTracker` keeps the one closest to the average of recent positions.
- The 63 frames with no detection fall in 29 gaps. The longest is 10 frames
  (0.4 s), and 27 of the 29 gaps are 4 frames or shorter.
- This is a detection rate, not an accuracy. A detection can be a false
  positive (see frame 375 below), so 91.6% is an upper bound on the frames
  where the ball is correctly found.

Per-frame counts: `outputs/m3/2e57b9_0_ball_detection.counts.csv`.

## Verification

Frames were extracted with ffmpeg and viewed.

`PLAYER_TRACKING`, frames 375 and 700:

- In both frames every outfield player has an ellipse at the feet and a
  tracker ID. One partly visible figure at the right edge of frame 375 has
  none. Players are orange-red, referees yellow and the goalkeeper (frame 700)
  blue. These are the example's per-class colors. This mode does not classify
  teams.
- The referee has ID 3 and an assistant referee on the far touchline has ID 21
  in both frames, 13 seconds apart.

`BALL_DETECTION`, zoomed crops of frames 14, 30, 375 and 700:

- Frames 14 and 700: the newest marker is on the ball at a player's feet.
- Frame 30 has no detection, but the example still draws a trail, because its
  `BallAnnotator` keeps the last 10 positions. A marker on screen does not by
  itself mean the ball was found in that frame.
- Frame 375 has 2 detections, and the one the tracker kept sits at knee height
  on a player lifting his leg near the touchline. In the original frames
  373-377 I could not see a ball at that spot, so this is possibly a false
  positive.

## Notes for M4

- Possession coverage will be lower than 91.6%, because a frame also needs a
  tracked player within the distance limit of the ball.
- With the 5-frame centered window, a ball gap of 4 frames or fewer can still
  get a smoothed call if the frames around it have calls.
- Ball detection dominates run time on this CPU (5.7 s/frame), so one full M4
  run on a 30-second clip will take over an hour here.
