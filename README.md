# soccer-possession

Soccer match analysis built on top of Roboflow's open-source
[sports](https://github.com/roboflow/sports) repo. Roboflow's pretrained models
and code handle player/ball detection, tracking, and team classification. This
project adds ball possession tracking on top of them.

> Status: Phase 1 in progress. The numbers and claims in this README will be
> filled in only after they have been measured.

## Roadmap

- **Phase 1: Possession tracking** (in progress). Detection, tracking and team
  classification come from Roboflow. Possession % per team plus a coverage stat
  are drawn onto the output video.
- **Phase 2: Radar / homography** (planned). Map players and the ball onto a
  top-down pitch view.
- **Phase 3: Jersey number reading** (stretch, planned).

## Setup

Requires Python 3.10+ (developed on 3.14) and ffmpeg. Runs on CPU; a GPU is optional.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Roboflow's sports repo: provides the setup script for weights and sample clips
git clone https://github.com/roboflow/sports.git third_party/sports
PATH="$PWD/.venv/bin:$PATH" bash third_party/sports/examples/soccer/setup.sh

.venv/bin/python -m pytest
```

Weights, videos, `third_party/`, `data/` and `outputs/` are gitignored.

## How possession is computed

`src/possession.py` has three functions, run in order on per-frame detections.

### `raw_call(ball_xy, players, max_distance_px)`

One frame's call, before smoothing.

- The player nearest the ball gets the call for their team if they are within
  `max_distance_px`. Distance is straight-line, from the ball center to the
  player's foot position, since that is where the ball is played.
- No ball detected, no players, or nobody in range: no call. A ball in flight
  always has a nearest player but nobody in possession, so the distance limit
  keeps those frames from being credited to whoever happens to be closest.
- A distance exactly at the limit counts as a call.
- Players from different teams exactly tied for nearest: no call, since the
  ball is contested. Tied players from the same team: that team.

### `smooth_calls(raw_calls, window)`

Removes one-frame flickers from the raw calls.

- Each frame takes the majority vote of the raw calls in a window of `window`
  frames.
- No-call frames don't vote. A short ball dropout doesn't erase possession,
  but a dropout longer than the window becomes no call.
- The window is centered on the frame. Processing is offline, so future frames
  are available, and centering keeps the call lined up with the frame it is
  drawn on. A trailing window would lag by about half a window. Windows are
  clipped at the start and end of the video.
- No votes, or a tied vote: no call. Without a majority the frame is
  uncertain, so it isn't given to either team.

### `possession_stats(smoothed_calls)`

- Possession % per team = that team's frames / frames with a call. No-call
  frames are excluded, never attributed to a team.
- Coverage % = frames with a call / total frames. A weak ball detector shows up
  as low coverage instead of being hidden behind confident possession numbers.

The two constants start at 80 px and 5 frames in `src/config.py` and have not
been tuned yet.

## Credits and licenses

- **[roboflow/sports](https://github.com/roboflow/sports)** (MIT License,
  Copyright (c) 2024 Roboflow). Provides the soccer example, the pretrained
  player/ball/pitch detection weights, the sample clips, ball tracking
  utilities, and the team classifier (SigLIP embeddings + UMAP + KMeans).
- **[supervision](https://github.com/roboflow/supervision)** (MIT). Provides
  ByteTrack tracking and the annotators.
- **[ultralytics](https://github.com/ultralytics/ultralytics)** (AGPL-3.0). The
  YOLOv8 runtime the detection weights use.
- The sample clips come from the
  [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout)
  Kaggle competition, via Roboflow.
