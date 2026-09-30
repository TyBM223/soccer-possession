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
