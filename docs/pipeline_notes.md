# M4: end-to-end pipeline results

Run on 2026-10-16 with `src/pipeline.py` at the default settings
(80 px, 5-frame window, seed 0). Outputs are in `outputs/m4/` (gitignored).

```bash
.venv/bin/python -m src.pipeline \
  --source_video_path third_party/sports/examples/soccer/data/2e57b9_0.mp4 \
  --target_video_path outputs/m4/2e57b9_0_possession.mp4
```

## What the pipeline does

Each frame goes through Roboflow's code, set up as in their soccer example:
player detection, ByteTrack tracking, team classification (SigLIP embeddings +
UMAP + KMeans, fitted on player crops from every 60th frame), goalkeeper team
assignment, and ball detection with their slicer and `BallTracker`. Referees
are dropped, and each player's foot position and team go to
`src/possession.py`. The output video shows Roboflow's ellipses and IDs, their
ball marker, and a possession panel.

The panel shows possession and coverage for the frames so far, so the last
frame shows the final numbers.

## Results (clip 2e57b9_0, 750 frames, 30 s)

| | |
|---|---|
| Exit code | 0, 750 output frames |
| Team classifier fitted on | 266 player crops |
| Frames with a ball detection | 687 (91.6%, same as M3) |
| Frames with a raw call | 545 (72.7%) |
| Frames with a smoothed call | 652 |
| **Possession** | **dark-kit team 34.7%, white-kit team 65.3%** |
| **Coverage** | **86.9%** |
| Time on CPU | 7030 s analyzing (9.4 s/frame), 12 s drawing |

The team ids are clusters, not names. In this run team 0 (pink) is the
dark-kit team and team 1 (blue) is the white-kit team. Another run could swap
them.

These numbers have not been checked against hand-labeled possession. They
show what the pipeline computes, not that it is accurate.

## Verification

Frames 40, 300, 375, 700 and 749 were extracted with ffmpeg and viewed.

- Every outfield player has an ellipse and a tracker ID. Referees are yellow.
- In frames 40, 375 and 700, white-kit players are blue and dark-kit players
  are pink. The exception is where two players overlap, for example IDs 9 and
  17 in frame 40.
- The two goalkeepers are on opposite teams, each with the team defending
  their goal. The red-kit keeper at the left goal is pink, and the yellow-kit
  keeper at the right goal (frame 700) is blue.
- The ball marker is on the ball at the red-kit keeper's feet in frame 40, and
  between IDs 16 and 12 in frame 700.
- The panel shows "no call yet" in gray before the first call, then the split
  bar, the coverage, and a dot in the current team's color or "no call". The
  last frame shows 35% / 65% and coverage 87%, matching the summary above.

## Team labels that change between frames

Roboflow's classifier labels every player crop independently in every frame,
so one player's team can change from frame to frame.

- 7.6% of player-frames (1134 of 14992) give a player the team that player has
  less often. 12 of the 26 player IDs seen in 5 or more frames keep one team
  throughout.
- The worst are IDs 22 and 8, wrong in about 38% of frames. They spend much of
  the clip standing close together (see frames 40 and 375), so their crops
  overlap.
- The players who have the ball most are mostly stable. ID 12 is the
  exception: it has the ball in 118 frames and is labeled with its less
  common team in 29 of them.

As a check (not in the pipeline), giving each player ID its most common team
for the whole clip changes the result to 30.3% / 69.7%, with coverage 87.9%.

On the 2-second test cut, the classifier was fitted on only 21 crops, and the
player with the ball switched teams in 17 of the 42 frames where he had it.
With 266 crops on the full clip, the same player is stable (90 of 92 frames).
Short clips give the classifier too little to fit on.

## Warnings in the log

- `Mean of empty slice` and `invalid value encountered in divide`, once at
  the start. ByteTrack only reports a player after 3 frames, so frames 0-2
  have no players yet, and Roboflow's goalkeeper helper averages an empty list.
  No goalkeeper is assigned in those frames.
- SigLIP load report: the checkpoint also has a text model, which
  `SiglipVisionModel` doesn't use.
- supervision deprecation warnings for the two older slicer arguments the
  example uses (see `docs/baseline_notes.md`).
