# Tuning log

The two possession settings are tuned by watching output video:

- `POSSESSION_MAX_DISTANCE_PX`: how close (in pixels) the nearest player must
  be to the ball for a possession call.
- `POSSESSION_SMOOTHING_WINDOW`: how many frames vote on each frame's call.

Watch each video, note what you see below, then put the values you keep in
`src/config.py`.

## Making the videos

```bash
.venv/bin/python scripts/tune_sweep.py --output_dir outputs/m5/first10s \
  --start_seconds 0 --duration_seconds 10
```

Each video is named for its settings: `d80px_w5.mp4` is 80 px and a 5-frame
window. `sweep_results.csv` in the same folder has each one's possession % and
coverage %. All videos in one sweep share one analysis, so the team colors
(pink and blue) mean the same team in all of them.

From Windows, the folder is
`\\wsl$\Ubuntu\home\tbm223\projects\soccer-possession\outputs\m5\first10s\`.

## What to look for

- The "Now" dot should follow the team on the ball and switch when the other
  team wins it.
- A pass or clearance in the air should show "no call", not the nearest
  player's team.
- Distance too small: the dot drops to "no call" while a player is dribbling.
  Too large: a call goes to a player who is only near the ball.
- Window too small: the dot flickers between teams. Too large: short spells
  of possession disappear, and the dot shows the wrong team for a few frames
  around a switch.

## Observations

Segment: clip `2e57b9_0`, seconds 0-10 (frames 0-249), in
`outputs/m5/first10s/`. In this sweep, pink is the white-kit team and blue is
the dark-kit team, including the red-kit goalkeeper.

| combo | what I observed | kept? |
|---|---|---|
| d60px_w3 | | |
| d60px_w5 | | |
| d60px_w9 | | |
| d80px_w3 | | |
| d80px_w5 | | |
| d80px_w9 | | |
| d100px_w3 | | |
| d100px_w5 | | |
| d100px_w9 | | |
