# Rules for this repo
- src/possession.py is written by the owner (Ty). Do not write or rewrite it. You may review it and suggest changes with explanations.
- Do not change POSSESSION_MAX_DISTANCE_PX or POSSESSION_SMOOTHING_WINDOW in src/config.py. The owner tunes these by watching output video.
- Do not train or fine-tune models. Use Roboflow's pretrained weights.
- Never commit weights, videos, or datasets.
- Commit and push after each verified milestone. Never force-push.
- Never claim something works without running it and checking output.
First commit and create the GitHub repo
bash
git add CLAUDE.md && git commit -m "Add project rules"
gh repo create soccer-possession --public --source=. --remote=origin --push

Check github.com. If the repo exists with CLAUDE.md in it, push works.

Check Python and ffmpeg
bash
python3 --version      # 3.10+ ideal
sudo apt install -y python3-venv ffmpeg
nvidia-smi             # optional; error = CPU mode, that's fine
Launch Claude Code
bash
claude

Run /model, pick Opus 5.5, press Shift+Tab until you're in plan mode, then paste the prompt.

The prompt

Starting from an empty repo (only CLAUDE.md). Read CLAUDE.md first and follow it strictly.

PROJECT: Soccer match analysis portfolio piece for a Forward Deployed Engineer application at Roboflow. Built on Roboflow's open-source sports repo (https://github.com/roboflow/sports) and its pretrained weights for player/ball detection, ByteTrack tracking via `supervision`, and team classification (SigLIP embeddings + UMAP + KMeans). The owner's original contribution is possession tracking layered on top.

POSSESSION SPEC (owner writes the implementation - see M2):
- Each frame: if ball detected, find nearest tracked player to ball center; if within POSSESSION_MAX_DISTANCE_PX, that player's team is the raw call for that frame. Otherwise the frame is "no call".
- Smoothed call = majority vote of raw calls over a rolling window of POSSESSION_SMOOTHING_WINDOW frames (no-call frames don't vote; if the window has no votes, smoothed = no call).
- Possession % per team = team's smoothed-call frames / total frames with a smoothed call. No-call frames are excluded, never attributed.
- Coverage % = frames with a smoothed call / total frames.

HARD RULES
- Do not write src/possession.py. That is the owner's job. OWNERSHIP_MODE = A. (If this line says B instead, you may write it, but explain every design choice in comments and in your summary.)
- Do not change the two possession constants after M1 sets initial defaults (80 px, 5 frames).
- Do not train or fine-tune anything.
- Use a Python venv, not Docker, until M6.
- Never claim success without evidence: command exits 0, output file exists and is non-empty, and you extracted frames with ffmpeg and viewed them to confirm annotations. Report what you saw.
- If a fix isn't obvious after two attempts, STOP and show me the error and what you tried.
- After each milestone: run tests, commit with a message stating what was verified and how, push to origin. One milestone per commit. Never force-push. Never commit weights, videos, or anything in data/ or outputs/.
- Credit Roboflow's sports repo (and its license) in the README from the first commit that uses it.

M1 - Scaffold
- .gitignore (venv, __pycache__, *.pt, *.pth, *.onnx, *.mp4, *.avi, data/, outputs/, third_party/).
- Create venv, install Roboflow's sports package and deps into it; pin versions in requirements.txt.
- Clone the sports repo into third_party/ (gitignored) only as needed to run its soccer example and setup script for weights and sample clips.
- Create src/config.py (paths, device auto-detect, possession constants at defaults above), empty src/__init__.py, tests/ folder, README.md skeleton with the phase roadmap (Phase 1 in progress; Phase 2 radar/homography planned; Phase 3 jersey number reading stretch, planned).
- Report GPU vs CPU.
- Commit + push: "M1: scaffold and dependencies".

M2 - Possession tests, then STOP
- Write tests/test_possession.py from the spec above using synthetic detections (no video). Cover at least: single-frame flicker smoothed out; ball missing -> no call, excluded from %; ball too far from all players -> no call; coverage computed correctly; empty window -> no call.
- Write a stub src/possession.py containing ONLY function signatures, docstrings, and `raise NotImplementedError`. Choose clear inputs (e.g. per-frame ball xy or None, player xys with team ids) and document them.
- Tests should currently fail. Commit + push: "M2: possession spec as tests + interface stub".
- STOP. Tell me the interface and how to run the tests. I will implement possession.py. When I say "possession done", run the tests, review my code (bugs, edge cases, clarity - explain, don't rewrite), and commit only if I approve.

M3 - Roboflow baseline, unmodified
- Run Roboflow's soccer example in PLAYER_TRACKING and BALL_DETECTION modes on one sample clip, unmodified.
- Verify with extracted frames. Measure ball detection rate (fraction of frames with a ball detection) and report it.
- Commit a notes file (outputs/ stays ignored): docs/baseline_notes.md. Commit + push: "M3: Roboflow baseline verified".

M4 - Pipeline + overlay
- src/pipeline.py: video -> detection -> tracking -> team classification -> my possession module -> annotated output video. Reuse Roboflow's code; don't reimplement it.
- src/overlay.py: possession bar with team %, plus coverage % visible on frame.
- Run on the sample clip. Verify from frames: boxes/IDs, team colors, ball marker, possession bar, coverage stat. Report final possession % and coverage %.
- Commit + push: "M4: end-to-end pipeline".

M5 - Tuning harness, then STOP
- scripts/tune_sweep.py: runs the pipeline over a grid of the two constants passed as CLI overrides (don't edit config.py), outputs one video per combo plus a CSV of possession % and coverage %. Default small grid (3x3), configurable.
- TUNING_LOG.md with empty table: combo | what I observed | kept?
- Run once on a short segment to prove it works. Commit + push: "M5: tuning harness".
- STOP. I'll watch the videos, fill in the log, set values in config.py, and commit that myself.

M6 - Only after I say "tuning done"
- Add Dockerfile reproducing the venv setup; verify it builds and runs the pipeline.
- README honesty pass: offline frame-by-frame processing (not "live"); team classification is SigLIP+UMAP+KMeans; clearly separate REUSED (Roboflow: detection, tracking, team classification, weights) from OWNER'S (possession logic and its design choices, coverage stat) and BUILT WITH CLAUDE CODE (pipeline integration, overlay, tests, tuning harness, Dockerfile). Include measured numbers and link TUNING_LOG.md. Don't call possession tracking novel; describe the design choices.
- Show me the README diff before committing. Commit + push: "M6: Docker + README reflects verified Phase 1".

At the end of each milestone: 3-5 line summary of what ran, how you verified it, what you changed, and anything surprising.
