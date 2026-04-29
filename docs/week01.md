# Week 01 — Research Notes
**Date:** April 2026  
**Goal:** BC baseline on all 5 Franka tasks with early stopping

---

## What We Did

1. Upgraded RobotZooDataset to RobotZooStepDataset — step level training
2. Added early stopping to training loop (patience=10, target=0.0001)
3. Trained BC on all 5 Franka tasks successfully
4. Committed all 5 checkpoints to GitHub

---

## Problems We Hit

### Problem 1 — ROS conflicting with scripts import
**What happened:** `from scripts.train_bc import train` triggered
ROS Humble's own scripts module instead of ours.  
**Root cause:** ROS adds its own packages to Python path including
a module called `scripts`. Name collision.  
**Fix:** Used `importlib.util.spec_from_file_location` to load
the file directly by path, bypassing Python's import system.

### Problem 2 — Fixed epochs wasted compute
**What happened:** Training pick_cube for 50 epochs when it
converged at epoch 9. 41 wasted epochs per task.  
**Fix:** Added early stopping — stops when loss hits 0.0001
or no improvement for 10 consecutive epochs.  
**Result:** Push trained in 2 epochs. Pick in 8. No wasted compute.

---

## Results Table

| Task | Epochs | Best Loss | Converged |
|------|--------|-----------|-----------|
| push | 2 | 0.000094 | ✅ |
| pick_cube | 8 | 0.000094 | ✅ |
| place | 17 | 0.000100 | ✅ |
| open_drawer | 100 | 0.000098 | ✅ |
| reach | 500 | 0.000564 | ❌ never hit target |

---

## Key Research Finding — Reach is Fundamentally Different

reach ran all 500 epochs and never hit the 0.0001 target.
Final loss: 0.000564 — 5x higher than all other tasks.

**Why:** Reach has multimodal demonstrations. Multiple valid
arm trajectories exist to reach the same target point. BC
sees conflicting demonstrations for similar observations
and cannot converge — it tries to average all valid paths
which produces a blurry, incorrect prediction.

**Hypothesis:** ACT will show the largest improvement over
BC specifically on reach. ACT's CVAE encoder models the
full distribution of valid actions instead of averaging them.
This is the most interesting comparison in Week 2.

**Paper implication:** This is evidence that task difficulty
for BC is not just about physical complexity but about
demonstration multimodality. Push is physically harder
than reach but BC learns it in 2 epochs because there is
only one valid push trajectory.

---

## Epoch Analysis — How Fast Did Each Task Learn?
push:        █ (2 epochs)
pick_cube:   ████ (8 epochs)
place:       ████████ (17 epochs)
open_drawer: ████████████████████████████████████████ (100 epochs)
reach:       ████████████████████████████████████████████████████ (500, incomplete)


Complexity order: push < pick < place < open_drawer << reach

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Total training samples | ~1.6M steps across 5 tasks |
| Fastest convergence | push — 2 epochs |
| Slowest convergence | reach — 500 epochs, incomplete |
| Training time per task | 5-120 seconds |
| Checkpoints saved | 5 |

---

## Questions for Next Week

- Does BC loss=0.0001 actually translate to success in Isaac Lab?
- Does reach's higher loss mean lower success rate in simulation?
- Will ACT close the gap on reach specifically?
- What success rate does BC achieve on franka/pick_cube?

---

## What's Next — Week 02

- Isaac Lab evaluation script — get real success rate numbers
- Train ACT on all 5 Franka tasks
- Compare BC vs ACT — especially on reach
- Start Stretch policies
