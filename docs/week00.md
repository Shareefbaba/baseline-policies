# Week 00 — Research Notes
**Date:** April 2026  
**Goal:** Environment setup, data loader, first BC training run

---

## What We Did

1. Verified LeRobot 0.4.2 + PyTorch 2.10 + CUDA on RTX 5060 Ti
2. Wrote RobotZooDataset — loads full episodes from .npz files
3. Ran 4 data sanity checks
4. Wrote RobotZooStepDataset — flattens episodes into individual timesteps
5. Wrote BCPolicy — 3-layer MLP, 139,272 parameters
6. Wrote train_bc.py — full training loop with epochs, optimizer, checkpointing
7. Trained first BC policy on franka/pick_cube — loss converged to 0.0000

---

## Problems We Hit

### Problem 1 — Markdown links corrupting filenames
**What happened:** Claude's responses had hyperlinks like
`robot_zoo_[dataset.py](http://dataset.py)` which corrupted
filenames when pasted into terminal.  
**Fix:** Used `nano` to manually create files instead of
`cat >> EOF` piping. Avoided copy-pasting filenames from chat.

### Problem 2 — Training too slow (30 minutes for 50 epochs)
**What happened:** RobotZooStepDataset was opening .npz files
from disk on every single batch — 19,580 file reads per run.  
**Root cause:** `np.load(ep_path)` inside `__getitem__` called
every time PyTorch requested a sample.  
**Fix:** Loaded all episodes into RAM at startup using
`np.concatenate` into 3 big tensors (obs_state, obs_env, action).
RAM usage: only 29.1MB for 250,500 steps.  
**Result:** 30 minutes → 28 seconds. 64x speedup.

### Problem 3 — Training only on timestep 0
**What happened:** First training script used
`batch["action"][:, 0, :]` — only the first timestep of each
episode. Network memorized starting positions, not full trajectories.  
**Fix:** Switched to RobotZooStepDataset which returns one
timestep per sample. Now training on all 250,500 timesteps.  
**Result:** Much more general policy — sees entire trajectory
not just start.

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Dataset size | 250,500 steps (1,002 episodes × 250 steps) |
| RAM usage | 29.1 MB |
| Training time | 28 seconds / 20 epochs |
| Final loss | 0.0000 |
| Parameters | 139,272 |
| GPU | RTX 5060 Ti 16.6GB |

---

## Key Insights

**Insight 1 — Disk I/O is the enemy of training speed.**
For small datasets that fit in RAM, always preload everything
at startup. The 64x speedup came from eliminating repeated
disk reads, not from any algorithmic change.

**Insight 2 — Loss = 0 does not mean good policy.**
Training loss hit 0.0000 but we haven't evaluated in Isaac Lab yet.
The network may have memorized training data without learning
a generalizable pick policy. Real validation is Week 1's job.

**Insight 3 — Episode vs step level matters.**
BC at episode level (1,002 samples) vs step level (250,500 samples)
is a 250x difference in training signal. Always train at step level
for imitation learning.

---

## Questions for Next Week

- Does loss=0 actually translate to successful picks in Isaac Lab?
- What success rate does BC achieve on franka/pick_cube?
- Does performance drop significantly on other tasks (reach, place, push)?

---

## What's Next — Week 01

- Train BC on all 5 Franka tasks
- Write Isaac Lab evaluation script
- Get first real success rate numbers
- Push checkpoints to HuggingFace
