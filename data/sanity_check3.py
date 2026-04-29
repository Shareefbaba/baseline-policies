import numpy as np
import torch
from pathlib import Path
from robot_zoo_dataset import RobotZooDataset

DATA_DIR = Path.home() / "robot_zoo/data/demos"

ds = RobotZooDataset(DATA_DIR, "franka", "pick_cube")

sample = ds[0]
actions = sample["action"]  # shape (250, 8)

# difference between consecutive steps
deltas = torch.diff(actions, dim=0)  # shape (249, 8)

print("=== TRAJECTORY SMOOTHNESS (episode 0) ===")
print(f"Max jump between steps: {deltas.abs().max().item():.4f}")
print(f"Mean jump between steps: {deltas.abs().mean().item():.4f}")
print(f"Std of jumps: {deltas.abs().std().item():.4f}")
print()
print("Per-joint max jump:")
for j in range(8):
    label = f"J{j+1}" if j < 7 else "Gripper"
    print(f"  {label}: {deltas[:, j].abs().max().item():.4f}")
