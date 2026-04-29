import numpy as np
import torch
from pathlib import Path
from robot_zoo_dataset import RobotZooDataset

DATA_DIR = Path.home() / "robot_zoo/data/demos"

ds = RobotZooDataset(DATA_DIR, "franka", "pick_cube")

lengths = []
for i in range(50):
    sample = ds[i]
    lengths.append(sample["action"].shape[0])

lengths = torch.tensor(lengths, dtype=torch.float32)

print("=== EPISODE LENGTHS (first 50 episodes) ===")
print(f"Min:    {lengths.min().item():.0f} steps")
print(f"Max:    {lengths.max().item():.0f} steps")
print(f"Mean:   {lengths.mean().item():.1f} steps")
print(f"Std:    {lengths.std().item():.1f} steps")
print(f"All same length: {lengths.std().item() == 0.0}")
