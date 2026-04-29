import numpy as np
import torch
from pathlib import Path
from robot_zoo_dataset import RobotZooDataset

DATA_DIR = Path.home() / "robot_zoo/data/demos"

ds = RobotZooDataset(DATA_DIR, "franka", "pick_cube")

# collect actions from first 10 episodes
all_actions = []
for i in range(10):
    sample = ds[i]
    all_actions.append(sample["action"])

all_actions = torch.cat(all_actions, dim=0)  # shape: (2500, 8)

print("=== ACTION RANGES (franka/pick_cube) ===")
print(f"Shape: {all_actions.shape}")
print(f"Min:   {all_actions.min(dim=0).values.numpy().round(3)}")
print(f"Max:   {all_actions.max(dim=0).values.numpy().round(3)}")
print(f"Mean:  {all_actions.mean(dim=0).numpy().round(3)}")
