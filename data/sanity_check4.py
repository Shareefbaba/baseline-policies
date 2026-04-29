import numpy as np
import torch
from pathlib import Path
from robot_zoo_dataset import RobotZooDataset

DATA_DIR = Path.home() / "robot_zoo/data/demos"

ds = RobotZooDataset(DATA_DIR, "franka", "pick_cube")

# grab cube starting position from first step of 50 episodes
start_positions = []
for i in range(50):
    sample = ds[i]
    # first step cube position
    cube_pos = sample["observation.environment_state"][0]
    start_positions.append(cube_pos)

start_positions = torch.stack(start_positions)  # shape (50, 3)

print("=== CUBE START POSITION VARIATION (50 episodes) ===")
print(f"X range: {start_positions[:,0].min():.3f} to {start_positions[:,0].max():.3f}")
print(f"Y range: {start_positions[:,1].min():.3f} to {start_positions[:,1].max():.3f}")
print(f"Z range: {start_positions[:,2].min():.3f} to {start_positions[:,2].max():.3f}")
print()
print(f"X std: {start_positions[:,0].std():.4f}")
print(f"Y std: {start_positions[:,1].std():.4f}")
print(f"Z std: {start_positions[:,2].std():.4f}")
print()
print("(Z std near 0 = cube always on table = correct)")
print("(X and Y std > 0 = cube randomly placed = correct)")
