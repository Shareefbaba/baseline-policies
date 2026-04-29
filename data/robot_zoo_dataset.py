import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class RobotZooDataset(Dataset):

    def __init__(self, data_dir, robot, task, success_only=True):
        folder = Path(data_dir) / f"{robot}_{task}"
        if not folder.exists():
            raise FileNotFoundError(f"No data folder found at {folder}")

        all_episodes = sorted(folder.glob("episode_*.npz"))

        if success_only:
            self.episodes = []
            for ep_path in all_episodes:
                ep = np.load(ep_path)
                if ep["success"][0]:
                    self.episodes.append(ep_path)
        else:
            self.episodes = all_episodes

        print(f"[RobotZooDataset] {robot}/{task}: "
              f"{len(self.episodes)} episodes loaded "
              f"(success_only={success_only})")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = np.load(self.episodes[idx])
        return {
            "observation.state": torch.tensor(
                ep["observation_state"], dtype=torch.float32),
            "observation.environment_state": torch.tensor(
                ep["observation_cube_pos"], dtype=torch.float32),
            "action": torch.tensor(
                ep["action"], dtype=torch.float32),
            "success": torch.tensor(
                ep["success"][0], dtype=torch.bool),
        }


if __name__ == "__main__":
    DATA_DIR = Path.home() / "robot_zoo/data/demos"

    combos = [
        ("franka", "pick_cube"),
        ("franka", "reach"),
        ("fetch",  "pick_cube"),
        ("stretch","pick_cube"),
    ]

    for robot, task in combos:
        try:
            ds = RobotZooDataset(DATA_DIR, robot, task)
            sample = ds[0]
            print(f"  obs.state:       {sample['observation.state'].shape}")
            print(f"  obs.env_state:   {sample['observation.environment_state'].shape}")
            print(f"  action:          {sample['action'].shape}")
            print()
        except FileNotFoundError as e:
            print(f"  SKIP: {e}\n")
