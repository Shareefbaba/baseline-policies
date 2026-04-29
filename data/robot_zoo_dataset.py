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


class RobotZooStepDataset(Dataset):
    """
    Loads all episodes into RAM at startup.
    Eliminates disk I/O during training.
    482MB dataset fits easily in 32GB RAM.
    """

    def __init__(self, data_dir, robot, task, success_only=True):
        episode_ds = RobotZooDataset(data_dir, robot, task, success_only)

        obs_states = []
        obs_envs = []
        actions = []

        print(f"[RobotZooStepDataset] Loading all episodes into RAM...")

        for ep_path in episode_ds.episodes:
            data = np.load(ep_path)
            obs_states.append(data["observation_state"])
            obs_envs.append(data["observation_cube_pos"])
            actions.append(data["action"])

        # stack everything into big tensors — shape (N, dim)
        self.obs_state = torch.tensor(
            np.concatenate(obs_states, axis=0), dtype=torch.float32)
        self.obs_env = torch.tensor(
            np.concatenate(obs_envs, axis=0), dtype=torch.float32)
        self.action = torch.tensor(
            np.concatenate(actions, axis=0), dtype=torch.float32)

        print(f"[RobotZooStepDataset] Total steps: {len(self):,}")
        print(f"[RobotZooStepDataset] RAM usage: "
              f"{(self.obs_state.nbytes + self.obs_env.nbytes + self.action.nbytes) / 1e6:.1f} MB")

    def __len__(self):
        return self.obs_state.shape[0]

    def __getitem__(self, idx):
        return {
            "observation.state": self.obs_state[idx],
            "observation.environment_state": self.obs_env[idx],
            "action": self.action[idx],
        }

if __name__ == "__main__":
    DATA_DIR = Path.home() / "robot_zoo/data/demos"
    ds = RobotZooStepDataset(DATA_DIR, "franka", "pick_cube")
    sample = ds[0]
    print()
    print("=== Single step sample ===")
    for k, v in sample.items():
        print(f"  {k}: {v.shape}")
