import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.append(str(Path.home() / "baseline_policies"))

from models.bc_policy import BCPolicy
from data.robot_zoo_dataset import RobotZooDataset


def train(robot, task, num_steps, lr, batch_size):

    # --- 1. load data ---
    DATA_DIR = Path.home() / "robot_zoo/data/demos"
    dataset = RobotZooDataset(DATA_DIR, robot, task)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- 2. figure out dimensions ---
    sample = dataset[0]
    obs_dim = (sample["observation.state"].shape[-1] +
               sample["observation.environment_state"].shape[-1])
    act_dim = sample["action"].shape[-1]
    print(f"obs_dim={obs_dim}, act_dim={act_dim}")

    # --- 3. build model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    policy = BCPolicy(obs_dim=obs_dim, act_dim=act_dim).to(device)

    # --- 4. optimizer and loss ---
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # --- 5. training loop ---
    print(f"\nTraining BC on {robot}/{task} for {num_steps} steps...")
    print(f"{'Step':>6}  {'Loss':>10}")
    print("-" * 20)

    data_iter = iter(dataloader)
    for step in range(num_steps):

        # get next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # build observation by concatenating state + cube pos
        obs_state = batch["observation.state"].to(device)
        obs_env   = batch["observation.environment_state"].to(device)
        obs = torch.cat([obs_state, obs_env], dim=-1)

        # actions are shape (batch, T, act_dim) — take first timestep
        action = batch["action"][:, 0, :].to(device)

        # forward pass
        pred_action = policy(obs[:, 0, :])

        # compute loss
        loss = loss_fn(pred_action, action)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log every 10 steps
        if step % 10 == 0:
            print(f"{step:>6}  {loss.item():>10.4f}")

    print("-" * 20)
    print("Training complete ✅")
    return policy


if __name__ == "__main__":
    policy = train(
        robot="franka",
        task="pick_cube",
        num_steps=1000,
        lr=1e-4,
        batch_size=32,
    )
    
    # save checkpoint
    os.makedirs("checkpoints/franka_pick_bc", exist_ok=True)
    torch.save(policy.state_dict(), "checkpoints/franka_pick_bc/step1000.pt")
    print("Checkpoint saved to checkpoints/franka_pick_bc/step1000.pt ✅")
