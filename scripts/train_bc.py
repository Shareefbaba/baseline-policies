import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.append(str(Path.home() / "baseline_policies"))

from models.bc_policy import BCPolicy
from data.robot_zoo_dataset import RobotZooStepDataset


def train(robot, task, epochs, lr, batch_size, save_dir):

    # --- 1. load data ---
    DATA_DIR = Path.home() / "robot_zoo/data/demos"
    dataset = RobotZooStepDataset(DATA_DIR, robot, task)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    # --- 2. figure out dimensions ---
    sample = dataset[0]
    obs_dim = (sample["observation.state"].shape[0] +
               sample["observation.environment_state"].shape[0])
    act_dim = sample["action"].shape[0]
    print(f"obs_dim={obs_dim}  act_dim={act_dim}")
    print(f"Dataset: {len(dataset):,} steps  "
          f"Batches per epoch: {len(dataloader)}")

    # --- 3. build model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    policy = BCPolicy(obs_dim=obs_dim, act_dim=act_dim).to(device)

    # --- 4. optimizer and loss ---
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # --- 5. training loop ---
    print(f"\nTraining BC — {robot}/{task} "
          f"for {epochs} epochs...\n")
    print(f"{'Epoch':>6}  {'Loss':>10}")
    print("-" * 20)

    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            obs = torch.cat([
                batch["observation.state"].to(device),
                batch["observation.environment_state"].to(device),
            ], dim=-1)

            action = batch["action"].to(device)

            pred_action = policy(obs)
            loss = loss_fn(pred_action, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        epoch_loss = total_loss / num_batches

        if epoch % 5 == 0:
            print(f"{epoch:>6}  {epoch_loss:>10.4f}")

        # save best checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "loss": best_loss,
                "robot": robot,
                "task": task,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "model_state_dict": policy.state_dict(),
            }, f"{save_dir}/best.pt")

    print("-" * 20)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoint saved to {save_dir}/best.pt ✅")
    return policy


if __name__ == "__main__":
    train(
        robot="franka",
        task="pick_cube",
        epochs=20,
        lr=1e-4,
        batch_size=256,
        save_dir="checkpoints/franka_pick_bc",
    )
