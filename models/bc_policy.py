import torch
import torch.nn as nn


class BCPolicy(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, obs):
        return self.net(obs)


if __name__ == "__main__":
    obs_dim = 21
    act_dim = 8

    policy = BCPolicy(obs_dim=obs_dim, act_dim=act_dim)

    print("=== BC Policy Architecture ===")
    print(policy)
    print()

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Total parameters: {total_params:,}")
    print()

    dummy_obs = torch.randn(4, obs_dim)
    dummy_out = policy(dummy_obs)
    print(f"Input shape:  {dummy_obs.shape}")
    print(f"Output shape: {dummy_out.shape}")
    print("Forward pass works ✅")
