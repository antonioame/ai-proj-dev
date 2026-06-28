"""Simple MLP for behavioral cloning (sensors → actions)."""

import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    """Multi-layer perceptron: 26 inputs (sensors) → 3 outputs (steer, accel, brake)."""

    def __init__(self, input_dim: int = 26, hidden_dims: list = None):
        """
        Parameters
        ----------
        input_dim : int
            Number of sensor features (angle, speed, trackPos, track_0-18, rpm, gear)
        hidden_dims : list
            Hidden layer sizes. Default: [128, 64]
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output: steer, accel, brake (3 actions)
        layers.append(nn.Linear(prev_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: normalized sensor state → action outputs (unclamped)."""
        return self.net(x)
