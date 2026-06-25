"""MLP policy network: sensor vector → action vector."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """Feedforward network mapping sensor observations to driving actions.

    Output heads:
    - continuous: steer, accel, brake (tanh / sigmoid activations)
    - discrete: gear (raw logits, CrossEntropyLoss during training)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[list[int]] = None,
        gear_classes: int = 8,  # gears -1,0,1,2,3,4,5,6 → 8 classes
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 256, 128]

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h

        self.backbone = nn.Sequential(*layers)

        # Continuous control head
        self.steer_head = nn.Sequential(nn.Linear(prev, 1), nn.Tanh())
        self.accel_head = nn.Sequential(nn.Linear(prev, 1), nn.Sigmoid())
        self.brake_head = nn.Sequential(nn.Linear(prev, 1), nn.Sigmoid())

        # Discrete gear head (logits only — apply softmax at inference)
        self.gear_head = nn.Linear(prev, gear_classes)

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "steer": self.steer_head(h).squeeze(-1),
            "accel": self.accel_head(h).squeeze(-1),
            "brake": self.brake_head(h).squeeze(-1),
            "gear_logits": self.gear_head(h),
        }

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convenience wrapper: returns gear as argmax (integer class)."""
        with torch.no_grad():
            out = self(x)
            out["gear"] = out["gear_logits"].argmax(dim=-1) - 1  # shift back to -1..6
        return out
