"""BC -> SAC warm-start utilities (REINFORCEMENT_LEARNING.md Section 6.1).

Phase 2's "BC-trained network" is, in this repo, actually a hybrid blend of
two separately-trained models (_DRIVER/driver.py: BCDriver blends
bc_from_attempt1_v1 for straights with bc_from_olddriver_v1 for corners, plus
post-hoc gain multipliers and manual RPM-based gear shifting outside either
network). There is no single network to transplant weights from as Section
6.1 assumes.

Decision (confirmed with the user 2026-07-08): warm-start the SAC actor from
bc_from_olddriver_v1 (the corner model) only — the more general-purpose of
the two. The blend logic, STEER/ACCEL/BRAKE gain multipliers and RPM gear
shifting are treated as BC-only heuristics; RL fine-tuning is free to
relearn/override them (gear shifting stays outside the network either way,
handled by training/rl/torcs_gym_env.py and drivers/rl/driver.py directly).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.rl.features import FEATURE_DIM

logger = logging.getLogger(__name__)

BC_MODELS_DIR = Path(__file__).resolve().parents[2] / "_DRIVER" / "models"
DEFAULT_BC_CHECKPOINT = BC_MODELS_DIR / "bc_from_olddriver_v1.pth"


def load_bc_backbone_into_actor(model: SAC, bc_checkpoint: Path = DEFAULT_BC_CHECKPOINT) -> None:
    """Copy the BC MLP backbone + action heads into model.policy.actor.

    BCPolicy layout (input_dim=26, hidden_dims=[128, 64]):
        backbone.0  Linear(26, 128)  -> actor.latent_pi.0
        backbone.2  Linear(128, 64)  -> actor.latent_pi.2
        head_steer  Linear(64, 1)  ┐
        head_accel  Linear(64, 1)  ├─ stacked -> actor.mu  Linear(64, 3)
        head_brake  Linear(64, 1)  ┘
        head_gear   Linear(64, 1)    (dropped — gear is handled outside the
                                       network; see torcs_gym_env.py)

    This is an approximate adapter, not a bit-exact transfer: BC applies
    tanh/sigmoid/sigmoid per head, while SB3's SAC actor always squashes `mu`
    through tanh internally before rescaling to the action space bounds. Close
    enough for a warm start — Section 6.1 explicitly allows "an adapter layer
    rather than retraining feature extraction from scratch" for this case.

    Requires the SAC policy to have been built with policy_kwargs
    net_arch=[128, 64] so actor.latent_pi/mu shapes match BCPolicy's exactly.
    """
    bc_state: dict[str, torch.Tensor] = torch.load(bc_checkpoint, map_location="cpu")
    actor = model.policy.actor

    if actor.latent_pi[0].in_features != FEATURE_DIM:
        raise ValueError(
            f"SAC actor input dim {actor.latent_pi[0].in_features} != BC feature dim {FEATURE_DIM}. "
            "Build the SAC model with an observation space matching training.rl.features.FEATURE_DIM."
        )

    with torch.no_grad():
        actor.latent_pi[0].weight.copy_(bc_state["backbone.0.weight"])
        actor.latent_pi[0].bias.copy_(bc_state["backbone.0.bias"])
        actor.latent_pi[2].weight.copy_(bc_state["backbone.2.weight"])
        actor.latent_pi[2].bias.copy_(bc_state["backbone.2.bias"])

        mu_weight = torch.cat(
            [bc_state["head_steer.weight"], bc_state["head_accel.weight"], bc_state["head_brake.weight"]],
            dim=0,
        )
        mu_bias = torch.cat(
            [bc_state["head_steer.bias"], bc_state["head_accel.bias"], bc_state["head_brake.bias"]],
            dim=0,
        )
        actor.mu.weight.copy_(mu_weight)
        actor.mu.bias.copy_(mu_bias)

        # Modest, non-zero initial exploration noise instead of SB3's default
        # random init — the actor starts close to the deterministic BC policy
        # (Section 4.2.4: prefer changes that don't unlearn safe BC behaviour
        # in early fine-tuning).
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(-1.0)

    logger.info("Warm-started SAC actor from BC checkpoint: %s", bc_checkpoint)


class WarmStartSAC(SAC):
    """SAC with a critic-only warm-up window.

    Section 6.1: "Initialize the SAC critic separately (it has no BC
    equivalent) — a few thousand steps of critic-only warm-up before joint
    actor-critic updates begin can reduce early instability, since the actor
    starts 'ahead' of an untrained critic."

    train() is a line-for-line copy of stable_baselines3.sac.sac.SAC.train()
    (SB3 2.9.0) with one change: while self._n_updates is below
    critic_warmup_steps, the actor and entropy-coefficient optimizer steps
    are skipped so only the critics (and their target networks) update.
    """

    def __init__(self, *args: Any, critic_warmup_steps: int = 3000, **kwargs: Any) -> None:
        self.critic_warmup_steps = critic_warmup_steps
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            warming_up = (self._n_updates + gradient_step) < self.critic_warmup_steps

            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None and not warming_up:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if not warming_up:
                q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
