"""Utility di warm-start BC -> SAC.

Nota storica: questo modulo descrive il BCDriver poi sostituito, in cui il
warm-start è stato progettato (pre-promozione bc_tita_v20/cem_v5). Il
BCDriver principale è cambiato da allora, ma il warm-start e i checkpoint
SAC che ne derivano restano legati a quei modelli storici.

Prima il "modello BC" era in realtà un blend di due reti separate
(bc_from_attempt1_v1 per i rettilinei, bc_from_olddriver_v1 per le curve, più
gain post-hoc e cambio marcia manuale su RPM, oggi replicato in
drivers/rl/legacy_bc_blend.py), non un'unica rete da cui trapiantare i pesi.

Decisione (confermata con l'utente): warm-start dell'attore SAC solo da
bc_from_olddriver_v1 (il più generalista dei due). Logica di fusione, gain
STEER/ACCEL/BRAKE e cambio marcia sono trattati come euristiche solo-BC; il
fine-tuning RL è libero di reimpararli (il cambio marcia resta comunque
esterno alla rete, gestito da training/rl/torcs_gym_env.py e drivers/rl/driver.py).
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
    """Copia il backbone MLP di BC + le teste di azione in model.policy.actor.

    Layout di BCPolicy (input_dim=26, hidden_dims=[128, 64]):
        backbone.0  Linear(26, 128)  -> actor.latent_pi.0
        backbone.2  Linear(128, 64)  -> actor.latent_pi.2
        head_steer  Linear(64, 1)  ┐
        head_accel  Linear(64, 1)  ├─ impilate -> actor.mu  Linear(64, 3)
        head_brake  Linear(64, 1)  ┘
        head_gear   Linear(64, 1)    (scartata, la marcia è gestita fuori
                                       dalla rete; vedi torcs_gym_env.py)

    È un adattatore approssimato, non un trasferimento bit-esatto: BC applica
    tanh/sigmoid/sigmoid per ciascuna testa, mentre l'attore SAC di SB3
    comprime sempre `mu` con tanh internamente prima di riscalare ai limiti
    dello spazio d'azione. Abbastanza vicino per un warm-start: è ammesso usare
    "an adapter layer rather than retraining feature extraction from scratch"
    per questo caso.

    Richiede che la policy SAC sia stata costruita con policy_kwargs
    net_arch=[128, 64] così le forme di actor.latent_pi/mu combaciano
    esattamente con quelle di BCPolicy.
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

        # Rumore di esplorazione iniziale piccolo ma non nullo (invece
        # dell'init casuale di SB3), così l'attore parte vicino alla policy BC
        # deterministica. -1.0 (std circa 0.368) sembrava già modesto ma non
        # lo è: su accel/brake (range [0,1]) copre oltre un terzo dello spazio
        # d'azione, e in 8 run indipendenti (reward/entropia/learning rate
        # diversi) la policy degradava sempre appena l'attore iniziava ad
        # aggiornarsi. -2.3 (std circa 0.100) è un ordine di grandezza più
        # vicino a un vero rumore di rifinitura.
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(-2.3)

    logger.info("Warm-started SAC actor from BC checkpoint: %s", bc_checkpoint)


class WarmStartSAC(SAC):
    """SAC con una finestra di warm-up solo per il critic.

    "Initialize the SAC critic separately (it has no BC
    equivalent): a few thousand steps of critic-only warm-up before joint
    actor-critic updates begin can reduce early instability, since the actor
    starts 'ahead' of an untrained critic."

    train() è una copia riga per riga di stable_baselines3.sac.sac.SAC.train()
    (SB3 2.9.0) con una modifica: finché self._n_updates è sotto
    critic_warmup_steps, gli step degli optimizer dell'attore e del
    coefficiente di entropia vengono saltati, così si aggiornano solo i
    critic (e le loro reti target).
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
