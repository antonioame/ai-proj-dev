"""Utility di warm-start BC -> SAC (REINFORCEMENT_LEARNING.md Sezione 6.1).

NOTA STORICA: la descrizione che segue si riferisce al BCDriver dell'epoca in
cui questo warm-start è stato progettato (pre-promozione del 2026-07-15) — il
BCDriver di produzione è cambiato due volte da allora (bc_tita_v20, modello
singolo, dal 2026-07-15; cem_v5 dal 2026-07-19), ma il warm-start e i
checkpoint SAC che ne derivano restano legati ai modelli del blend storico,
quindi la descrizione resta quella corretta per questo modulo.

La "rete addestrata con BC" della Fase 2 era, all'epoca, in realtà una
fusione ibrida di due modelli addestrati separatamente (il BCDriver di allora
fondeva bc_from_attempt1_v1 per i rettilinei con bc_from_olddriver_v1
per le curve, più moltiplicatori di guadagno applicati a posteriori e cambio
marcia manuale basato su RPM esterno a entrambe le reti; oggi quel blend è
replicato in drivers/rl/legacy_bc_blend.py). Non esisteva un'unica
rete da cui trapiantare i pesi come assume la Sezione 6.1.

Decisione (confermata con l'utente il 2026-07-08): fare il warm-start
dell'attore SAC solo da bc_from_olddriver_v1 (il modello curva) — il più
generalista dei due. La logica di fusione, i moltiplicatori di guadagno
STEER/ACCEL/BRAKE e il cambio marcia basato su RPM sono trattati come
euristiche solo-BC; il fine-tuning RL è libero di reimpararli/sostituirli (il
cambio marcia resta comunque esterno alla rete in entrambi i casi, gestito
direttamente da training/rl/torcs_gym_env.py e drivers/rl/driver.py).
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
        head_gear   Linear(64, 1)    (scartata — la marcia è gestita fuori
                                       dalla rete; vedi torcs_gym_env.py)

    È un adattatore approssimato, non un trasferimento bit-esatto: BC applica
    tanh/sigmoid/sigmoid per ciascuna testa, mentre l'attore SAC di SB3
    comprime sempre `mu` con tanh internamente prima di riscalare ai limiti
    dello spazio d'azione. Abbastanza vicino per un warm-start — la Sezione
    6.1 permette esplicitamente "an adapter layer rather than retraining
    feature extraction from scratch" per questo caso.

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

        # Rumore di esplorazione iniziale modesto e non nullo, invece
        # dell'inizializzazione casuale di default di SB3 — l'attore parte
        # vicino alla policy BC deterministica (Sezione 4.2.4: preferire
        # cambiamenti che non disimparino il comportamento BC sicuro nelle
        # prime fasi del fine-tuning).
        #
        # -1.0 (std≈0.368) si è rivelato tutt'altro che "modesto": su
        # accel/brake, il cui range valido è [0,1], un sigma di 0.368 copre
        # oltre un terzo dell'intero spazio d'azione. In 8 run diretti
        # indipendenti (reward diverse, entropia auto/fissa da 0.02 a 0.08,
        # learning rate da 3e-4 a 5e-5) la policy degrada sistematicamente
        # nello stesso identico modo appena l'attore inizia ad aggiornarsi —
        # sempre a valle di questo stesso rumore iniziale enorme. -2.3
        # (std≈0.100) è un ordine di grandezza più vicino a un vero rumore di
        # rifinitura attorno a una policy già buona.
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(-2.3)

    logger.info("Warm-started SAC actor from BC checkpoint: %s", bc_checkpoint)


class WarmStartSAC(SAC):
    """SAC con una finestra di warm-up solo per il critic.

    Sezione 6.1: "Initialize the SAC critic separately (it has no BC
    equivalent) — a few thousand steps of critic-only warm-up before joint
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
