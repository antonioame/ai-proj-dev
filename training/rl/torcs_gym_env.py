"""Wrapper Gymnasium attorno al client SCR di TORCS esistente.

Puramente additivo: importa torcs_env.client.TORCSClient così com'è e lo
orchestra dall'esterno. Nessuna modifica a torcs_env/client.py, sensors.py o
actions.py (REINFORCEMENT_LEARNING.md Sezione 1 / Sezione 6.3).

Lo spazio d'azione è solo steer/accel/brake — la marcia resta automatica
(basata su RPM, con le stesse soglie usate da _DRIVER/driver.py), rispettando
il requisito della Sezione 3 secondo cui lo spazio d'azione RL deve restare
identico a quello già usato da Fase 1/2, così il layout di output
dell'attore con warm-start BC coincide con quello dell'ambiente.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torcs_env.actions import Action
from torcs_env.client import RESTART, SHUTDOWN, TORCSClient
from torcs_env.sensors import SensorState
from training.rl.features import FEATURE_DIM, build_feature_vector
from training.rl.reward import REWARD_VERSIONS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Soglie del cambio marcia automatico, rispecchiate da _DRIVER/driver.py.BCDriver
# così la policy RL e l'eventuale RLDriver vedono lo stesso comportamento marce.
_GEAR_UP_RPM = 12000.0
_GEAR_DOWN_RPM = 6000.0

# Guadagni post-hoc identici a _DRIVER/driver.py.BCDriver (STEER_GAIN/ACCEL_GAIN/
# BRAKE_GAIN). Scoperti mancanti dall'intera pipeline SAC diretta dopo 5 run
# che non sono mai scesi sotto ~127-130s: sac_warmstart.load_bc_backbone_into_actor
# copia solo i pesi grezzi della rete BC nell'attore SAC, MAI i guadagni
# applicati a posteriori nell'output di BCDriver.step(). Di conseguenza
# l'attore con warm-start, anche a zero step di training, guida come una BC
# "non amplificata" — nel primissimo run diretto (critic_warmup_steps troppo
# alto, attore mai aggiornato) il checkpoint congelato valutava 143,244s
# contro i 121,978s della vera BC: ~21s persi SOLO per l'assenza di questi tre
# moltiplicatori, prima ancora che qualunque fine-tuning RL entri in gioco.
# Applicarli qui allinea il punto di partenza del warm-start al comportamento
# BC reale, così il fine-tuning RL parte alla pari con la baseline da battere
# invece che ~20s indietro.
_STEER_GAIN = 1.8
_ACCEL_GAIN = 1.40
_BRAKE_GAIN = 0.80

# Periodo di grazia all'avvio: applica pieno gas/sterzo zero per questo numero
# di step prima di passare il controllo alla policy. Rispecchia
# BCDriver.STARTUP_STEPS — evita che l'attore con warm-start veda mai lo stato
# di lancio fuori distribuzione (OOD: velocità≈0, marcia=0) che non ha mai
# visto durante il training BC.
_STARTUP_STEPS = 80

# Soglie di terminazione episodio.
_OFF_TRACK_STEPS_LIMIT = 25  # ~0.5s a 50 step/s — "terminazione aggressiva" per la Sezione 8
_STANDING_STILL_KMH = 5.0
_STANDING_STILL_STEPS_LIMIT = 150  # ~3s
_MAX_EPISODE_STEPS = 20000  # un giro Corkscrew è ~5.800 step a 50 step/s (vedi CLAUDE.md, Fase 3) — ampio margine
_EPISODE_START_RETRIES = 4

# Gestione del processo TORCS. Ogni episodio ottiene un processo TORCS NUOVO
# invece di un restart in-gara con meta=1: empiricamente quel restart in-gara
# è inaffidabile su questo setup (la connessione si riprende a metà e poi cade
# dopo pochi step nell'episodio successivo — la stessa instabilità che
# affliggeva il precedente tentativo di Fase 3, poi rimosso). Rilanciare il
# processo aggira il problema del tutto. Due requisiti di lancio, imparati a
# caro prezzo ed entrambi indispensabili:
#   1. La CWD deve essere la cartella d'installazione di TORCS, altrimenti
#      TORCS non riesce a risolvere i file di categoria dell'auto ("Bad Car
#      category for driver scr_server 1") e non apre mai la porta SCR.
#   2. Connettersi solo DOPO una breve pausa di grazia una volta che la porta
#      è aperta — TORCS apre la porta un istante prima che il suo loop di
#      simulazione sia davvero in grado di inviare i sensori, e connettersi
#      in quella finestra causa un deadlock.
TORCS_EXE = Path(os.environ.get("TORCS_EXE", r"U:\AI-Partition\torcs\torcs\wtorcs.exe"))
RACE_XML = PROJECT_ROOT / "torcs_env" / "race_config" / "corkscrew_solo.xml"
_TORCS_PORT_TIMEOUT = 30.0
_TORCS_STARTUP_GRACE = 4.0

DEFAULT_NORM_STATS_PATH = (
    PROJECT_ROOT / "_DRIVER" / "models" / "bc_from_olddriver_v1.npz"
)


def _port_is_bound(port: int) -> bool:
    """True se qualcosa occupa già *port* in UDP (cioè TORCS è attivo)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("", port))
        return False
    except OSError:
        return True
    finally:
        sock.close()


class TorcsSacEnv(gym.Env):
    """Un episodio = un tentativo di giro (termina al completamento del giro,
    a un'uscita di pista prolungata, o a una sosta prolungata)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        reward_version: str = "baseline_v1",
        norm_stats_path: Path = DEFAULT_NORM_STATS_PATH,
        max_episode_steps: int = _MAX_EPISODE_STEPS,
        auto_launch_torcs: bool = True,
    ) -> None:
        super().__init__()
        if reward_version not in REWARD_VERSIONS:
            raise ValueError(f"Unknown reward_version {reward_version!r}; choices: {list(REWARD_VERSIONS)}")
        self.reward_version = REWARD_VERSIONS[reward_version]
        self.max_episode_steps = max_episode_steps
        self._auto_launch = auto_launch_torcs
        self._port = int(port or os.environ.get("TORCS_PORT", "3001"))

        stats = np.load(norm_stats_path)
        self._obs_mean = stats["mean"].astype(np.float32)
        self._obs_std = stats["std"].astype(np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(FEATURE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._client = TORCSClient(host=host, port=port)
        self._torcs_proc: Optional[subprocess.Popen] = None
        self._connected = False
        self._needs_start = False
        self._current_gear = 1
        self._step_in_episode = 0
        self._off_track_run = 0
        self._standing_run = 0
        self._prev_lap_time = 0.0
        self._last_state: Optional[SensorState] = None

    # ------------------------------------------------------------------
    # API Gymnasium
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current_gear = 1
        self._step_in_episode = 0
        self._off_track_run = 0
        self._standing_run = 0

        if self._auto_launch:
            # DIFFERISCE il lancio+connessione di TORCS al primo step(). Con un
            # training a update del gradiente per-episodio, SB3 esegue il
            # (lungo) blocco di update del gradiente DOPO reset() e PRIMA del
            # primo step(). Se lanciassimo+connettessimo TORCS qui, resterebbe
            # fermo ad aspettare per tutta quella pausa, il che disturba la
            # partenza da fermo e manda in crash l'episodio dopo ~300 step
            # (verificato: succede anche con un'azione puro-BC). Differire
            # significa che gli update del gradiente girano mentre nessun
            # TORCS è in attesa; il primo step() lancia poi un processo nuovo.
            # Vedi il commento sulla config SAC in train_sac.py.
            self._needs_start = True
            self._last_state = None
            return np.zeros(FEATURE_DIM, dtype=np.float32), {}

        state = self._start_episode()
        self._prev_lap_time = state.lastLapTime
        self._last_state = state
        obs = self._normalize(build_feature_vector(state))
        return obs, {"distRaced": state.distRaced}

    def _ensure_started(self) -> None:
        """Esegue il lancio+connessione+partenza da fermo differiti al primo
        step() di un episodio (vedi reset())."""
        if self._needs_start:
            state = self._start_episode()
            self._prev_lap_time = state.lastLapTime
            self._last_state = state
            self._needs_start = False

    def step(self, action):
        self._ensure_started()
        steer, accel, brake = (float(a) for a in np.asarray(action, dtype=np.float32))
        self._update_gear(self._last_state.rpm)
        cmd = Action(
            steer=steer * _STEER_GAIN,
            accel=accel * _ACCEL_GAIN,
            brake=brake * _BRAKE_GAIN,
            gear=self._current_gear,
        )
        return self._send_and_observe(cmd)

    def _send_and_observe(self, cmd: Action):
        """Invia un comando di controllo, attende il prossimo pacchetto di
        sensori e lo trasforma nella tupla Gym (obs, reward, terminated,
        truncated, info). Condiviso tra lo step() ad azione diretta e lo
        step() dell'env residual."""
        self._client.send(cmd.clamp())

        try:
            state = self._await_fresh_state()
        except ConnectionError as exc:
            # Connessione SCR caduta a metà episodio. La transizione
            # (obs, action, next_obs) non è più valida, quindi l'episodio
            # termina per truncation invece di alimentare una transizione
            # corrotta nel replay buffer. Segna la disconnessione così il
            # prossimo reset() rilancia TORCS in modo pulito.
            logger.warning("Connection dropped mid-episode: %s", exc)
            self._connected = False
            obs = self._normalize(build_feature_vector(self._last_state))
            info = {"termination_reason": "connection_lost"}
            return obs, 0.0, False, True, info

        self._step_in_episode += 1

        reward, terminated, reason = self._reward_and_termination(self._last_state, state)
        truncated = (not terminated) and self._step_in_episode >= self.max_episode_steps

        if terminated or truncated:
            logger.info(
                "Episode end: step=%d reason=%s dist=%.0f speed=%.1f trackPos=%.3f",
                self._step_in_episode, reason or ("max_steps" if truncated else "?"),
                state.distRaced, state.speed, state.trackPos,
            )

        self._last_state = state
        obs = self._normalize(build_feature_vector(state))
        info = {
            "distRaced": state.distRaced,
            "trackPos": state.trackPos,
            "speed": state.speed,
            "termination_reason": reason,
            "lastLapTime": state.lastLapTime,
            "damage": state.damage,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._connected:
            self._client.close()
            self._connected = False
        self._kill_torcs()

    # ------------------------------------------------------------------
    # Gestione del processo TORCS
    # ------------------------------------------------------------------

    def _kill_torcs(self) -> None:
        proc = self._torcs_proc
        self._torcs_proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _launch_torcs(self) -> None:
        """Avvia un nuovo TORCS headless e blocca finché non è pronto a guidare.

        Vedi il commento a livello di modulo su TORCS_EXE per il motivo per
        cui sia la cwd sia la grazia post-bind sono entrambe obbligatorie.
        """
        if not TORCS_EXE.exists():
            raise FileNotFoundError(f"TORCS executable not found: {TORCS_EXE}")
        self._kill_torcs()
        proc = subprocess.Popen(
            [str(TORCS_EXE), "-r", str(RACE_XML)],
            cwd=str(TORCS_EXE.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._torcs_proc = proc
        deadline = time.monotonic() + _TORCS_PORT_TIMEOUT
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise ConnectionError(f"TORCS exited during startup (code {proc.returncode}).")
            if _port_is_bound(self._port):
                time.sleep(_TORCS_STARTUP_GRACE)
                return
            time.sleep(0.2)
        self._kill_torcs()
        raise ConnectionError(f"TORCS did not open the SCR port within {_TORCS_PORT_TIMEOUT}s.")

    # ------------------------------------------------------------------
    # Funzioni di supporto
    # ------------------------------------------------------------------

    def _await_fresh_state(self) -> SensorState:
        while True:
            result = self._client.receive()
            if result == RESTART:
                continue
            if result == SHUTDOWN:
                raise ConnectionError("TORCS server sent shutdown while the RL env was running.")
            return result

    def _start_episode(self, retries: int = _EPISODE_START_RETRIES) -> SensorState:
        """Porta l'auto a una partenza da fermo nuova e guidabile.

        Ogni episodio ottiene il proprio processo TORCS (vedi il commento su
        TORCS_EXE sul perché si rilancia invece di usare un restart in-gara
        con meta=1). L'intera sequenza — (ri)lancio, connessione, esecuzione
        del periodo di grazia con gas in avvio — viene ritentata come unità,
        poiché il lancio di TORCS stesso può occasionalmente andare in crash
        o l'handshake SCR può cadere su questo setup; un rilancio pulito
        recupera situazioni in cui un restart in-gara non potrebbe.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                if self._connected:
                    self._client.close()
                    self._connected = False
                if self._auto_launch:
                    self._launch_torcs()

                self._client.connect()
                self._connected = True
                state = self._await_fresh_state()
                return self._run_startup(state)
            except (ConnectionError, OSError) as exc:
                last_exc = exc
                logger.warning(
                    "Episode start failed (attempt %d/%d): %s", attempt, retries, exc,
                )
                try:
                    self._client.close()
                except Exception:
                    pass
                self._connected = False
        raise ConnectionError(f"Could not start an episode after {retries} attempts.") from last_exc

    def _run_startup(self, state: SensorState) -> SensorState:
        """Applica pieno gas/sterzo zero per _STARTUP_STEPS step per far
        partire l'auto da ferma prima che la policy prenda il controllo (vedi
        _STARTUP_STEPS). Sovrascrivibile: l'env residual salta questo passo e
        lascia che sia il suo driver base BC a gestire il lancio.
        """
        for _ in range(_STARTUP_STEPS):
            gear = self._startup_gear(state.speed)
            self._current_gear = gear
            self._client.send(Action(steer=0.0, accel=1.0, brake=0.0, gear=gear).clamp())
            state = self._await_fresh_state()
        return state

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        return ((raw - self._obs_mean) / self._obs_std).astype(np.float32)

    def _startup_gear(self, speed: float) -> int:
        if speed < 15.0:
            return 1
        if speed < 45.0:
            return 2
        return 3

    def _update_gear(self, rpm: float) -> None:
        if rpm > _GEAR_UP_RPM and self._current_gear < 6:
            self._current_gear += 1
        elif rpm < _GEAR_DOWN_RPM and self._current_gear > 1:
            self._current_gear -= 1

    def _reward_and_termination(self, prev_state: SensorState, state: SensorState):
        rv = self.reward_version
        reward = rv.step_fn(prev_state, state)

        off_track = abs(state.trackPos) > 1.0
        self._off_track_run = self._off_track_run + 1 if off_track else 0

        standing = state.speed < _STANDING_STILL_KMH
        self._standing_run = self._standing_run + 1 if standing else 0

        lap_completed = state.lastLapTime > 0 and state.lastLapTime != self._prev_lap_time

        if self._off_track_run >= _OFF_TRACK_STEPS_LIMIT:
            return reward + rv.off_track_penalty, True, "off_track"
        if self._standing_run >= _STANDING_STILL_STEPS_LIMIT:
            return reward + rv.off_track_penalty * 0.25, True, "standing_still"
        if lap_completed:
            return reward + rv.lap_bonus, True, "lap_completed"
        return reward, False, None
