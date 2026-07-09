#!/usr/bin/env python3
"""
MARL core for the FLoRa bridge -- shared policy/env building blocks.

Provides the pieces reused across the bridge:
    - _load_env_class / ENV_KWARGS / OBS_AGENT_ID : construct MultiFlyingLoRaEnv
    - Actor + DEFAULT_CKPT                         : the trained MAPPO RNN actor
    - MarlController : greedy, GRU-stateful policy driver over the env, used by
      run_python_eval_logged.py for the standalone (pure-Python) evaluation path.

The closed-loop path instead splits policy (policy_server.py) from env (env_socket.py);
this module is imported by both to share the env construction and checkpoint constants.
"""

import os
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Paths -- project root is the parent of flora_run/ (derived from this file's location,
# so no absolute path is hard-coded). Override with env var TRAJECTORY_MAPPO_ROOT if needed.
# ----------------------------------------------------------------------------
BASE = os.environ.get(
    "TRAJECTORY_MAPPO_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
MULTI_LORA_PATH = os.path.join(BASE, "src/envs/MADE/gym_mdde1/envs/multi_lora.py")
# GLo-MAPPO, 2 UAVs / 50 EDs, seed 41 -- final (~2M-step) checkpoint.
# Shipped with the repo (see models/README.md), so inference matches the reported results.
CKPT_REL = "models/glo_ed50_seed41/agent.th"
DEFAULT_CKPT = os.path.join(BASE, CKPT_REL)


# Env construction args — must match the training config (sacred run 1).
ENV_KWARGS = dict(
    num_uavs=2,
    num_eds=50,
    area_size=[1000, 1000],
    uav_altitude=150,
    max_episode_steps=500,
    max_speed=30,
    comm_range=300,
    safe_distance=3,
)

HIDDEN_DIM = 128          # config.hidden_dim
OBS_AGENT_ID = True       # config.obs_agent_id
OBS_LAST_ACTION = False   # config.obs_last_action


# ----------------------------------------------------------------------------
# Load MultiFlyingLoRaEnv directly from file (no package import side effects)
# ----------------------------------------------------------------------------
def _load_env_class(path=MULTI_LORA_PATH):
    spec = importlib.util.spec_from_file_location("multi_lora", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MultiFlyingLoRaEnv


# ----------------------------------------------------------------------------
# Actor network — same topology as src/modules/agents/rnn_agent.py so the
# agent.th state_dict loads as-is (keys: fc1.*, rnn.*, fc2.*).
# ----------------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, inputs, h):
        x = F.relu(self.fc1(inputs))
        h = self.rnn(x, h)
        q = self.fc2(h)
        return q, h


class MarlController:
    """Greedy, GRU-stateful policy driver over MultiFlyingLoRaEnv."""

    def __init__(self, checkpoint=DEFAULT_CKPT, hidden_dim=HIDDEN_DIM,
                 env_kwargs=None, device="cpu"):
        self.device = torch.device(device)
        EnvCls = _load_env_class()
        self.env = EnvCls(**(env_kwargs or ENV_KWARGS))

        # Derived dims straight from the env (single source of truth)
        self.n_agents = self.env.n_agents
        self.max_ed_per_uav = self.env.max_ed_per_uav
        self.action_nvec = list(self.env.action_nvec)          # per-head sizes
        self.n_actions = int(sum(self.action_nvec))            # logit width
        self.obs_flat = int(np.prod(self.env.observation_space[0].shape))
        self.input_shape = self.obs_flat + (self.n_agents if OBS_AGENT_ID else 0)
        if OBS_LAST_ACTION:
            self.input_shape += self.n_actions

        # Build + load actor
        self.actor = Actor(self.input_shape, hidden_dim, self.n_actions).to(self.device)
        sd = torch.load(checkpoint, map_location=self.device)
        self._check_dims(sd)
        self.actor.load_state_dict(sd)
        self.actor.eval()

        self.h = None
        self.t = 0
        self.obs = None

    def _check_dims(self, sd):
        in_ck = sd["fc1.weight"].shape[1]
        out_ck = sd["fc2.weight"].shape[0]
        assert in_ck == self.input_shape, \
            f"input mismatch: checkpoint {in_ck} vs derived {self.input_shape}"
        assert out_ck == self.n_actions, \
            f"n_actions mismatch: checkpoint {out_ck} vs derived {self.n_actions}"

    # ---- episode lifecycle ----
    def reset(self):
        self.obs = self.env.reset()                      # tuple of (M,16) arrays
        self.h = torch.zeros(self.n_agents, self.actor.fc1.out_features,
                             device=self.device)
        self.t = 0
        return self.snapshot(done=False)

    def _build_inputs(self):
        rows = []
        for a in range(self.n_agents):
            flat = np.asarray(self.obs[a], dtype=np.float32).reshape(-1)   # 400
            if OBS_AGENT_ID:
                oh = np.zeros(self.n_agents, dtype=np.float32)
                oh[a] = 1.0
                flat = np.concatenate([flat, oh])
            rows.append(flat)
        return torch.tensor(np.stack(rows), dtype=torch.float32, device=self.device)

    def _greedy(self, q, avail):
        """Per-head masked argmax (test_greedy). Returns list of length-52 action vecs."""
        actions = []
        for a in range(self.n_agents):
            qa = q[a].clone()
            mask = torch.tensor(avail[a], dtype=torch.bool, device=self.device)
            qa[~mask] = -1e10
            act, start = [], 0
            for n in self.action_nvec:
                act.append(int(torch.argmax(qa[start:start + n]).item()))
                start += n
            actions.append(np.array(act, dtype=np.int64))
        return actions

    @torch.no_grad()
    def step(self):
        # avail mask corresponds to the CURRENT obs (env.get_obs was last called
        # in reset() or the previous step()); must be read before env.step().
        avail = self.env.get_avail_actions()
        inp = self._build_inputs()
        q, self.h = self.actor(inp, self.h)
        actions = self._greedy(q, avail)

        self.obs, reward, done, info = self.env.step(actions)
        self.t += 1
        return self.snapshot(done=bool(done), reward=reward, actions=actions)

    # ---- state extraction for FLoRa ----
    def snapshot(self, done, reward=None, actions=None):
        env = self.env
        uav_xy = [(float(p[0]), float(p[1])) for p in env.uav_positions]
        # Per-ED allocation: SF, TP(dBm), serving UAV (-1 if unserved this step)
        sf = np.asarray(env.sf_allocations_all_eds, dtype=int)
        tp = np.asarray(env.tp_allocations_all_eds, dtype=float)
        assoc = np.full(env.num_eds, -1, dtype=int)
        # associations is a lil_matrix (uav x ed) of datarate>0 for served EDs
        for u in range(env.n_agents):
            for e in env.associations[u].nonzero()[1]:
                assoc[e] = u
        served = int((assoc >= 0).sum())
        return {
            "t": self.t,
            "done": done,
            "reward": reward,
            "uav_xy": uav_xy,
            "ed_sf": sf,
            "ed_tp": tp,
            "ed_assoc": assoc,
            "served": served,
            "actions": actions,
        }
