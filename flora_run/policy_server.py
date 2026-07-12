#!/usr/bin/env python3

import argparse
import os
import pickle
import signal
import socket
import struct
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from marl_core import DEFAULT_CKPT   # reuse the same default checkpoint path


BASE = os.environ.get(
    "TRAJECTORY_MAPPO_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
MULTI_LORA_PATH = os.path.join(BASE, "src/envs/MADE/gym_mdde1/envs/multi_lora.py")

# GLo-MAPPO, 2 UAVs / 50 EDs, seed 41 
CKPT_REL = "models/glo_ed50_seed41/agent.th"
DEFAULT_CKPT = os.path.join(BASE, CKPT_REL)

# ---- actor topology: must match src/modules/agents/rnn_agent.py so agent.th loads
class Actor(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x, h):
        x = F.relu(self.fc1(x))
        h = self.rnn(x, h)
        return self.fc2(h), h


# ---- length-prefixed pickle framing (Python<->Python, localhost, trusted)
def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        c = conn.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("<I", len(data)) + data)


def recv_msg(conn):
    hdr = recv_exact(conn, 4)
    if hdr is None:
        return None
    (n,) = struct.unpack("<I", hdr)
    body = recv_exact(conn, n)
    return pickle.loads(body) if body is not None else None


class PolicyServer:
    def __init__(self, ckpt, host, port, device="cpu"):
        self.host, self.port = host, port
        self.device = torch.device(device)
        print("=" * 62)
        print("  GLo-MAPPO  POLICY SERVER   (Terminal A -- the brain)")
        print("=" * 62)
        print(f"[policy] Loading trained model ...")
        print(f"[policy]   checkpoint: {ckpt}")
        sd = torch.load(ckpt, map_location=self.device)
        # derive dims straight from the checkpoint (single source of truth)
        self.input_shape = int(sd["fc1.weight"].shape[1])
        self.hidden_dim = int(sd["fc1.weight"].shape[0])
        self.n_actions = int(sd["fc2.weight"].shape[0])
        print(f"[policy] Building actor network ...")
        self.actor = Actor(self.input_shape, self.hidden_dim, self.n_actions).to(self.device)
        self.actor.load_state_dict(sd)
        self.actor.eval()
        self.action_nvec = None
        self.n_agents = None
        self.h = None
        print(f"[policy] ✓ Model loaded successfully  (device={self.device})")
        print(f"[policy]   actor dims: input_shape={self.input_shape}  "
              f"hidden={self.hidden_dim}  n_actions={self.n_actions}")

    def _reset_h(self):
        self.h = torch.zeros(self.n_agents, self.hidden_dim, device=self.device)

    @torch.no_grad()
    def _act(self, inputs, avail):
        """Forward + per-head masked argmax (test_greedy). inputs:(n_agents,input_shape),
        avail:(n_agents,n_actions) -> actions:(n_agents,n_heads)."""
        x = torch.as_tensor(np.asarray(inputs), dtype=torch.float32, device=self.device)
        q, self.h = self.actor(x, self.h)
        avail = np.asarray(avail)
        actions = []
        for a in range(self.n_agents):
            qa = q[a].clone()
            mask = torch.as_tensor(avail[a], dtype=torch.bool, device=self.device)
            qa[~mask] = -1e10
            act, start = [], 0
            for nv in self.action_nvec:
                act.append(int(torch.argmax(qa[start:start + nv]).item()))
                start += nv
            actions.append(act)
        return np.asarray(actions, dtype=np.int64)

    def _handle(self, conn, addr):
        print(f"[policy] B connected: {addr}")
        while True:
            msg = recv_msg(conn)
            if msg is None:
                print("[policy] B disconnected")
                return
            t = msg.get("type")
            if t == "hello":
                self.action_nvec = list(msg["action_nvec"])
                self.n_agents = int(msg["n_agents"])
                if sum(self.action_nvec) != self.n_actions:
                    send_msg(conn, {"type": "error",
                                    "msg": f"action_nvec sum {sum(self.action_nvec)} "
                                           f"!= checkpoint n_actions {self.n_actions}"})
                    return
                self._reset_h()
                send_msg(conn, {"type": "ok", "input_shape": self.input_shape,
                                "n_actions": self.n_actions, "hidden_dim": self.hidden_dim})
                print(f"[policy] hello: n_agents={self.n_agents} "
                      f"n_heads={len(self.action_nvec)}")
            elif t == "reset":
                self._reset_h()
                send_msg(conn, {"type": "ok"})
            elif t == "step":
                actions = self._act(msg["inputs"], msg["avail"])
                send_msg(conn, {"type": "action", "actions": actions})
            elif t == "close":
                print("[policy] close")
                return
            else:
                send_msg(conn, {"type": "error", "msg": f"unknown type {t}"})

    def serve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(4)
            print(f"[policy] ✓ READY -- listening on {self.host}:{self.port}  (Ctrl-C to stop)")
            print(f"[policy]   waiting for Terminal B (env_socket) to connect ...")
            while True:
                conn, addr = s.accept()
                try:
                    self._handle(conn, addr)
                except Exception as e:
                    import traceback
                    print(f"[policy] error with {addr}: {e}")
                    traceback.print_exc()
                finally:
                    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6000)
    args = ap.parse_args()
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    PolicyServer(args.ckpt, args.host, args.port).serve()


if __name__ == "__main__":
    main()
