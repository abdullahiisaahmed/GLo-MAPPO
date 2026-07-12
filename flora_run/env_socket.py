#!/usr/bin/env python3

import argparse
import json
import os
import signal
import socket
import statistics
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from marl_core import _load_env_class, ENV_KWARGS, OBS_AGENT_ID
import policy_server as P   # reuse send_msg / recv_msg / recv_exact framing
import episode_logger       # shared per-episode JSON logger

MAGIC = 0x4D41524C
MSG_STEP, MSG_RESET, MSG_CLOSE = 0, 1, 2
REQ_HDR_FMT = "<iiiid"
REQ_HDR_SIZE = struct.calcsize(REQ_HDR_FMT)       


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        c = conn.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


class EnvSocket:
    def __init__(self, policy_host, policy_port, host, port, scenario_path, timing=False,
                 log_dir=None, mode_label="closed_loop"):
        self.host, self.port = host, port
        self.timing = timing                 # log per-step FLoRa/policy/socket times
        self._t_flora, self._t_policy, self._t_bridge = [], [], []
        self.log_dir = log_dir               # if set, write a per-episode JSON log here
        self.mode_label = mode_label         # log tag: "tier1"/"tier2" (FLoRa channel tier) or "closed_loop"
        self.episode_index = 0               # our own episode counter (each FLoRa run = 1 episode)
        self._logged = False                 # guard against double-write (CLOSE + disconnect)
        print("=" * 62)
        print("  GLo-MAPPO  ENV SOCKET   (Terminal B -- FLoRa <-> policy pipeline)")
        print("=" * 62)
        print("[socket] Loading environment (multi_lora) ...")
        self.env = _load_env_class()(**ENV_KWARGS)
        self.n_agents = self.env.n_agents
        self.num_eds = self.env.num_eds
        self.delta_t = float(self.env.config["delta_t"])      # control-step duration [s]
        self.arrival_step = [None] * self.n_agents            # first step each UAV reached CS
        self.done_step = None                                 # step the episode terminated
        self.action_nvec = list(self.env.action_nvec)
        self.obs_flat = int(np.prod(self.env.observation_space[0].shape))
        self.input_shape = self.obs_flat + (self.n_agents if OBS_AGENT_ID else 0)
        self.area_y = float(self.env.area_size[1])

        # request: header + (n_eds * n_uavs) per-(ed,uav) measurements
        self.meas_count = self.num_eds * self.n_agents
        self.req_meas_fmt = "<" + "ddi" * self.meas_count
        self.req_size = REQ_HDR_SIZE + struct.calcsize(self.req_meas_fmt)
        # response (identical to marl_server.py)
        self.resp_header_fmt = "<iiii"
        self.resp_uav_fmt = "<" + "dd" * self.n_agents
        self.resp_ed_fmt = "<" + "iii" * self.num_eds
        self.resp_size = (struct.calcsize(self.resp_header_fmt)
                          + struct.calcsize(self.resp_uav_fmt)
                          + struct.calcsize(self.resp_ed_fmt))

        print(f"[socket] ✓ Environment ready  ({self.n_agents} UAVs x {self.num_eds} EDs)")
        self.env.reset()
        print("[socket] Writing scenario topology for FLoRa ...")
        self._write_scenario(scenario_path)

        # connect to the pure policy (Terminal A) and handshake
        print(f"[socket] Connecting to policy server (Terminal A) at {policy_host}:{policy_port} ...")
        self.psock = socket.create_connection((policy_host, policy_port))
        P.send_msg(self.psock, {"type": "hello", "action_nvec": self.action_nvec,
                                "n_agents": self.n_agents})
        ok = P.recv_msg(self.psock)
        if not ok or ok.get("type") != "ok" or ok.get("input_shape") != self.input_shape:
            raise RuntimeError(f"policy handshake failed / dim mismatch: {ok} "
                               f"(socket input_shape={self.input_shape})")
        print(f"[socket] ✓ Policy server connected & handshake OK  "
              f"(input_shape={ok['input_shape']} n_actions={ok['n_actions']})")
        print(f"[socket] dims: {self.n_agents} UAVs x {self.num_eds} EDs | "
              f"req={self.req_size}B resp={self.resp_size}B")

    # ---- topology export (same content as marl_server.py) ----
    def _write_scenario(self, path):
        env = self.env
        data = {
            "num_uavs": self.n_agents, "num_eds": self.num_eds,
            "area_size": [float(env.area_size[0]), float(env.area_size[1])],
            "uav_altitude": float(env.uav_altitude), "comm_range": float(env.comm_range),
            "frequency_hz": float(env.config["frequency"]),
            "bandwidth_hz": float(env.config["bandwidth"]),
            "sf_options": [int(s) for s in env.config["sf_options"]],
            "tp_options": [int(t) for t in env.config["tp_options"]],
            "delta_t": float(env.config["delta_t"]),
            "max_episode_steps": int(env.max_episode_steps),
            "cs_position": [float(env.cs_position[0]), float(env.cs_position[1])],
            "ed_positions": [[float(p[0]), float(p[1])] for p in env.ed_positions],
            "uav_initial_positions": [[float(p[0]), float(p[1])] for p in env.uav_positions],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[socket] scenario written -> {path}")

    # ---- helpers ----
    def _build_inputs(self, obs):
        rows = []
        for a in range(self.n_agents):
            flat = np.asarray(obs[a], dtype=np.float32).reshape(-1)
            if OBS_AGENT_ID:
                oh = np.zeros(self.n_agents, dtype=np.float32)
                oh[a] = 1.0
                flat = np.concatenate([flat, oh])
            rows.append(flat)
        return np.stack(rows)

    def _parse_measurements(self, body):
        """body -> (gain[n_agents,num_eds] linear, fresh[n_agents,num_eds] bool).
        FLoRa sends the channel gain ED->UAV in dB, so gain_linear = 10^(gain_db/10)
        (TP-independent). Layout: ed-major, uav-minor."""
        vals = struct.unpack(self.req_meas_fmt, body)
        gain = np.zeros((self.n_agents, self.num_eds), dtype=np.float64)
        fresh = np.zeros((self.n_agents, self.num_eds), dtype=bool)
        for e in range(self.num_eds):
            for u in range(self.n_agents):
                k = 3 * (e * self.n_agents + u)
                gain_db, _snr, fr = vals[k], vals[k + 1], vals[k + 2]
                if fr:
                    gain[u, e] = 10.0 ** (gain_db / 10.0)
                    fresh[u, e] = True
        return gain, fresh

    def _snapshot(self, done):
        env = self.env
        uav_xy = [(float(p[0]), float(p[1])) for p in env.uav_positions]
        sf = np.asarray(env.sf_allocations_all_eds, dtype=int)
        tp = np.asarray(env.tp_allocations_all_eds, dtype=float)
        assoc = np.full(self.num_eds, -1, dtype=int)
        for u in range(self.n_agents):
            for e in env.associations[u].nonzero()[1]:
                assoc[e] = u
        return {"done": done, "uav_xy": uav_xy, "ed_sf": sf, "ed_tp": tp, "ed_assoc": assoc}

    def _pack(self, snap):
        done = 1 if snap["done"] else 0
        buf = struct.pack(self.resp_header_fmt, MAGIC, done, self.n_agents, self.num_eds)
        uav_vals = []
        for (x, y) in snap["uav_xy"]:
            uav_vals += [float(x), self.area_y - float(y)]      # flip y for FLoRa display
        buf += struct.pack(self.resp_uav_fmt, *uav_vals)
        ed_vals = []
        for e in range(self.num_eds):
            ed_vals += [int(snap["ed_sf"][e]),
                        int(round(float(snap["ed_tp"][e]))),
                        int(snap["ed_assoc"][e])]
        buf += struct.pack(self.resp_ed_fmt, *ed_vals)
        return buf

    # ---- one control step: FLoRa measurement -> obs -> policy -> action -> FLoRa ----
    def _do_step(self, body):
        t0 = time.perf_counter()
        gain, fresh = self._parse_measurements(body)
        self.env.set_measured_gain(gain, fresh)          # inject measured channel
        obs = self.env.get_obs()                          # rebuild obs w/ measured gain
        avail = self.env.get_avail_actions()
        inputs = self._build_inputs(obs)
        t_pol = time.perf_counter()
        P.send_msg(self.psock, {"type": "step", "inputs": inputs, "avail": avail})
        reply = P.recv_msg(self.psock)                    # Terminal A (policy) round-trip
        policy_ms = (time.perf_counter() - t_pol) * 1000.0
        actions = [np.asarray(a) for a in reply["actions"]]
        _obs, _r, done, _info = self.env.step(actions)
        # mission timing: record the first step each UAV reaches the CS, and episode end
        ts = int(self.env.timestep)
        for u in range(self.n_agents):
            if self.arrival_step[u] is None and bool(self.env.has_reached_cs[u]):
                self.arrival_step[u] = ts
        if done and self.done_step is None:
            self.done_step = ts
        snap = self._snapshot(bool(done))
        bridge_ms = (time.perf_counter() - t0) * 1000.0 - policy_ms   # env-side compute
        return snap, policy_ms, bridge_ms

    @staticmethod
    def _alloc_summary(snap):
        """Compact SF/TP histogram over the EDs served this step."""
        served = snap["ed_assoc"] >= 0
        sf = snap["ed_sf"][served]
        tp = np.round(snap["ed_tp"][served]).astype(int)
        sf_h = {int(s): int((sf == s).sum()) for s in sorted(set(sf.tolist())) if s > 0}
        tp_h = {int(t): int((tp == t).sum()) for t in sorted(set(tp.tolist()))}
        return f"SF{sf_h} TP{tp_h}"

    def _timing_summary(self):
        if not self.timing or not self._t_flora:
            return
        f = statistics.mean(self._t_flora)
        a = statistics.mean(self._t_policy)
        b = statistics.mean(self._t_bridge)
        tot = f + a + b
        print(f"\n[timing] ==== per control step (means over {len(self._t_flora)} steps, "
              "warmup skipped) ====")
        print(f"[timing]  FLoRa (sim+apply+measure) : {f:8.1f} ms")
        print(f"[timing]  A (policy round-trip)     : {a:8.1f} ms")
        print(f"[timing]  socket (inject+obs+step)  : {b:8.1f} ms")
        print(f"[timing]  total per step            : {tot:8.1f} ms   "
              f"(FLoRa {100*f/tot:.0f}% | A {100*a/tot:.0f}% | socket {100*b/tot:.0f}%)")

    def _write_episode_log(self):
        """Persist this episode's timing + mission + quality metrics to JSON (once)."""
        if not self.log_dir or self._logged:
            return
        try:
            rec = episode_logger.build_record(
                mode=self.mode_label, n_uavs=self.n_agents, n_eds=self.num_eds,
                delta_t=self.delta_t,
                timing_ms={"policy_inference": self._t_policy,
                           "flora_step": self._t_flora,
                           "bridge_env": self._t_bridge},
                arrival_steps=self.arrival_step, done_step=self.done_step,
                steps_run=int(self.env.timestep), env_stats=self.env.get_stats())
            path = episode_logger.write_record(rec, self.log_dir)
            self._logged = True
            print(f"[socket] episode log written -> {path}")
        except Exception as e:
            print(f"[socket] WARN: could not write episode log: {e}")

    def _handle(self, conn, addr):
        print(f"[socket] FLoRa connected: {addr}")
        steps = 0
        policy_ms = bridge_ms = 0.0
        while True:
            t_recv = time.perf_counter()
            hdr = recv_exact(conn, REQ_HDR_SIZE)
            wait_ms = (time.perf_counter() - t_recv) * 1000.0   # FLoRa apply+PHY+measure
            if hdr is None:
                self._timing_summary()
                self._write_episode_log()
                print(f"[socket] FLoRa disconnected after {steps} steps")
                return
            magic, mtype, ep, step, simt = struct.unpack(REQ_HDR_FMT, hdr)
            if magic != MAGIC:
                print(f"[socket] BAD MAGIC {magic:#x}; closing")
                return
            # STEP/RESET both carry the measurement block (RESET's is ignored)
            body = recv_exact(conn, struct.calcsize(self.req_meas_fmt))
            if body is None:
                print("[socket] short read on measurement block; closing")
                return
            if mtype == MSG_CLOSE:
                self._timing_summary()
                self._write_episode_log()
                print(f"[socket] CLOSE (ep={self.episode_index}, {steps} steps)")
                return
            elif mtype == MSG_RESET:
                self.env.reset()
                self.episode_index += 1          # new episode (persists across FLoRa runs)
                # fresh episode -> clear mission/timing trackers
                self.arrival_step = [None] * self.n_agents
                self.done_step = None
                self._t_flora.clear(); self._t_policy.clear(); self._t_bridge.clear()
                self._logged = False
                P.send_msg(self.psock, {"type": "reset"})
                P.recv_msg(self.psock)
                snap = self._snapshot(False)
            else:
                snap, policy_ms, bridge_ms = self._do_step(body)
                steps += 1
                if self.timing and steps > 2:           # skip warmup (init + first step)
                    self._t_flora.append(wait_ms)
                    self._t_policy.append(policy_ms)
                    self._t_bridge.append(bridge_ms)
            conn.sendall(self._pack(snap))
            if mtype != MSG_RESET:
                served = int((snap["ed_assoc"] >= 0).sum())
                uav = ", ".join(f"({x:.0f},{y:.0f})" for x, y in snap["uav_xy"])
                extra = (f"  | FLoRa={wait_ms:7.0f}ms A={policy_ms:5.1f}ms socket={bridge_ms:5.1f}ms"
                         if self.timing else "")
                print(f"[socket] ep={self.episode_index} t={step:3d} served={served:2d} UAV=[{uav}] "
                      f"{self._alloc_summary(snap)} done={snap['done']}{extra}")

    def serve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(4)
            print(f"[socket] ✓ READY -- listening for FLoRa on {self.host}:{self.port}")
            print(f"[socket]   now launch FLoRa (./glo_env_flora.cmd) or OMNeT++  (./glo_env_omnetpp.cmd)")
            while True:
                conn, addr = s.accept()
                try:
                    self._handle(conn, addr)
                except Exception as e:
                    import traceback
                    print(f"[socket] error with {addr}: {e}")
                    traceback.print_exc()
                finally:
                    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-host", default="127.0.0.1")
    ap.add_argument("--policy-port", type=int, default=6000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--scenario", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "scenario.json"))
    ap.add_argument("--timing", action="store_true",
                    help="log per-step FLoRa / policy(A) / socket timing + a summary")
    ap.add_argument("--logdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "timing_logs"),
        help="dir for the per-episode JSON log (set empty '' to disable)")
    ap.add_argument("--tier", type=int, choices=(1, 2, 3), default=None,
        help="FLoRa channel tier for the log tag: 1=A2G mean, 2=A2G+shadowing, 3=Oulu terrestrial. "
             "Must MATCH the .ini's channelTier. Omit to tag logs as generic 'closed_loop'.")
    args = ap.parse_args()
    mode_label = f"tier{args.tier}" if args.tier else "closed_loop"
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    EnvSocket(args.policy_host, args.policy_port, args.host, args.port, args.scenario,
              timing=args.timing, log_dir=(args.logdir or None),
              mode_label=mode_label).serve()


if __name__ == "__main__":
    main()
