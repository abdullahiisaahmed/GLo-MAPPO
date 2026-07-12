#!/usr/bin/env python3

import argparse
import os
import time

from marl_core import MarlController, ENV_KWARGS
import episode_logger

# Use the exact checkpoint the closed loop serves, so both sides run identical weights.
from policy_server import DEFAULT_CKPT as CLOSED_LOOP_CKPT


def run_episode(ctrl):
    """Drive one greedy episode; return mission-timing + end-of-episode env stats."""
    ctrl.reset()
    n = ctrl.n_agents
    arrival = [None] * n            # first step each UAV reached the CS
    done_step = None
    step_ms = []
    max_steps = ctrl.env.max_episode_steps
    for _ in range(max_steps):
        t0 = time.perf_counter()
        snap = ctrl.step()
        step_ms.append((time.perf_counter() - t0) * 1e3)
        ts = int(ctrl.env.timestep)
        for u in range(n):
            if arrival[u] is None and bool(ctrl.env.has_reached_cs[u]):
                arrival[u] = ts
        if snap["done"] and done_step is None:
            done_step = ts
        if snap["done"]:
            break
    return arrival, done_step, int(ctrl.env.timestep), step_ms, ctrl.env.get_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--ckpt", default=CLOSED_LOOP_CKPT,
                    help="defaults to the checkpoint policy_server.py serves")
    ap.add_argument("--logdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "timing_logs"))
    args = ap.parse_args()

    print(f"[python-eval] checkpoint : {args.ckpt}")
    print(f"[python-eval] logdir     : {args.logdir}")
    ctrl = MarlController(checkpoint=args.ckpt, env_kwargs=ENV_KWARGS)
    delta_t = float(ctrl.env.config["delta_t"])
    print(f"[python-eval] n_uavs={ctrl.n_agents}  n_eds={ctrl.env.num_eds}  "
          f"delta_t={delta_t}  episodes={args.episodes}\n")

    for ep in range(args.episodes):
        arrival, done_step, steps_run, step_ms, stats = run_episode(ctrl)
        rec = episode_logger.build_record(
            mode="python", n_uavs=ctrl.n_agents, n_eds=ctrl.env.num_eds,
            delta_t=delta_t,
            timing_ms={"python_step": step_ms},   # full python control-step time
            arrival_steps=arrival, done_step=done_step,
            steps_run=steps_run, env_stats=stats)
        path = episode_logger.write_record(rec, args.logdir)
        q = rec["episode_quality"]
        print(f"  ep {ep + 1:2d}/{args.episodes}: "
              f"EE={q.get('global_energy_efficiency', float('nan')):.3e}  "
              f"PDR={q.get('pdr', float('nan')):.3f}  "
              f"cov={q.get('coverage_rate', float('nan')):.3f}  "
              f"reached_cs={rec['mission']['num_reached_cs']}  "
              f"-> {os.path.basename(path)}")
    print("\n[python-eval] done. Now compare with: python report_cosim_comparison.py")


if __name__ == "__main__":
    main()
