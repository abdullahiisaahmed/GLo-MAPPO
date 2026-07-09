#!/usr/bin/env python3
"""Per-episode logger shared by the FLoRa closed-loop bridge (env_socket.py) and the
pure-Python driver, so the two runs use an IDENTICAL JSON schema and are directly
comparable. One JSON file per episode; summarize with report_cosim_comparison.py.

Records, per episode:
  - timing_ms       : per-step summaries (mean/std/p50/p95/min/max/n) for each phase
  - mission         : steps_to_complete, mission_time_s, per-UAV arrival step/time, etc.
  - episode_quality : coverage, PDR, EE, datarate, propulsion, collisions, duty-cycle
"""
import csv
import datetime
import json
import os
import statistics
import uuid

import numpy as np

# episode-quality keys pulled from env.get_stats() (only those present are kept)
QUALITY_KEYS = [
    "coverage_rate", "pdr", "global_energy_efficiency", "total_system_datarate",
    "uavs_propulsion_power", "unique_eds_served", "uavs_reached_cs", "uav_collisions",
    "pkt_collision_rate", "ed_dc_usage_max",
]


def _summ(values):
    """mean/std/p50/p95/min/max/n over a list of per-step times (ms)."""
    a = sorted(float(x) for x in values if x is not None)
    if not a:
        return None
    pct = lambda p: a[min(len(a) - 1, int(p * len(a)))]
    return {"mean": statistics.mean(a),
            "std": statistics.pstdev(a) if len(a) > 1 else 0.0,
            "p50": pct(0.50), "p95": pct(0.95),
            "min": a[0], "max": a[-1], "n": len(a)}


def _scalar(v):
    a = np.asarray(v).reshape(-1)
    return float(a[0]) if a.size else float("nan")


def build_record(mode, n_uavs, n_eds, delta_t, timing_ms, arrival_steps,
                 done_step, steps_run, env_stats, extra=None):
    """Assemble the per-episode record. arrival_steps[u] = step index at which UAV u
    first reached the CS (None if it never did). done_step = step the episode ended."""
    rec = {
        "mode": mode,                       # "closed_loop" (FLoRa) or "python"
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_uavs": n_uavs, "n_eds": n_eds, "delta_t_s": delta_t,
        "timing_ms": {k: _summ(v) for k, v in timing_ms.items()},
        "mission": {
            "steps_run": steps_run,
            "steps_to_complete": done_step,
            "mission_time_s": (done_step * delta_t) if done_step is not None else None,
            "uav_arrival_step": list(arrival_steps),
            "uav_arrival_time_s": [s * delta_t if s is not None else None for s in arrival_steps],
            "num_reached_cs": int(sum(s is not None for s in arrival_steps)),
            "all_reached_cs": bool(all(s is not None for s in arrival_steps)),
        },
        "episode_quality": {k: _scalar(env_stats[k]) for k in QUALITY_KEYS if k in env_stats},
    }
    if extra:
        rec.update(extra)
    return rec


def write_record(rec, log_dir):
    """Write the per-episode JSON and append a flat row to summary.csv."""
    os.makedirs(log_dir, exist_ok=True)
    ts = rec["timestamp"].replace(":", "").replace("-", "")
    # microseconds + short random token so fast (sub-second) episodes never collide/overwrite
    uniq = f"{datetime.datetime.now().strftime('%f')}{uuid.uuid4().hex[:4]}"
    path = os.path.join(log_dir, f"episode_{rec['mode']}_{ts}_{uniq}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)

    # flat one-line CSV row for quick eyeballing / spreadsheets
    pol = (rec["timing_ms"].get("policy_inference") or {})
    row = {
        "mode": rec["mode"], "timestamp": rec["timestamp"],
        "mission_time_s": rec["mission"]["mission_time_s"],
        "steps_to_complete": rec["mission"]["steps_to_complete"],
        "num_reached_cs": rec["mission"]["num_reached_cs"],
        "policy_ms_mean": pol.get("mean"), "policy_ms_p95": pol.get("p95"),
        **{f"q_{k}": v for k, v in rec["episode_quality"].items()},
    }
    csv_path = os.path.join(log_dir, "summary.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    return path
