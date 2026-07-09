#!/usr/bin/env python3
"""Channel-fidelity validation: GLo-MAPPO in the standalone (analytic) simulator vs the
FLoRa closed loop at three channel tiers.

Reads the per-episode JSON logs written by:
  - run_python_eval_logged.py  -> mode="python"  (GLo-MAPPO standalone reference)
  - env_socket.py --tier 1     -> mode="tier1"    (analytic A2G mean, deterministic)
  - env_socket.py --tier 2     -> mode="tier2"    (analytic A2G + log-normal shadowing)
  - env_socket.py --tier 3     -> mode="tier3"    (FLoRa native Oulu terrestrial model)
(a plain closed-loop run with no --tier is tagged "closed_loop" and shown too, if present).

Compares 3 validation metrics -- PDR, Energy Efficiency, Coverage -- each stressing a
different subsystem (MAC/PHY reliability, energy, channel/geometry), and reports each
tier's discrepancy vs the standalone reference. Emits a console table + CSV.

    python report_cosim_comparison.py --logdir timing_logs
"""
import argparse
import csv
import glob
import json
import os
import statistics

# (quality-key, display label, scale divide)
METRICS = [
    ("pdr",                       "PDR (%)",      0.01),
    ("global_energy_efficiency",  "EE (Mbit/J)",  1e6),
    ("coverage_rate",             "Coverage (%)", 0.01),
]

# columns rendered (in order) if present in the logs; "python" is the reference.
COLUMN_ORDER = ["python", "tier1", "tier2", "tier3", "closed_loop"]
COLUMN_TITLE = {"python": "GLo-MAPPO (Standalone)", 
                "tier1": "Tier-1 (A2G mean)",
                "tier2": "Tier-2 (A2G+shadow)", 
                "tier3": "Tier-3 (Oulu)",
                "closed_loop": "Closed-loop"}


def load(logdir):
    data = {}
    for fp in sorted(glob.glob(os.path.join(logdir, "episode_*.json"))):
        try:
            rec = json.load(open(fp))
        except Exception:
            continue
        data.setdefault(rec.get("mode"), []).append(rec.get("episode_quality", {}))
    return data


def series(records, key, scale):
    return [r[key] / scale for r in records if key in r and r[key] == r[key]]


def agg(vals):
    if not vals:
        return None
    return (statistics.mean(vals),
            statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "timing_logs"))
    ap.add_argument("--out", default="cosim_comparison")
    args = ap.parse_args()

    data = load(args.logdir)
    cols = [m for m in COLUMN_ORDER if m in data]           # present modes, in order
    counts = {m: len(data[m]) for m in cols}
    print("episodes found:  " + "  ".join(f"{m}={counts[m]}" for m in cols) + "\n")
    if "python" not in cols or not any(c in cols for c in ("tier1", "tier2", "closed_loop")):
        print("Need the python reference AND at least one closed-loop mode (tier1/tier2).")
        print("Run:  run_python_eval_logged.py  +  env_socket.py --tier 1  +  --tier 2")
        return

    tiers = [m for m in cols if m != "python"]              # closed-loop columns to score

    # ---- console ----
    head = f"{'Metric':13}" + f"{COLUMN_TITLE['python']:>20}"
    for m in tiers:
        head += f"{COLUMN_TITLE[m]:>18}{'Δ%':>8}"
    print(head)
    print("-" * len(head))
    table = []   # (label, {mode: (mean,std,n)}, {tier: disc})
    for key, label, scale in METRICS:
        cell = {m: agg(series(data[m], key, scale)) for m in cols}
        disc = {}
        line = f"{label:13}"
        p = cell["python"]
        line += f"{p[0]:11.3f}+-{p[1]:<6.3f}" if p else f"{'--':>20}"
        for m in tiers:
            c = cell[m]
            if c and p and p[0] != 0:
                d = abs(c[0] - p[0]) / abs(p[0]) * 100
            else:
                d = float("nan")
            disc[m] = d
            line += (f"{c[0]:9.3f}+-{c[1]:<5.3f}" if c else f"{'--':>16}") + f"{d:7.2f}%"
        print(line)
        table.append((label, cell, disc))

    # ---- CSV ----
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["Metric"]
        for m in cols:
            hdr += [f"{m}_mean", f"{m}_std", f"{m}_n"]
        for m in tiers:
            hdr += [f"{m}_discrepancy_pct"]
        w.writerow(hdr)
        for label, cell, disc in table:
            row = [label]
            for m in cols:
                c = cell[m]
                row += ([f"{c[0]:.4f}", f"{c[1]:.4f}", c[2]] if c else ["", "", ""])
            for m in tiers:
                row += [f"{disc[m]:.2f}"]
            w.writerow(row)

    print(f"\nwrote {args.out}.csv")


if __name__ == "__main__":
    main()
