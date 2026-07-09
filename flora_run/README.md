# `flora_run/` — Closed-Loop Co-Simulation Bridge

The Python side of the GLo-MAPPO ⇄ FLoRa closed loop. It serves the trained policy and relays observations/actions between the FLoRa/OMNeT++ simulator and the environment, so the policy drives the two UAV gateways live inside a packet-level LoRa network.

## Files

| File | Role |
|---|---|
| `policy_server.py`   | **Terminal A.** Loads the trained actor (`models/glo_ed50_seed41/agent.th`) and serves greedy actions over TCP `:6000`. Left running across all tiers. |
| `env_socket.py`      | **Terminal B.** The env sidecar: holds `MultiFlyingLoRaEnv`, injects FLoRa's measured channel, relays obs ↔ action, and applies the tier (`--tier 1\|2\|3`). Listens for FLoRa on `:5000` and writes the scenario topology FLoRa reads. |
| `marl_core.py`       | Shared core: loads the policy checkpoint + actor network and constructs the environment with the exact training kwargs. Imported by the two servers and the standalone eval. |
| `episode_logger.py`  | Shared per-episode metrics logger (PDR / energy efficiency / coverage / timing) → one JSON per episode under `timing_logs/`, plus a flat `summary.csv`. Same schema for standalone and closed-loop runs so they are directly comparable. |
| `run_python_eval_logged.py` | **Standalone reference eval** (no FLoRa): runs the same policy over the Python environment and logs the same metrics — the baseline the tiers are compared against. |
| `report_cosim_comparison.py` | Reads the `timing_logs/` JSONs and prints/saves the **Standalone vs Tier-1/2/3** comparison table (per-tier PDR/EE/coverage discrepancy). |

## Runtime outputs

- `timing_logs/` — per-episode JSON logs + `summary.csv` (auto-created on first run).
- `scenario.json` — topology written by `env_socket.py` for FLoRa (regenerated each run).
- `cosim_comparison.csv` — the table emitted by `report_cosim_comparison.py`.

## Channel tiers

`env_socket.py --tier N` must match the `glo-2gw-50ed-a2g-tierN.ini` launched in FLoRa:

| Tier | Channel |
|---|---|
| 1 | analytic air-to-ground (A2G) |
| 2 | A2G + log-normal shadowing (σ = 8 dB) |
| 3 | native Oulu terrestrial path loss |
