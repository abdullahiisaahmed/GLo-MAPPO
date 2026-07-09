#!/usr/bin/env bash
# glo_env_socket.cmd  --  Terminal B: the env sidecar (FLoRa <-> policy).
#
# Injects FLoRa's measured channel, builds the obs, calls Terminal A for the action,
# steps the env, replies to FLoRa. Uses the conda marl_lora python (NOT use_flora.sh).
# Start this SECOND (after Terminal A / policy_server), then launch FLoRa.
#
#     ./glo_env_socket.cmd                               # defaults: policy:6000, FLoRa:5000, --timing on
#     ./glo_env_socket.cmd --policy-port 7000            # match a non-default policy port
#     ./glo_env_socket.cmd --port 5001                   # different FLoRa-facing port
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/.conda/envs/marl_lora/bin/python"
[ -x "$PY" ] || { echo "!! marl_lora python not found at $PY (conda env missing?)" >&2; exit 1; }

echo ">> Terminal B: env_socket (policy:6000 -> FLoRa:5000) -- start AFTER Terminal A"
cd "$ROOT"
exec "$PY" -u flora_run/env_socket.py --timing "$@"
