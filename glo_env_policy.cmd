#!/usr/bin/env bash
# glo_env_policy.cmd  --  Terminal A: the policy "brain" (obs -> action).
#
# The closed-loop POLICY server. Uses the conda marl_lora python (NOT use_flora.sh).
# Start this FIRST, before Terminal B (env_socket) and FLoRa.
#
#     ./glo_env_policy.cmd                              # defaults: ckpt=DEFAULT_CKPT, port 6000
#     ./glo_env_policy.cmd --ckpt /path/to/agent.th     # a specific trained checkpoint
#     ./glo_env_policy.cmd --port 7000                   # different port
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/.conda/envs/marl_lora/bin/python"
[ -x "$PY" ] || { echo "!! marl_lora python not found at $PY (conda env missing?)" >&2; exit 1; }

echo ">> Terminal A: policy_server (port 6000) -- start this FIRST"
cd "$ROOT"
exec "$PY" -u flora_run/policy_server.py "$@"
