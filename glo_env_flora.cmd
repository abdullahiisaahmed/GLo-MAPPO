#!/usr/bin/env bash
# glo_env_flora.cmd  --  ONE-SHOT launcher: prepare the env + open FLoRa (Qtenv GUI).
#
# It bundles every setup step (strip conda, load GCC/Qt5, source OMNeT++/INET) so you
# don't have to. It is a bash script despite the .cmd name.
#
# Run from a terminal INSIDE the OnDemand / VNC desktop (the GUI needs a DISPLAY):
#     ./glo_env_flora.cmd                          # default closed-loop a2g scenario
#     ./glo_env_flora.cmd examples/other.ini       # any other .ini
#     ./glo_env_flora.cmd '' Cmdenv                 # headless (no GUI), default ini
#     ./glo_env_flora.cmd &                         # background it to keep the terminal
#
# NOTE: for the CLOSED LOOP, start Terminal A (policy_server) + B (env_socket) first.
# (No 'set -u' -- the OMNeT++/INET setenv scripts reference unbound vars by design.)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FS="$ROOT/flora-stack"

# 1) prepare the FLoRa runtime env (strips conda, loads GCC/Qt5, OMNeT++/INET libs)
source "$FS/use_flora.sh"

# 2) GUI needs a DISPLAY
INI="${1:-examples/marl-2gw-50ed-a2g.ini}"
UI="${2:-Qtenv}"
if [ "$UI" = "Qtenv" ] && [ -z "${DISPLAY:-}" ]; then
  echo "!! DISPLAY is empty -- the GUI will not show. Open this from the OnDemand/VNC desktop." >&2
  echo "   (or run headless:  ./glo_env_flora.cmd '$INI' Cmdenv )" >&2
fi

# 3) launch
echo ">> launching FLoRa ($UI) on $INI"
cd "$FS/flora/simulations"
exec ./run -u "$UI" -f "$INI"
