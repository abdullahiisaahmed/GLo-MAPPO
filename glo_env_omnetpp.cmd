#!/usr/bin/env bash
# glo_env_omnetpp.cmd  --  ONE-SHOT launcher: prepare the env + open the OMNeT++ IDE.
#
# It bundles every setup step (strip conda, load GCC/Qt5 + Java, source OMNeT++/INET)
# so you don't have to. It is a bash script despite the .cmd name.
#
# Run from a terminal INSIDE the OnDemand / VNC desktop (the IDE needs a DISPLAY):
#     ./glo_env_omnetpp.cmd        # opens the IDE (terminal stays attached)
#     ./glo_env_omnetpp.cmd &      # background it to keep the terminal free
# (No 'set -u' -- the OMNeT++/INET setenv scripts reference unbound vars by design.)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FS="$ROOT/flora-stack"

# 1) prepare the env (strips conda, loads GCC/Qt5, OMNeT++/INET libs, puts omnetpp on PATH)
source "$FS/use_flora.sh"

# 2) the Eclipse-based IDE needs a JRE
module load Java/17.0.15 2>/dev/null || module load Java 2>/dev/null || true

# 3) GUI needs a DISPLAY
if [ -z "${DISPLAY:-}" ]; then
  echo "!! DISPLAY is empty -- the IDE will not show. Open this from the OnDemand/VNC desktop." >&2
fi

echo ">> launching the OMNeT++ IDE ..."
exec omnetpp
