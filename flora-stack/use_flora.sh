#!/usr/bin/env bash
# Source this in an OnDemand / VNC desktop terminal to prepare a shell for
# running FLoRa (Qtenv GUI or headless Cmdenv):
#
#     source ~/flora-stack/use_flora.sh
#     cd ~/flora-stack/flora/simulations
#     ./run -u Qtenv omnetpp.ini          # GUI
#     ./run -u Cmdenv omnetpp.ini         # headless
#
# NOTE: this is meant to be *sourced*, not executed. It does NOT set -e.

# auto-detect this script's own directory = the flora-stack root (robust to moves)
FS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) strip conda — its libstdc++ (GLIBCXX_3.4.29) shadows GCC 12.3's 3.4.30
#    and the FLoRa binaries fail at runtime with "GLIBCXX_3.4.30 not found".
if [ -n "${CONDA_PREFIX:-}" ]; then
  echo ">> removing conda (${CONDA_DEFAULT_ENV:-?}) from this shell for FLoRa"
  PATH="$(echo "$PATH" | tr ':' '\n' | grep -vi "$CONDA_PREFIX" | grep -vi '/\.conda/' | grep -vi anaconda | paste -sd: -)"
  [ -n "${LD_LIBRARY_PATH:-}" ] && \
    LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -vi "$CONDA_PREFIX" | grep -vi anaconda | paste -sd: -)"
  export PATH LD_LIBRARY_PATH
  unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH
fi

# 2) toolchain + Qt5 (Qt5 = the GUI runtime libs)
#    Only init lmod if 'module' isn't already defined (re-sourcing lmod.sh can
#    clobber MODULEPATH and hide the EasyBuild GCC tree).
command -v module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh
#    rhel8/global puts the EasyBuild software tree on MODULEPATH so GCC is visible.
module load rhel8/global 2>/dev/null || true
module load GCC/12.3.0 Qt5/5.15.10-GCCcore-12.3.0

# verify GCC actually loaded — the FLoRa binaries need its libstdc++ (GLIBCXX_3.4.30)
if [ -z "${EBROOTGCCCORE:-}" ] || [ "$(command -v g++)" = "/usr/bin/g++" ]; then
  echo "!! GCC/12.3.0 did NOT load (g++ = $(command -v g++))." >&2
  echo "   FLoRa will fail at runtime with 'GLIBCXX_3.4.30 not found'." >&2
  echo "   Fix: in this shell run 'module load GCC/12.3.0' and check 'module spider GCC/12.3.0'." >&2
else
  echo ">> GCC ready: $(g++ --version | head -1)"
fi

# 3) OMNeT++ and INET environments (sourced; no set -e so quirks are harmless)
source "$FS/omnetpp-6.0.3/setenv" -f
source "$FS/inet4.4/setenv" -f

# 4) The stack was built at a different path then moved here, so the absolute
#    rpath baked into the binaries is stale. Point LD_LIBRARY_PATH at the CURRENT
#    lib locations (overrides the dead rpath) for OMNeT++, INET and FLoRa libs.
export LD_LIBRARY_PATH="$FS/omnetpp-6.0.3/lib:$FS/omnetpp-6.0.3/tools/linux.x86_64/lib:$FS/inet4.4/src:$FS/flora/src:${LD_LIBRARY_PATH:-}"

echo ">> FLoRa env ready."
echo "   DISPLAY = '${DISPLAY:-<empty - GUI will NOT show; are you in the OnDemand desktop?>}'"
echo "   Launch:  cd $FS/flora/simulations && ./run -u Qtenv omnetpp.ini"
