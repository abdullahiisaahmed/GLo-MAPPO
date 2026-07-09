#!/usr/bin/env bash
# =============================================================================
# Cluster install (GUI-enabled): OMNeT++ 6.0.3 -> INET 4.4.1 -> FLoRa 1.1.0
# Builds the Qtenv runtime GUI (visualize sims). Userspace, no root.
# Cross-validation target for the custom MultiFlyingLoRaEnv.
#
# Usage:
#   bash install_flora_stack.sh all        # run every phase in order
#   bash install_flora_stack.sh omnet      # only build OMNeT++ (+ Qtenv)
#   bash install_flora_stack.sh inet       # only build INET   (needs omnet done)
#   bash install_flora_stack.sh flora      # only build FLoRa  (needs inet done)
#   bash install_flora_stack.sh smoke      # run a tiny FLoRa example to verify
#
# Run on a COMPUTE node (NOT login). To SEE the GUI you also need a display
# (VNC / Open OnDemand) — the build itself does not need one.
#   e.g.  srun --pty -c 16 --mem=16G -t 02:00:00 bash install_flora_stack.sh all
# =============================================================================
# NOTE: deliberately NO '-u' (nounset). OMNeT++/INET 'setenv' scripts reference
# internal guard vars before defining them, which is fatal under nounset.
set -eo pipefail

# ---- where everything goes (writable project space) ------------------------
# Defaults to the repo's own flora-stack/ (where the custom Marl/LoRaPhy files and the
# .ini scenarios already live). Override with:  FLORA_STACK_PREFIX=/path bash install_flora_stack.sh all
PREFIX="${FLORA_STACK_PREFIX:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/flora-stack}"
OMNET_VER="6.0.3"
INET_VER="4.4.1"
FLORA_VER="1.1.0"

OMNET_DIR="$PREFIX/omnetpp-$OMNET_VER"
INET_DIR="$PREFIX/inet4.4"          # FLoRa's makefiles hardcode ../inet4.4 — keep this name
FLORA_DIR="$PREFIX/flora"
JOBS="$(nproc)"

# Module names (verified available on this cluster)
MOD_GCC="GCC/12.3.0"
MOD_CMAKE="CMake/3.26.3-GCCcore-12.3.0"
MOD_QT5="Qt5/5.15.10-GCCcore-12.3.0"

# ---- source a vendor env script WITHOUT killing the build on its quirks ----
# OMNeT++/INET setenv scripts return non-zero / touch unset vars, which would
# trip 'set -e'. Disable it just around the source, then restore.
safe_source() { set +e; source "$@"; set -e; }

# ---- strip any active conda env (it ships an older libstdc++ that clashes) --
# GCC/12.3.0 needs GLIBCXX_3.4.30; conda's libstdc++ is 3.4.29 -> "not found".
strip_conda() {
  if [ -n "${CONDA_PREFIX:-}" ]; then
    echo ">> conda env detected (${CONDA_DEFAULT_ENV:-?}) -> removing it from this build shell"
    PATH="$(echo "$PATH" | tr ':' '\n' | grep -vi "$CONDA_PREFIX" | grep -vi "/\.conda/" | grep -vi "miniconda" | grep -vi "anaconda" | paste -sd: -)"
    export PATH
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
      LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -vi "$CONDA_PREFIX" | grep -vi "/\.conda/" | grep -vi "anaconda" | paste -sd: -)"
      export LD_LIBRARY_PATH
    fi
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER PYTHONPATH
  fi
}

load_toolchain() {
  strip_conda
  safe_source /etc/profile.d/lmod.sh
  module purge >/dev/null 2>&1 || true
  module load "$MOD_GCC"
  module load "$MOD_CMAKE"
  module load "$MOD_QT5" 2>/dev/null || true   # Qtenv GUI dep; verified below

  echo ">> toolchain : $(g++ --version | head -1)"
  echo ">> g++       : $(command -v g++)"
  echo ">> Qt5 root  : ${EBROOTQT5:-<NOT LOADED>}"

  # Fail FAST if Qt5 didn't load — otherwise OMNeT++ silently builds without the
  # GUI and you only find out at the end. EBROOTQT5 is set ONLY by the Qt5 module
  # (don't trust 'qmake', which Anaconda also provides).
  if [ -z "${EBROOTQT5:-}" ]; then
    echo "!! Qt5 module ($MOD_QT5) did not load — the Qtenv GUI cannot be built." >&2
    echo "   In your interactive shell, check:  module spider $MOD_QT5" >&2
    echo "   Then fix the module name at the top of this script and re-run." >&2
    exit 1
  fi
}

phase_omnet() {
  echo "==================== PHASE 1: OMNeT++ $OMNET_VER (with Qtenv GUI) ===================="
  mkdir -p "$PREFIX"; cd "$PREFIX"
  if [ ! -d "$OMNET_DIR" ]; then
    echo ">> downloading OMNeT++ core tarball..."
    wget -q --show-progress -O "omnetpp-$OMNET_VER-core.tgz" \
      "https://github.com/omnetpp/omnetpp/releases/download/omnetpp-$OMNET_VER/omnetpp-$OMNET_VER-core.tgz"
    tar xzf "omnetpp-$OMNET_VER-core.tgz"
  fi
  cd "$OMNET_DIR"
  safe_source setenv -f
  # OMNeT++ 6 reads WITH_* flags from configure.user (NOT env vars). Set them there:
  #   QTENV=yes  -> build the GUI;  OSG/OSGEARTH=no -> skip 3D (not installed/needed)
  echo ">> setting WITH_QTENV=yes, WITH_OSG=no, WITH_OSGEARTH=no in configure.user..."
  sed -i 's/^WITH_QTENV=.*/WITH_QTENV=yes/'       configure.user
  sed -i 's/^WITH_OSG=.*/WITH_OSG=no/'            configure.user
  sed -i 's/^WITH_OSGEARTH=.*/WITH_OSGEARTH=no/'  configure.user
  grep -E '^WITH_(QTENV|OSG|OSGEARTH)=' configure.user | sed 's/^/   /'
  echo ">> configuring..."
  ./configure
  # sanity: confirm configure enabled Qtenv
  if grep -q '^WITH_QTENV=yes' Makefile.inc 2>/dev/null; then
    echo ">> Qtenv ENABLED in build config."
  else
    echo "!! WARNING: configure did not enable Qtenv — check Qt5 detection above." >&2
  fi
  echo ">> building OMNeT++ (this is the long one, ~20-40 min)..."
  make -j"$JOBS" MODE=release
  echo ">> OMNeT++ done. opp_run: $(command -v opp_run)"
}

phase_inet() {
  echo "==================== PHASE 2: INET $INET_VER ===================="
  safe_source "$OMNET_DIR/setenv" -f
  cd "$PREFIX"
  if [ ! -d "$INET_DIR" ]; then
    echo ">> downloading INET..."
    wget -q --show-progress -O "inet-$INET_VER-src.tgz" \
      "https://github.com/inet-framework/inet/releases/download/v$INET_VER/inet-$INET_VER-src.tgz"
    tar xzf "inet-$INET_VER-src.tgz"     # extracts to inet4.4
  fi
  cd "$INET_DIR"
  safe_source setenv -f
  echo ">> generating makefiles + building INET (~15-30 min)..."
  make makefiles
  make -j"$JOBS" MODE=release
  echo ">> INET done. INET_ROOT=$INET_ROOT"
}

phase_flora() {
  echo "==================== PHASE 3: FLoRa $FLORA_VER ===================="
  safe_source "$OMNET_DIR/setenv" -f
  safe_source "$INET_DIR/setenv" -f     # sets INET_ROOT; FLoRa links against it
  cd "$PREFIX"

  # This repo ships ONLY the custom overlay (src/Marl, src/LoRaPhy, MarlLoRaNetwork.ned,
  # simulations/examples/*.ini) inside $FLORA_DIR -- not a full FLoRa tree. Detect that by
  # the absence of FLoRa's top-level Makefile, then download stock FLoRa and MERGE it in,
  # letting our overlay win on any overlapping file.
  if [ ! -f "$FLORA_DIR/Makefile" ]; then
    OVERLAY=""
    if [ -d "$FLORA_DIR" ]; then
      echo ">> found custom FLoRa overlay in $FLORA_DIR -- preserving it"
      OVERLAY="$PREFIX/.flora_overlay_backup"
      rm -rf "$OVERLAY"; cp -a "$FLORA_DIR" "$OVERLAY"
      rm -rf "$FLORA_DIR"
    fi
    echo ">> downloading stock FLoRa..."
    wget -q --show-progress -O "flora-$FLORA_VER.tar.gz" \
      "https://github.com/florasim/flora/archive/refs/tags/v$FLORA_VER.tar.gz"
    tar xzf "flora-$FLORA_VER.tar.gz"
    mv "flora-$FLORA_VER" "$FLORA_DIR"   # sibling of inet4.4 so ../inet4.4 resolves
    if [ -n "$OVERLAY" ]; then
      echo ">> re-applying custom overlay on top of stock FLoRa (overlay wins)..."
      cp -a "$OVERLAY/." "$FLORA_DIR/"
      rm -rf "$OVERLAY"
    fi
  fi
  cd "$FLORA_DIR"
  echo ">> generating makefiles + building FLoRa..."
  make makefiles
  make -j"$JOBS" MODE=release
  echo ">> FLoRa build complete."
}

phase_smoke() {
  echo "==================== SMOKE TEST: run a FLoRa example (headless) ===================="
  safe_source "$OMNET_DIR/setenv" -f
  safe_source "$INET_DIR/setenv" -f
  cd "$FLORA_DIR/simulations"
  echo ">> available configs in omnetpp.ini:"
  opp_run -l "$FLORA_DIR/src/flora" \
          -n ".:$FLORA_DIR/src:$INET_DIR/src" \
          -u Cmdenv -a omnetpp.ini 2>/dev/null | head -30 || true
  echo ""
  echo ">> running first config for a short sim-time as a sanity check..."
  CFG="$(opp_run -l "$FLORA_DIR/src/flora" -n ".:$FLORA_DIR/src:$INET_DIR/src" -u Cmdenv -a omnetpp.ini 2>/dev/null | grep -oE 'Config [A-Za-z0-9_]+' | awk '{print $2}' | head -1)"
  if [ -n "${CFG:-}" ]; then
    echo ">> running config: $CFG"
    opp_run -l "$FLORA_DIR/src/flora" \
            -n ".:$FLORA_DIR/src:$INET_DIR/src" \
            -u Cmdenv -c "$CFG" --sim-time-limit=30s omnetpp.ini
    echo ">> SMOKE TEST PASSED — FLoRa runs."
  else
    echo "!! could not auto-detect a config name; inspect omnetpp.ini and run manually."
  fi
}

# ---- driver ----------------------------------------------------------------
TARGET="${1:-all}"
load_toolchain
case "$TARGET" in
  omnet) phase_omnet ;;
  inet)  phase_inet ;;
  flora) phase_flora ;;
  smoke) phase_smoke ;;
  all)   phase_omnet; phase_inet; phase_flora; phase_smoke ;;
  *) echo "unknown target: $TARGET  (use: all|omnet|inet|flora|smoke)"; exit 1 ;;
esac

echo ""
echo "============================================================"
echo " DONE: $TARGET"
echo " To use the stack in a NEW shell (e.g. inside a VNC/OnDemand desktop):"
echo "   source /etc/profile.d/lmod.sh"
echo "   module load $MOD_GCC $MOD_QT5"
echo "   source $OMNET_DIR/setenv -f"
echo "   source $INET_DIR/setenv -f"
echo " Then launch the GUI from flora/simulations:"
echo "   cd $FLORA_DIR/simulations && opp_run -l $FLORA_DIR/src/flora \\"
echo "       -n .:$FLORA_DIR/src:$INET_DIR/src -u Qtenv omnetpp.ini"
echo "============================================================"
