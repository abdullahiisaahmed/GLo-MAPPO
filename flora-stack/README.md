# `flora-stack/` — FLoRa / OMNeT++ Simulator for the Closed-Loop Co-Simulation

This directory is the **network simulator side** of GLo-MAPPO. The trained policy is validated in a packet-level LoRa simulator (FLoRa on OMNeT++) that talks to the Python policy over a socket (see [`../flora_run/`](../flora_run)).

> **Only the custom overlay is committed here** — the ~GB OMNeT++/INET/FLoRa toolchains are **not** in this repo. You fetch and build them once (script provided), and the custom files below are merged on top.

## Pinned versions

| Component | Version | Source |
|---|---|---|
| OMNeT++ (with Qtenv GUI) | **6.0.3** | https://github.com/omnetpp/omnetpp |
| INET | **4.4.1** | https://github.com/inet-framework/inet |
| FLoRa | **1.1.0** | https://github.com/florasim/flora |

## The custom overlay

```
flora-stack/
├── use_flora.sh                                  # source this to prepare a shell to run FLoRa
└── flora/
    ├── src/Marl/                                 # ★ custom module — the co-simulation bridge
    │   ├── MarlController.{cc,h,ned}             #   drives UAV trajectories + per-node SF/TP,
    │   │                                         #   opens the :5000 socket to the Python env,
    │   │                                         #   selects the channel tier (channelTier=1|2|3)
    │   └── ExternalMobility.{cc,h,ned}           #   applies Python-supplied UAV positions
    ├── src/LoRaPhy/LoRaPathLossOulu.{cc,h,ned}   # FLoRa's Oulu terrestrial path-loss model
    │                                             #   (Tier-3 channel); pinned here for an exact match
    └── simulations/
        ├── MarlLoRaNetwork.ned                   # the 2-gateway / 50-node network
        └── examples/
            ├── glo-2gw-50ed-a2g-tier1.ini        # Tier-1: analytic air-to-ground (A2G)
            ├── glo-2gw-50ed-a2g-tier2.ini        # Tier-2: A2G + log-normal shadowing (σ=8 dB)
            ├── glo-2gw-50ed-a2g-tier3.ini        # Tier-3: native Oulu terrestrial
            └── marl-2gw-50ed-a2g.ini             # base scenario the tier inis `include`
```

## Build it

From the **repo root**:

```bash
bash install_flora_stack.sh all
```

This downloads OMNeT++ 6.0.3, INET 4.4.1 and FLoRa 1.1.0 into this `flora-stack/` directory, **merges the custom overlay above on top of stock FLoRa (overlay wins)**, and builds everything (`MODE=release`, Qtenv GUI enabled). You can also run phases individually:

```bash
bash install_flora_stack.sh omnet     # OMNeT++ only
bash install_flora_stack.sh inet      # INET only (needs omnet)
bash install_flora_stack.sh flora     # FLoRa only (needs inet) — does the overlay merge
bash install_flora_stack.sh smoke     # tiny headless run to verify the build
```

## Build it (manual)

1. Install **OMNeT++ 6.0.3** into `flora-stack/omnetpp-6.0.3/` and build it (with Qtenv).
2. Install **INET 4.4.1** into `flora-stack/inet4.4/` and build it (`make makefiles && make`). Keep the name `inet4.4`. FLoRa's makefiles reference `../inet4.4`.
3. Download **FLoRa 1.1.0** into `flora-stack/flora/`, then **copy the committed overlay files above into it** (they sit at their correct relative paths), and build:
   ```bash
   source flora-stack/use_flora.sh
   cd flora-stack/flora && make makefiles && make -j4 MODE=release
   ```

## Running

Prepare a shell (strips conda, loads the toolchain, points `LD_LIBRARY_PATH` at the built libs):

```bash
source flora-stack/use_flora.sh
cd flora-stack/flora/simulations
./run -u Qtenv -f examples/glo-2gw-50ed-a2g-tier1.ini    # GUI (needs a display)
./run -u Cmdenv -f examples/glo-2gw-50ed-a2g-tier1.ini   # terminal
```

The full three-terminal closed-loop procedure (policy server + env bridge + FLoRa) and the three channel tiers are documented in the **"Closed-loop co-simulation"** section of the top-level [`../README.md`](../README.md).

