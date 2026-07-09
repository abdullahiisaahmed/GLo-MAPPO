# GLo-MAPPO: Multi-Agent Deep Reinforcement Learning for Energy-Efficient UAV-Assisted LoRa Networks

Reference implementation for the paper:

> **GLo-MAPPO: Multi-Agent Deep Reinforcement Learning for Energy-Efficient UAV-Assisted LoRa Networks.**

GLo-MAPPO trains a team of UAV gateways to (i) fly energy-efficient trajectories, (ii) associate ground LoRa end devices, and (iii) set each served device's **spreading factor (SF)** and **transmission power (TP)**, maximizing network energy efficiency. The policy is trained with **MAPPO** (multi-agent PPO) and validated in closed loop against a packet-level **FLoRa / OMNeT++** LoRa simulator.

---

## Repository layout

```
GLo_MAPPO/
├── src/                     # MAPPO / EPyMARL training + evaluation code
│   ├── main.py              #   training entry point (Sacred)
│   ├── config/              #   algorithm + env YAML configs (mappo.yaml, envs/gymma.yaml, ...)
│   └── envs/MADE/gym_mdde1/ #   the custom LoRa gym environment (MultiFlyingLoRaEnv)
├── models/                  # pretrained GLo-MAPPO policy (agent.th)  — see models/README.md
├── flora_run/               # closed-loop co-simulation bridge (policy server + env socket)
│   ├── policy_server.py     #   Terminal A: serves the trained policy on :6000
│   ├── env_socket.py        #   Terminal B: env sidecar; injects FLoRa's channel, relays obs<->action
│   ├── run_python_eval_logged.py    # standalone eval (no FLoRa)
│   └── report_cosim_comparison.py   # builds the standalone-vs-tier validation table
├── flora-stack/             # FLoRa/OMNeT++ simulator overlay + build — see flora-stack/README.md
├── assets/                  # figures for this README (drop your screenshot here)
├── requirements.txt         # pinned Python dependencies
├── install_flora_stack.sh   # one-shot fetch + build of OMNeT++/INET/FLoRa
├── LICENSE  /  NOTICE       # Apache-2.0 (this code derives from EPyMARL/PyMARL)
└── README.md                # you are here
```




## Setup

Python 3.11 is highly recommended as it was the version used for this project.

**Option A — `venv` (works anywhere):**
```bash
cd GLo_MAPPO
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Option B — `conda`:**
```bash
conda create -n glo_mappo python=3.11 -y
conda activate glo_mappo
pip install -r requirements.txt
```

Then install the custom LoRa gym environment (editable, so `import gym_mdde1` resolves):
```bash
pip install -e src/envs/MADE
```

Quick check:
```bash
python -c "import torch, gym, gym_mdde1; print('env OK')"
```

> The FLoRa/OMNeT++ simulator is **only needed for the closed-loop co-simulation inference**.

---

## Training

```bash
cd src
python main.py --config=mappo --env-config=gymma with \
    name="glo_ed50" \
    seed=41 \
    runner="parallel" \
    use_rnn=True \
    buffer_cpu_only=False \
    batch_size_run=20 \
    lr=0.003 \
    lr_decay=True \
    lr_decay_min_frac=0.1 \
    gamma=0.99 \
    eps_clip=0.2 \
    epochs=4 \
    entropy_coef=0.001 \
    t_max=2000000 \
    save_model_interval=25000 \
    save_model=True \
    env_args.key="LoRaEnv-v1" \
    env_args.time_limit=500 \
    env_args.max_episode_steps=500 \
    env_args.num_uavs=2 \
    env_args.num_eds=50 \
    env_args.area_size="[1000, 1000]" \
    env_args.uav_altitude=150 \
    env_args.max_speed=30 \
    env_args.comm_range=300 \
    env_args.safe_distance=3
```

- `--config=mappo` selects the MAPPO algorithm ([`src/config/algs/mappo.yaml`](src/config/algs/mappo.yaml));
  `--env-config=gymma` the environment wrapper ([`src/config/envs/gymma.yaml`](src/config/envs/gymma.yaml)).
- Training runs to `t_max = 2 000 000` steps; the model is saved under
  `src/results/models/<name>-...` and can then be pointed to by `flora_run/` for the closed loop.
- The environment id `LoRaEnv-v1` is registered by `import gym_mdde1` (done in `main.py`); variant ids for the ablation studies (`...-v3/-v4`) are registered alongside it — see [`src/envs/MADE/gym_mdde1/__init__.py`](src/envs/MADE/gym_mdde1/__init__.py) and the ablation knobs (`ablate`, `assoc_mode`, `opt_mode`) in [`src/config/envs/gymma.yaml`](src/config/envs/gymma.yaml).

---

## Closed-loop co-simulation (FLoRa / OMNeT++)

The trained GLo-MAPPO policy drives **two UAV gateways** live inside a packet-level LoRa simulator. Every 0.5 s, FLoRa measures the channel, the Python policy responds with UAV moves + per-node SF/TP, and FLoRa applies them — over three channel tiers (**Tier-1** analytic air-to-ground, **Tier-2** A2G + log-normal shadowing, **Tier-3** native Oulu terrestrial).

![GLo-MAPPO closed-loop environment: two UAV gateways serving 50 LoRa end devices in FLoRa/OMNeT++](assets/closed_loop_env.png)

*Two UAV gateways (from the bottom-left) serve ground LoRa nodes and re-associate as they fly toward the charging station. Green links mark nodes actively served each step.*

### ▶ Demo video

**Watch the closed-loop demo on YouTube: https://youtu.be/5H73WmQZgK4**

### Run it

**Step 1 — build the simulator (once).** Fetch + build OMNeT++ 6.0.3 / INET 4.4.1 / FLoRa 1.1.0
and merge the custom overlay (details in **[`flora-stack/README.md`](flora-stack/README.md)**):

```bash
bash install_flora_stack.sh all
```

**Step 2 — run the three-terminal closed loop.** Use three shells **on the same host**, started in order **A → B → C**. The `--tier N` on B must match the `...-tierN.ini` on C (`N = 1` A2G, `2` A2G+shadowing, `3` Oulu). Ports: policy `6000`, env bridge `5000`.

```bash
# Terminal A — policy server (your venv/conda env). Leave running for all tiers.
python flora_run/policy_server.py                 # waits for: [policy] READY on :6000

# Terminal B — env bridge (same env), matching the tier you will run in C:
python flora_run/env_socket.py --tier 1 --logdir flora_run/timing_logs

# Terminal C — FLoRa (NO conda/venv active; source the toolchain instead):
source flora-stack/use_flora.sh
cd flora-stack/flora/simulations
./run -u Qtenv  -f examples/glo-2gw-50ed-a2g-tier1.ini     # GUI (needs a display)
# ./run -u Cmdenv -f examples/glo-2gw-50ed-a2g-tier1.ini   # terminal
```

If nothing moves in the GUI, Terminals A and B aren't up yet (the sim blocks at step 1). Terminal B prints `[socket] ep=… t=… served=…` each control step — that's your "it's live".

**Step 3 (optional) — validation table.** Log a standalone reference and each tier, then compare:

```bash
python flora_run/run_python_eval_logged.py --episodes 10 --logdir flora_run/timing_logs   # standalone (no FLoRa)
# ...then run each tier headless in Terminal C:  ./run -u Cmdenv -r 0..9 -f examples/glo-2gw-50ed-a2g-tierN.ini
python flora_run/report_cosim_comparison.py --logdir flora_run/timing_logs --out cosim_comparison
```

The policy loaded in the closed loop is the same shipped `models/glo_ed50_seed41/agent.th`.

---

## Citation

If you use this code, please cite:

```bibtex
Coming soon...!!!
```

## Acknowledgements

This project builds upon and integrates the following open-source frameworks and simulators:

* **EPyMARL / PyMARL:** The underlying Python/MARL code implementation. [https://github.com/uoe-agents/epymarl](https://github.com/uoe-agents/epymarl)
* **Gym:** Used for environment wrappers and reinforcement learning tracking. [https://gymnasium.farama.org/index.html](https://gymnasium.farama.org/index.html)
* **OMNeT++:** The core component of the network simulation environment. [https://github.com/omnetpp/omnetpp](https://github.com/omnetpp/omnetpp)
* **INET Framework:** Provides the foundational network models and protocol stacks. [https://github.com/inet-framework/inet](https://github.com/inet-framework/inet)
* **FLoRa:** The LoRa network simulator extended by the `flora-stack/` simulator overlay. [https://github.com/florasim/flora](https://github.com/florasim/flora)