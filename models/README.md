# Pretrained GLo-MAPPO Policy

This directory ships the **trained actor network** used for the closed-loop co-simulation demo.

```
models/
└── glo_ed50_seed41/
    └── agent.th          # PyTorch state_dict of the shared actor (GRU policy)
```

## What this checkpoint is

| Field | Value |
|---|---|
| Algorithm | GLo-MAPPO (MAPPO backbone) |
| Run name | `glo_ed50` |
| Seed | 41 |
| Training length | ~1.98 M environment steps (`t_max = 2 000 000`) |
| Agents (UAV gateways) | 2 |
| End devices (LoRa nodes) | 50 |
| Area | 1000 × 1000 m, UAV altitude 150 m |
| Max speed / comm. range | 30 m/s / 300 m |
| Episode length | 500 steps |

