from .multiagentenv import MultiAgentEnv
from .gymma import GymmaWrapper

from gym.envs import registry as gym_registry


def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)


REGISTRY = {}
REGISTRY["gymma"] = gymma_fn
