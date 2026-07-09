from collections.abc import Iterable
import importlib
import warnings

import gym
from gym.spaces import flatdim, MultiDiscrete
import numpy as np
from .multiagentenv import MultiAgentEnv

from .wrappers import TimeLimit, FlattenObservation
import pretrained

from gym.envs import registry as gym_registry

# from .registry import REGISTRY
# from .multi_lora_env import MultiLoRaEnv

class GymmaWrapper(MultiAgentEnv):
    def __init__(
        self,
        key,
        time_limit,
        pretrained_wrapper,
        seed,
        common_reward,
        reward_scalarisation,
        **kwargs,
    ):
        # gym 0.26+ intercepts max_episode_steps and never forwards it to the env
        # constructor. We need to pass it directly, so instantiate via entry_point.
        _ep_steps = kwargs.pop('max_episode_steps', time_limit)
        # registry.get() in gym 0.26, registry.spec() in newer gym
        _get_spec = getattr(gym_registry, 'get', None) or getattr(gym_registry, 'spec', None)
        _spec = _get_spec(key) if _get_spec is not None else None
        if _spec is not None and _ep_steps is not None:
            _mod, _cls = _spec.entry_point.split(':')
            _env_cls = getattr(importlib.import_module(_mod), _cls)
            self.original_env = _env_cls(max_episode_steps=_ep_steps, **kwargs)
        else:
            self.original_env = gym.make(f"{key}", **kwargs)
        self.episode_limit = time_limit
        self._env = TimeLimit(self.original_env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        self.n_agents = self._env.n_agents
        self._obs = None
        self._info = None

        _space0 = self._env.action_space[0]
        self._is_multidiscrete = isinstance(_space0, MultiDiscrete)
        if self._is_multidiscrete:
            self.action_nvec = list(_space0.nvec)
            self._n_actions_per_agent = sum(self.action_nvec)
            self.longest_action_space = _space0
        else:
            self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
            self._n_actions_per_agent = self.longest_action_space.n

        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = seed
        try:
            self._env.seed(self._seed)
        except AttributeError:
            pass

        self.common_reward = common_reward
        if self.common_reward:
            if reward_scalarisation == "sum":
                self.reward_agg_fn = lambda rewards: sum(rewards)
            elif reward_scalarisation == "mean":
                self.reward_agg_fn = lambda rewards: sum(rewards) / len(rewards)
            else:
                raise ValueError(
                    f"Invalid reward_scalarisation: {reward_scalarisation} (only support 'sum' or 'mean')"
                )

    def step(self, actions):
        """Returns reward, terminated, info"""
        if self._is_multidiscrete:
            # actions may be CUDA tensors (per agent: a 4-vector); move to host first
            actions = [
                a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
                for a in actions
            ]
        else:
            actions = [int(a) for a in actions]
        self._obs, reward, done, self._info = self._env.step(actions)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        if self.common_reward and isinstance(reward, Iterable):
            reward = float(self.reward_agg_fn(reward))
        elif not self.common_reward and not isinstance(reward, Iterable):
            warnings.warn(
                "common_reward is False but received scalar reward from the environment, returning reward as is"
            )

        if isinstance(done, Iterable):
            done = all(done)
        return reward, done, {}

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self._obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        raise self._obs[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return flatdim(self.longest_observation_space)

    # def get_state(self):
    #     return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state(self):
        if hasattr(self.original_env, "get_state"):
            return self.original_env.get_state()
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """Returns the shape of the state"""
        if hasattr(self.original_env, "state_size"):
            return self.original_env.state_size
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        if self._is_multidiscrete:
            # Delegate to the env when it provides a real per-head mask (e.g. per-ED
            # SF/TP slots where padded slots expose only the null action); otherwise
            # default to all-available.
            if hasattr(self.original_env, "get_avail_agent_actions"):
                return self.original_env.get_avail_agent_actions(agent_id)
            return self._n_actions_per_agent * [1]
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """Returns the total number of actions (logit width) an agent can take."""
        return self._n_actions_per_agent

    def get_env_info(self):
        info = super().get_env_info()
        if self._is_multidiscrete:
            info["action_nvec"] = self.action_nvec
        return info

    def reset(self):
        """Returns initial observations and states"""
        # print("Calling from inside gymma reset")
        # # self._obs, self._info = self._env.reset()

        self._obs = self._env.reset()

        # print("B"*20)
        # for ob in self._obs:
        #     print(type(ob))
        # # self._obs = self._obs[0]
        # print(self._obs[1].shape)
        # o = self._obs[1]
        # self._obs = [[o, o]]
        # print("D"*20)
        # for ob in self._obs:
        #     print(ob.shape)
        # print("Inside gymma reset")
        # print(f"{len(self._obs)=}")
        # print(f"First element: {type(self._obs[0])}")
        # print(f"Second element: {type(self._obs[1])}")

        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        # print("V"*20)
        # for ob in self._obs:
        #     print(ob.shape)
        
        return self.get_obs(), self.get_state()

    def render(self, save_path=None):
        if save_path is not None and hasattr(self.original_env, 'render'):
            self.original_env.render(save_path=save_path)
        else:
            self._env.render()

    def save_trajectories(self, save_dir):
        if hasattr(self.original_env, 'save_trajectories'):
            self.original_env.save_trajectories(save_dir)

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    # def get_stats(self):
    #     return {}

    def get_stats(self):
        return self._env.get_stats()
