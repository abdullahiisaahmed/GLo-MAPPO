import os
from functools import partial

import numpy as np

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY


class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](
            **self.args.env_args,
            common_reward=self.args.common_reward,
            reward_scalarisation=self.args.reward_scalarisation,
        )
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.train_global_energy_efficiency = []
        self.test_global_energy_efficiency = []
        self.train_total_system_datarate = []
        self.test_total_system_datarate = []
        self.train_total_ed_tp = []
        self.test_total_ed_tp = []
        self.train_uavs_reached_cs = []
        self.test_uavs_reached_cs = []
        self.train_unique_eds_served = []
        self.test_unique_eds_served = []
        self.train_uav_collisions = []
        self.test_uav_collisions = []
        self.train_uavs_propulsion_power = []
        self.test_uavs_propulsion_power = []
        # packet-level (ED) capture metrics
        self.train_pdr = []
        self.test_pdr = []
        self.train_pkt_collision_rate = []
        self.test_pkt_collision_rate = []
        self.train_coverage_rate = []
        self.test_coverage_rate = []

        # Separate train/test save counters + intervals (Option B), so test never
        # consumes a train save window and each cadence is independent.
        self.train_episode_count = 0
        self.test_episode_count = 0
        self._save_train_interval = getattr(args, 'save_train_interval', 20)
        self._save_test_interval = getattr(args, 'save_test_interval', 20)
        self._last_train_save_bucket = 0
        self._last_test_save_bucket = 0

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        if self.args.common_reward:
            episode_return = 0
        else:
            episode_return = np.zeros(self.args.n_agents)
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )

            reward, terminated, env_info = self.env.step(actions[0])
            if test_mode and self.args.render:
                self.env.render()
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if self.args.common_reward:
                post_transition_data["reward"] = [(reward,)]
            else:
                post_transition_data["reward"] = [tuple(reward)]

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        # Per-mode save cadence (Option B). Bucket-crossing makes the cadence exact
        # regardless of batch size; separate counters stop test stealing a train save.
        if test_mode:
            self.test_episode_count += 1
            cur_episode = self.test_episode_count
            _bucket = cur_episode // self._save_test_interval
            should_save = _bucket > self._last_test_save_bucket
            if should_save:
                self._last_test_save_bucket = _bucket
        else:
            self.train_episode_count += 1
            cur_episode = self.train_episode_count
            _bucket = cur_episode // self._save_train_interval
            should_save = _bucket > self._last_train_save_bucket
            if should_save:
                self._last_train_save_bucket = _bucket

        if should_save and getattr(self.args, 'save_episode_data', True):
            prefix = "test" if test_mode else "train"
            save_dir = os.path.join(
                self.args.local_results_path,
                "episode_data",
                self.args.unique_token,
                prefix,
                f"episode_{cur_episode}",
            )
            os.makedirs(save_dir, exist_ok=True)
            self.env.save_trajectories(save_dir)
            plot_path = os.path.join(save_dir, "trajectory_plot.png")
            self.env.render(save_path=plot_path)

        env_stats = self.env.get_stats()
        uav_collisions = np.sum(env_stats['uav_collisions'])   # UAV-UAV near-collisions
        data_rate = np.sum(env_stats['total_system_datarate'])
        energy_efficiency = np.mean(env_stats['global_energy_efficiency'])
        uavs_reached_cs = env_stats['uavs_reached_cs']
        unique_eds_served = env_stats['unique_eds_served']
        total_ed_tp = np.sum(env_stats['total_ed_tp'])
        uavs_propulsion_power = np.sum(env_stats['uavs_propulsion_power'])
        # packet-level (ED) capture metrics
        pdr = np.mean(env_stats['pdr'])
        pkt_collision_rate = np.mean(env_stats['pkt_collision_rate'])
        coverage_rate = np.mean(env_stats['coverage_rate'])

        if test_mode:
            self.test_global_energy_efficiency.append(energy_efficiency)
            self.test_total_system_datarate.append(data_rate)
            self.test_total_ed_tp.append(total_ed_tp)
            self.test_uavs_propulsion_power.append(uavs_propulsion_power)
            self.test_uav_collisions.append(uav_collisions)
            self.test_uavs_reached_cs.append(uavs_reached_cs)
            self.test_unique_eds_served.append(unique_eds_served)
            self.test_pdr.append(pdr)
            self.test_pkt_collision_rate.append(pkt_collision_rate)
            self.test_coverage_rate.append(coverage_rate)
        else:
            self.train_global_energy_efficiency.append(energy_efficiency)
            self.train_total_system_datarate.append(data_rate)
            self.train_total_ed_tp.append(total_ed_tp)
            self.train_uavs_propulsion_power.append(uavs_propulsion_power)
            self.train_uav_collisions.append(uav_collisions)
            self.train_uavs_reached_cs.append(uavs_reached_cs)
            self.train_unique_eds_served.append(unique_eds_served)
            self.train_pdr.append(pdr)
            self.train_pkt_collision_rate.append(pkt_collision_rate)
            self.train_coverage_rate.append(coverage_rate)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
            self._log_new_stats(test_mode)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            self._log_new_stats(test_mode)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        if self.args.common_reward:
            self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
            self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        else:
            for i in range(self.args.n_agents):
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_mean",
                    np.array(returns)[:, i].mean(),
                    self.t_env,
                )
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_std",
                    np.array(returns)[:, i].std(),
                    self.t_env,
                )
            total_returns = np.array(returns).sum(axis=-1)
            self.logger.log_stat(
                prefix + "total_return_mean", total_returns.mean(), self.t_env
            )
            self.logger.log_stat(
                prefix + "total_return_std", total_returns.std(), self.t_env
            )
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()

    def _log_new_stats(self, test_mode):
        prefix = "test_" if test_mode else ""

        stats = [
            'global_energy_efficiency',
            'total_system_datarate',
            'total_ed_tp',
            'uavs_propulsion_power',
            'uav_collisions',              # UAV-UAV near-collisions (total)
            'uavs_reached_cs',
            'unique_eds_served',
            'pdr',                         # packet delivery ratio (mean/std)
            'pkt_collision_rate',          # ED packet collision rate (mean/std)
            'coverage_rate',               # unique EDs served / total (mean/std)
        ]

        for stat in stats:
            cur_list = getattr(self, f"{'test' if test_mode else 'train'}_{stat}")

            if len(cur_list) == 0:
                continue

            if stat in ['uav_collisions', 'uavs_reached_cs']:
                self.logger.log_stat(f"{prefix}{stat}", sum(cur_list), self.t_env)
                cur_list.clear()
                continue

            if stat == 'uavs_propulsion_power':
                stat_values = [np.sum(x) for x in cur_list]
            else:
                stat_values = cur_list

            self.logger.log_stat(f"{prefix}{stat}_mean", np.mean(stat_values), self.t_env)
            self.logger.log_stat(f"{prefix}{stat}_std", np.std(stat_values), self.t_env)
            cur_list.clear()
