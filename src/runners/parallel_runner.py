import datetime
import os
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
import math

class ParallelRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]

        for i in range(self.batch_size):
            env_args[i]["seed"] += i
            env_args[i]["common_reward"] = self.args.common_reward
            env_args[i]["reward_scalarisation"] = self.args.reward_scalarisation

        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg)))) 
                   for env_arg, worker_conn in zip(env_args, self.worker_conns)]
        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]
        self.t = 0
        self.t_env = 0

        # Initialize lists for new stats
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        
        # New stats tracking

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

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))
        pre_transition_data = {"state": [], "avail_actions": [], "obs": []}
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
        self.batch.update(pre_transition_data, ts=0)
        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()
        all_terminated = False

        if self.args.common_reward:
            episode_returns = [0.0 for _ in range(self.batch_size)]
        else:
            episode_returns = [
                np.zeros(self.args.n_agents) for _ in range(self.batch_size)
            ]

        # episode_returns = [0 for _ in range(self.batch_size)]

        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []

        while True:
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()
            actions_chosen = {"actions": actions.unsqueeze(1)}
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated and not terminated[idx]:
                    parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1

            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            if all(terminated):
                break

            post_transition_data = {"reward": [], "terminated": []}
            pre_transition_data = {"state": [], "avail_actions": [], "obs": []}

            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    post_transition_data["reward"].append((data["reward"],))
                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1
                    env_terminated = data["terminated"]
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            self.batch.update(
                post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            
            self.t += 1
            self.batch.update(
                pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # for parent_conn in self.parent_conns:
        #     parent_conn.send(("get_stats", None))
        # env_stats = [parent_conn.recv() for parent_conn in self.parent_conns]

        cur_returns = self.test_returns if test_mode else self.train_returns
        cur_returns.extend(episode_returns)

        # Per-mode save cadence (Option B). Counters advance by batch_size; bucket-
        # crossing makes the cadence ~exact regardless of batch size, and separate
        # counters stop test stealing a train save window.
        if test_mode:
            self.test_episode_count += self.batch_size
            cur_episode = self.test_episode_count
            _bucket = cur_episode // self._save_test_interval
            should_save = _bucket > self._last_test_save_bucket
            if should_save:
                self._last_test_save_bucket = _bucket
        else:
            self.train_episode_count += self.batch_size
            cur_episode = self.train_episode_count
            _bucket = cur_episode // self._save_train_interval
            should_save = _bucket > self._last_train_save_bucket
            if should_save:
                self._last_train_save_bucket = _bucket

        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))
        env_stats = [parent_conn.recv() for parent_conn in self.parent_conns]

        # save trajectory data and render plot (worker 0 only, to avoid batch_size duplicates)
        if should_save and getattr(self.args, 'save_episode_data', True):
            prefix = "test" if test_mode else "train"
            save_dir = os.path.join(
                self.args.local_results_path,
                "episode_data",
                self.args.unique_token,
                prefix,
                f"episode_{cur_episode}",
            )
            self.parent_conns[0].send(("save_episode_data", save_dir))
            self.parent_conns[0].recv()  # wait for completion

        # Process new stats
        # In the stats processing loop:
        for env_stat in env_stats:
            # Existing metrics
            uav_collisions = np.sum(env_stat['uav_collisions'])   # UAV-UAV near-collisions
            data_rate = np.sum(env_stat['total_system_datarate'])  # Changed from data_rate
            energy_efficiency = np.mean(env_stat['global_energy_efficiency'])
            uavs_reached_cs = env_stat['uavs_reached_cs']
            unique_eds_served = env_stat['unique_eds_served']

            # New metrics
            total_ed_tp = np.sum(env_stat['total_ed_tp'])
            uavs_propulsion_power = np.sum(env_stat['uavs_propulsion_power'])
            # packet-level (ED) capture metrics
            pdr = np.mean(env_stat['pdr'])
            pkt_collision_rate = np.mean(env_stat['pkt_collision_rate'])
            coverage_rate = np.mean(env_stat['coverage_rate'])

            if test_mode:
                # Update test lists
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
                # Update train lists
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


        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
            self._log_new_stats(test_mode)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            self._log_new_stats(test_mode)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
        return self.batch


    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean",
                             np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std",
                             np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean",
                                     v/stats["n_episodes"], self.t_env)
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
            'coverage_rate'                # unique EDs served / total (mean/std)
        ]
        
        for stat in stats:
            train_list = getattr(self, f"train_{stat}")
            test_list = getattr(self, f"test_{stat}")
            cur_list = test_list if test_mode else train_list

            if len(cur_list) == 0:
                continue

            # Handle special cases where we want total counts instead of mean/std
            if stat in ['uav_collisions', 'uavs_reached_cs']: #, 'unique_eds_served'
                total_value = sum(cur_list)
                self.logger.log_stat(f"{prefix}{stat}", total_value, self.t_env)
                cur_list.clear()
                continue  # Skip further processing for these stats

            # Existing handling for other stats
            if stat == 'uavs_propulsion_power':
                # Sum across all UAVs for each episode
                stat_values = [np.sum(x) for x in cur_list]
            else:
                stat_values = cur_list

            mean_val = np.mean(stat_values)
            std_val = np.std(stat_values)
            self.logger.log_stat(f"{prefix}{stat}_mean", mean_val, self.t_env)
            self.logger.log_stat(f"{prefix}{stat}_std", std_val, self.t_env)
            cur_list.clear()


def env_worker(remote, env_fn):
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            reward, terminated, env_info = env.step(actions)
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                "state": state, "avail_actions": avail_actions, "obs": obs,
                "reward": reward, "terminated": terminated, "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({"state": env.get_state(), "avail_actions": env.get_avail_actions(), "obs": env.get_obs()})
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "save_episode_data":
            save_dir = data
            os.makedirs(save_dir, exist_ok=True)
            env.save_trajectories(save_dir)
            plot_path = os.path.join(save_dir, "trajectory_plot.png")
            env.render(save_path=plot_path)
            remote.send(True)
        else:
            raise NotImplementedError

class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


