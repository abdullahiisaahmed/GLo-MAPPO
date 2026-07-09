import torch as th
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import numpy as np
import os
from os.path import dirname, abspath
from run import run
from utils.logging import get_logger
import gym_mdde1  # Assuming this is the correct import for your environment


import collections
from _collections_abc import Mapping

from copy import deepcopy
import sys
import yaml
import gym


# # Set the capture mode to tee and increase the timeout
# SETTINGS.CAPTURE_MODE = 'tee'
# SETTINGS.CAPTURE_MODE['tee']['stdout']['timeout'] = 600  # Increase to 600 seconds or appropriate value
# SETTINGS.CAPTURE_MODE['tee']['stderr']['timeout'] = 600  # Increase to 600 seconds or appropriate value

logger = get_logger()

SETTINGS["CAPTURE_MODE"] = (
    "fd"  # set to "no" if you want to see stdout/stderr in console
)
logger = get_logger()


ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
# Directory containing the checkpoint's agent.th. Defaults to the model shipped with the
# repo; point this at any results/models/<run>/<step>/ directory to test a different one.
checkpoint_path = os.path.join(dirname(dirname(abspath(__file__))), "models", "glo_ed50_seed41")

@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config["env_args"]["seed"] = config["seed"]

    # run the framework
    run(_run, config, _log)

def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

def load_and_test_model():
    # Load the configuration for the environment and algorithm
    config = {
        "config": "mappo",
        "env_config": "gymma",
        "env_args": {
            "key": "LoRaEnv-v0"
        },
        "checkpoint_path": checkpoint_path
    }

    ex.add_config(config)
    
    # Initialize the environment and model
    # env = gym_mdde1.make(config["env_args"]["key"])
    env = gym.make('LoRaEnv-v0')

    model = th.load(config["checkpoint_path"])

    test_max = 100  # Set your test maximum here
    acc = 0
    reward_t = 0

    for test in range(1, test_max + 1):
        obs = env.reset()
        print(f"Test {test}")
        
        n_steps = 100  # Set the number of steps per episode
        for step in range(n_steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            
            if done:
                print(f"Assignment completed, reward (EE) = {reward}")
                reward_t += reward
                if reward[0] != 0:
                    acc += 1
                else:
                    print("Incomplete")
                break

    print(f'EE using MAPPO1-based allocation per {test_max} tests: {reward_t / acc if acc != 0 else 0}')
    print(f'Accuracy of MAPPO1-based allocation per {test_max} tests: {100 * acc / test_max}%')

if __name__ == "__main__":
    load_and_test_model()
