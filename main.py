"""
This file is an example train and test loop for the different environments.
Selecting different environments is done through setting the 'env_name' variable.

TODO:
* add rgb_array rendering for the different environments to allow saving videos
"""
import argparse
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.env_checker import check_env

import numpy as np

import bluesky_gym
import bluesky_gym.envs

from bluesky_gym.utils import logger

bluesky_gym.register_envs()

# env_name = 'DescentEnvXYZ-v0'
algorithm = SAC

# TRAIN = True
EVAL_EPISODES = 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type = str)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model_path", type=str)

    args = parser.parse_args()

    env_name = args.env_name

    # Initialize logger
    log_dir = f'./logs/{env_name}/'
    file_name = f'{env_name}_{str(algorithm.__name__)}.csv'
    csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)

    env = gym.make(env_name, render_mode=None)

    # check_env(env)

    obs, info = env.reset()
    # model = algorithm("MultiInputPolicy", env, verbose=1,learning_rate=3e-4)
    if args.model_path:
        model = algorithm.load(args.model_path, env=env)
        # Save backup if loading from existing
        if args.train:
            model.save(f"{args.model_path}_backup")
    else:
        model = algorithm("MultiInputPolicy", env, verbose=1,learning_rate=3e-4)

    if args.train:
        ts = 5000
        print(f"Training for {ts} timesteps")
        model.learn(total_timesteps=ts, callback=csv_logger_callback)
        model.save(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}")
        del model
    env.close()
    
    # Test the trained model
    model = algorithm.load(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}", env=env)
    env = gym.make(env_name, render_mode="human")
    for i in range(EVAL_EPISODES):

        done = truncated = False
        obs, info = env.reset()
        tot_rew = 0
        while not (done or truncated):
            # action = np.array(np.random.randint(-100,100,size=(2))/100)
            # action = np.array([0,-1])
            action, _states = model.predict(obs, deterministic=True)
            # print(action)
            obs, reward, done, truncated, info = env.step(action[()])
            # print(obs)
            tot_rew += reward
        print(tot_rew)
    env.close()