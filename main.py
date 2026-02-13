"""
Training and evaluation script for BlueSky Gym environments.

Usage:
  Train:    python main.py --train --timesteps 50000
  Eval:     python main.py --eval --model_path models/DescentEnvCR3D-v0/best_model
  Both:     python main.py --train --eval --timesteps 50000
  Resume:   python main.py --train --model_path models/DescentEnvCR3D-v0/best_model --timesteps 25000
"""
import argparse
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

import numpy as np

import bluesky_gym
import bluesky_gym.envs
from bluesky_gym.utils import logger

bluesky_gym.register_envs()

ALGORITHMS = {
    "SAC": SAC,
    "PPO": PPO,
    "TD3": TD3,
    "DDPG": DDPG,
}

SCEN_ENVS = ["DescentEnvCR3D-v0"]


def train(env_name, algorithm, scenario_path, timesteps, lr, model_path=None, checkpoint_freq=10000):
    log_dir = f"./logs/{env_name}/"
    file_name = f"{env_name}_{algorithm.__name__}.csv"
    csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)

    checkpoint_dir = f"models/{env_name}/checkpoints/"
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix=f"{env_name}_{algorithm.__name__}",
    )

    callbacks = CallbackList([csv_logger_callback, checkpoint_callback])

    if env_name not in SCEN_ENVS:
        env = gym.make(env_name, render_mode= None)
    else: 
        env = gym.make(env_name, render_mode=None, scenario_path=scenario_path)
    
    env.reset()

    if model_path:
        model = algorithm.load(model_path, env=env)
        model.save(f"{model_path}_backup")
        print(f"Resumed from {model_path} (backup saved)")
    else:
        model = algorithm("MultiInputPolicy", env, verbose=1, learning_rate=lr)

    print(f"Training {algorithm.__name__} on {env_name} for {timesteps} timesteps")
    model.learn(total_timesteps=timesteps, callback=callbacks)

    save_path = f"models/{env_name}/{env_name}_{algorithm.__name__}"
    model.save(save_path)
    print(f"Model saved to {save_path}")

    del model
    env.close()
    return save_path


def evaluate(env_name, algorithm, model_path, scenario_path, episodes=10):
    if env_name not in SCEN_ENVS:
        env = gym.make(env_name, render_mode= "human")
    else: 
        env = gym.make(env_name, render_mode="human", scenario_path=scenario_path)
    
    model = algorithm.load(model_path, env=env)

    print(f"Evaluating {model_path} for {episodes} episodes")
    for i in range(episodes):
        done = truncated = False
        obs, info = env.reset()
        tot_rew = 0
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action[()])
            tot_rew += reward
        print(f"  Episode {i+1}: reward = {tot_rew:.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate BlueSky Gym environments")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--env", type=str, default="DescentEnvCR3D-v0", help="Environment name")
    parser.add_argument("--algorithm", type=str, default="SAC", choices=ALGORITHMS.keys(), help="RL algorithm")
    parser.add_argument("--timesteps", type=int, default=30000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model (for resume or eval)")
    parser.add_argument("--scenario_path", type=str, default="scenarios_kord", help="Scenario file or directory")
    parser.add_argument("--eval_scenario", type=str, default="scenarios_kord/scenario_001.scn", help="Scenario for evaluation")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--checkpoint_freq", type=int, default=10000, help="Save checkpoint every N timesteps")

    args = parser.parse_args()
    algorithm = ALGORITHMS[args.algorithm]

    model_save_path = args.model_path

    if args.train:
        model_save_path = train(
            env_name=args.env,
            algorithm=algorithm,
            scenario_path=args.scenario_path,
            timesteps=args.timesteps,
            lr=args.lr,
            model_path=args.model_path,
            checkpoint_freq=args.checkpoint_freq,
        )

    if args.eval:
        eval_model = args.model_path or model_save_path
        if eval_model is None:
            eval_model = f"models/{args.env}/{args.env}_{args.algorithm}"
        evaluate(
            env_name=args.env,
            algorithm=algorithm,
            model_path=eval_model,
            scenario_path=args.eval_scenario,
            episodes=args.eval_episodes,
        )
