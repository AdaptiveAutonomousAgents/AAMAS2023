import argparse
import gym
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from typing import List

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type = str, default = "Hopper-v3")
    parser.add_argument('--reward_learning', type = int, default = 1, help = "whether to implement reward learning")
    parser.add_argument('--display_steps', type = int, default = 4, help = "iterations to display return and loss in terminal")
    parser.add_argument('--horizon', type = int, default = 32, help = "time horizon of an episode for reward learning")
    parser.add_argument('--hidden_sizes', type = List[int], default = [64, 64], help = "width and depth of hidden layers in reward net")
    parser.add_argument('--lr', type = float, default = 6.4e-3, help = "learning rate of reward net")
    parser.add_argument('--buffer_size', type = int, default = 64, help = "number of episodes stored")
    parser.add_argument('--bs', type = int, default = 64, help = "number of episodes for updating the reward net")
    parser.add_argument('--n_policies', type = int, default = 8)
    parser.add_argument('--eval_episodes', type = int, default = 5)
    parser.add_argument('--policy', type = str, default = "MlpPolicy")
    parser.add_argument('--learning_rate', type = float, default = 3e-4)
    parser.add_argument('--n_steps', type = int, default = 2048)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--n_epochs', type = int, default = 10)
    parser.add_argument('--gamma', type = float, default = 0.99)
    parser.add_argument('--gae_lambda', type = float, default = 0.95)
    parser.add_argument('--clip_range', type = float, default = 0.2)
    parser.add_argument('--normalize_advantage', type = bool, default = True)
    parser.add_argument('--ent_coef', type = float, default = 0.0)
    parser.add_argument('--vf_coef', type = float, default = 0.5)
    parser.add_argument('--max_grad_norm', type = float, default = 0.5)
    parser.add_argument('--policy_kwargs', type = dict, default = None)
    parser.add_argument('--total_timesteps', type = int, default = 5e5)
    parser.add_argument('--data_path', type = str, default = "evaluation.npz")

    args = parser.parse_args()
    print(args)

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)

    if args.reward_learning == 1:
        if_reward_learning = True
    else:
        if_reward_learning = False
    
    model = PPO(
        policy = args.policy,
        env = env,
        test_env = test_env,
        reward_learning = if_reward_learning,
        display_steps = args.display_steps,
        horizon = args.horizon,
        hidden_sizes = args.hidden_sizes,
        lr = args.lr,
        buffer_size = args.buffer_size,
        bs = args.bs,
        n_policies = args.n_policies,
        eval_episodes = args.eval_episodes,
        learning_rate = args.learning_rate,
        n_steps = args.n_steps,
        batch_size = args.batch_size,
        n_epochs = args.n_epochs,
        gamma = args.gamma,
        gae_lambda = args.gae_lambda,
        clip_range = args.clip_range,
        normalize_advantage = args.normalize_advantage,
        ent_coef = args.ent_coef,
        vf_coef = args.vf_coef,
        max_grad_norm = args.max_grad_norm,
        policy_kwargs = args.policy_kwargs
    )

    model.learn(total_timesteps = args.total_timesteps)
    
    np.savez(args.data_path, timesteps = model.timesteps, returns = model.returns)
