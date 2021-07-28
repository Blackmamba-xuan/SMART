#! /usr/bin/env python
#import rospy
import argparse
import torch
import time
import os
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from train.algorithms.dqn import DQN
from train.Smarlp.Env import Env
from train.utils.buffer import ReplayBuffer

USE_CUDA = torch.cuda.is_available()

def run(config):

    load_dir = 'Evaluation/dqn.pt'

    #env = Env()
    dqn = DQN.init_from_save(load_dir)
    #
    # t = 0
    # for ep_i in range(0, config.n_episodes):
    #     obs = env.reset()
    #     dqn.prep_rollouts(device='gpu')
    #
    #     collision_flag=0
    #     rws=[[],[],[]]
    #
    #     for et_i in range(config.episode_length):
    #         print(et_i)
    #         collision_flag=collision_flag+1
    #         # env.render(close=False)
    #         # rearrange observations to be per agent, and convert to torch Variable
    #         torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])).cuda(),
    #                               requires_grad=False)
    #                      for i in range(dqn.nagents)]
    #         # get actions as torch Variables
    #         agent_actions = dqn.step(torch_obs, explore=True)
    #         # convert actions to numpy arrays
    #         # rearrange actions to be per environment
    #         actions = [[ac for ac in agent_actions] for i in range(config.n_rollout_threads)]
    #         next_obs, rewards, dones = env.step(actions)
    #         rws[0].append(rewards[0][0])
    #         rws[1].append(rewards[0][1])
    #         rws[2].append(rewards[0][2])
    #         print(rewards)
    #         obs = next_obs
    #         t += config.n_rollout_threads
    #         print(np.max(dones))
    #         if np.max(dones)==1:
    #             break
    #         time.sleep(0.8)
    #     print(np.mean(rws[0]))
    #     print(np.mean(rws[1]))
    #     print(np.mean(rws[2]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment", default="Autodriving")
    parser.add_argument("--model_name",
                        help="Name of directory to store " +
                             "model/training contents", default="DQN")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--episode_length", default=30, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=15000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=500, type=int)
    parser.add_argument("--hidden_dim", default=32, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="DQN", type=str,
                        choices=['DQN', 'DQN'])
    parser.add_argument("--adversary_alg",
                        default="DQN", type=str,
                        choices=['DQN', 'DQN'])
    parser.add_argument("--discrete_action", default=True, type=bool)

    config = parser.parse_args()
    run(config)