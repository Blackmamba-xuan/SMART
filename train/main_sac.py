#! /usr/bin/env python
import rospy
import argparse
import torch
import time
import os
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from train.algorithms.masac import MASAC
from train.env import Env
from train.utils.buffer import ReplayBuffer

USE_CUDA = torch.cuda.is_available()

def run(config):
    log_dir = 'checkpoints_SAC_10_26/'
    run_dir = log_dir + 'logs/'
    logger = SummaryWriter(str(log_dir))

    env = Env()
    maddpg = MASAC.init_from_env(agent_num=3, agent_alg=config.agent_alg,num_in_pol=364,num_out_pol=4,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [364]*3,
                                 [4]*3)
    t = 0
    for ep_i in range(0, config.n_episodes):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 2,
                                        config.n_episodes))
        obs = env.reset()
        maddpg.prep_rollouts(device='cpu')

        # explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        # maddpg.scale_noise(
        #     config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        # maddpg.reset_noise()

        for et_i in range(config.episode_length):
            print(et_i)
            # env.render(close=False)
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.stepAll(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones = env.step(actions)
            print(rewards)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
            print(np.max(dones))
            if np.max(dones)==1:
                break
            time.sleep(1.5)

        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment", default="police_and_thief_resurgence_new")
    parser.add_argument("--model_name",
                        help="Name of directory to store " +
                             "model/training contents", default="multi-agent")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=50000, type=int)
    parser.add_argument("--episode_length", default=30, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=32, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="multi-agent", type=str,
                        choices=['multi-agent', 'single-agent'])
    parser.add_argument("--adversary_alg",
                        default="multi-agent", type=str,
                        choices=['multi-agent', 'single-agent'])
    parser.add_argument("--discrete_action", default=True, type=bool)

    config = parser.parse_args()
    run(config)

    # env=Env()
    # obs=env.reset()
    # #print('end success')
    # while True:
    #     actions = [[0, 0, 0, 0] for i in range(3)]
    #     actions = np.array(actions)
    #     next_obs, rewards, dones = env.step(actions)
    #     print('step one')
