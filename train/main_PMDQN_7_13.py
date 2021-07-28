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

from train.algorithms.pmdqn import PMDQN
from train.env import Env
from train.utils.buffer import ReplayBuffer
import pickle
import threading

USE_CUDA = False
global replay_buffer
global epi_record

class myThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        print ("start save file, episode=", epi_record)
        with open('checkpoints_PMDQN_6_22/logs/incremental' +'dqnBuffer' + str(epi_record) + '.pkl', 'wb') as file:
            pickle.dump(replay_buffer, file)
        print ("finish save file：")

def run(config):
    log_dir = 'checkpoints_PMDQN_6_22_epi20/'
    run_dir = log_dir + 'logs/'
    logger = SummaryWriter(str(log_dir))

    env = Env()
    pmdqn = PMDQN.init_from_env(agent_alg=config.agent_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    global replay_buffer
    replay_buffer = ReplayBuffer(config.buffer_length, pmdqn.nagents,
                                 [362]*3,
                                 [4]*3)
    t = 0
    global epi_record
    for ep_i in range(0, config.n_episodes):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 2,
                                        config.n_episodes))
        epi_record = ep_i
        obs = env.reset()
        pmdqn.prep_rollouts(device='cpu')

        collision_flag = 0
        for et_i in range(config.episode_length):
            print(et_i)
            collision_flag = collision_flag + 1
            # env.render(close=False)
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(pmdqn.nagents)]
            # get actions as torch Variables
            agent_actions = pmdqn.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            # rearrange actions to be per environment
            actions = [[ac for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones = env.step(actions)
            print(rewards)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    pmdqn.prep_training(device='cpu')
                else:
                    pmdqn.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(pmdqn.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        pmdqn.update(sample, a_i, logger=logger)
                    pmdqn.update_all_targets()
                pmdqn.prep_rollouts(device='cpu')
            print(np.max(dones))
            if np.max(dones)==1:
                break
            time.sleep(0.5)

        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
        # if ep_i % config.save_interval < config.n_rollout_threads:
        #     os.makedirs(run_dir + 'incremental', exist_ok=True)
        #     dqn.save(run_dir + 'incremental' + ('model_ep%i.pt' % (ep_i + 1)))
        #     dqn.save(run_dir + 'model.pt')
        #     save_thread = myThread()
        #     save_thread.start()
        if collision_flag == config.episode_length:
            logger.add_scalar('collision', 0, ep_i)
        else:
            logger.add_scalar('collision',1, ep_i)


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
    parser.add_argument("--buffer_length", default=int(1e5), type=int)
    parser.add_argument("--n_episodes", default=20000, type=int)
    parser.add_argument("--episode_length", default=20, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=5000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
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