import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from train.utils.networks import MLPNetwork
from train.utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from train.utils.agents import DRONAgent
import numpy as np
import copy

MSELoss = torch.nn.MSELoss()

class DRON(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=32,
                 discrete_action=True):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DRONAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.niter = 0
        self.pol_dev='gpu'
        self.TARGET_UPDATE = 10
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        # return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
        #                                                          observations)]
        actionList=[]
        for agent_i, a, obs in zip(range(self.nagents),self.agents,observations):
            other_obs = copy.deepcopy(observations)
            other_obs.pop(agent_i)
            #other_obs=copy.deepcopy(observations).pop(agent_i)
            action=a.step(obs,other_obs)
            actionList.append(action)
        return actionList


    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]
        curr_obs=obs[agent_i]
        curr_acs=acs[agent_i]
        curr_rews=rews[agent_i]
        curr_next_obs=next_obs[agent_i]
        oppo_next_obs=[next_obs[i] for i in range(len(self.agents)) if i != agent_i]
        oppo_obs = [obs[i] for i in range(len(self.agents)) if i != agent_i]
        curr_agent.optimizer_1.zero_grad()
        curr_agent.optimizer_2.zero_grad()
        curr_agent.optimizer_op.zero_grad()

        #vf_in=torch.cat((*obs,),dim=1)
        curr_acs_index=curr_acs.max(1)[1].view(-1,1)
        #print(curr_acs_index)
        #calculate curr_q
        hidd_s = curr_agent.agent_encoder.forward(curr_obs)
        hidd_o = curr_agent.oppo_encoder.forward(torch.cat((curr_obs, *oppo_obs), dim=1))
        q_1 = curr_agent.q1.forward(hidd_s)
        q_2 = curr_agent.q2.forward(hidd_s)
        w_out = curr_agent.opps_q.forward(hidd_o)
        w_out = F.softmax(w_out,dim=1)
        q_mat=torch.cat((q_1.view(-1,1,4), q_2.view(-1,1,4)), dim=1)
        q_total = w_out.view(-1,1,2).matmul(q_mat).squeeze()
        actual_values=q_total.gather(1,curr_acs_index)

        # calculate next_q_values
        next_hidd_s = curr_agent.agent_encoder.forward(curr_next_obs)
        next_hidd_o = curr_agent.oppo_encoder.forward(torch.cat((curr_next_obs,*oppo_next_obs), dim=1))
        next_q_1 = curr_agent.target_q1.forward(next_hidd_s)
        next_q_2 = curr_agent.target_q2.forward(next_hidd_s)
        next_w_out = curr_agent.target_opps_q.forward(next_hidd_o)
        next_w_out =F.softmax(next_w_out,dim=1)
        next_q_mat=torch.cat((next_q_1.view(-1,1,4), next_q_2.view(-1,1,4)),dim=1)
        next_q_total = next_w_out.view(-1,1,2).matmul(next_q_mat).squeeze()
        next_values = next_q_total.max(1)[0].unsqueeze(1).detach()


        target_values=curr_rews.view(-1, 1)+self.gamma*(1 - dones[agent_i].view(-1, 1))*next_values
        loss = MSELoss(actual_values, target_values.detach())
        loss.backward()
        if parallel:
            average_gradients(curr_agent.q1)
            average_gradients(curr_agent.q2)
            average_gradients(curr_agent.opps_q)
        for param in curr_agent.q1.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
        for param in curr_agent.q2.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
        for param in curr_agent.opps_q.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
        curr_agent.optimizer_1.step()
        curr_agent.optimizer_2.step()
        curr_agent.optimizer_op.step()
        self.niter=self.niter+1
        curr_agent.EPSILON = curr_agent.EPSILON * curr_agent.EPS_DEC if curr_agent.EPSILON > \
                                                      curr_agent.EPS_MIN else curr_agent.EPS_MIN
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'loss': loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)

        """

        for a in self.agents:
            # soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_q1, a.q1, self.tau)
            soft_update(a.target_q2, a.q2, self.tau)
            soft_update(a.target_opps_q, a.opps_q, self.tau)
        # if self.niter % self.TARGET_UPDATE==0:
        #     print('Update Target')
        #     for a in self.agents:
        #         # soft_update(a.target_critic, a.critic, self.tau)
        #         soft_update(a.target_policy, a.policy, self.tau)
        #self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.agent_encoder.train()
            a.oppo_encoder.train()
            a.q1.train()
            a.q2.train()
            a.opps_q.train()
            a.target_q1.train()
            a.target_q2.train()
            a.target_opps_q.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.agent_encoder = fn(a.agent_encoder)
                a.oppo_encoder = fn(a.oppo_encoder)
                a.q1 = fn(a.q1)
                a.q2 = fn(a.q2)
                a.opps_q = fn(a.opps_q)
            self.pol_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_q1 = fn(a.target_q1)
                a.target_q2 = fn(a.target_q2)
                a.target_opps_q = fn(a.target_opps_q)
            self.trgt_pol_dev = device

    def prep_rollouts(self, device='gpu'):
        for a in self.agents:
            a.agent_encoder.eval()
            a.oppo_encoder.eval()
            a.q1.eval()
            a.q2.eval()
            a.opps_q.eval()
            a.target_q1.eval()
            a.target_q2.eval()
            a.target_opps_q.eval()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        for a in self.agents:
            a.agent_encoder = fn(a.agent_encoder)
            a.oppo_encoder=fn(a.oppo_encoder)
            a.q1=fn(a.q1)
            a.q2=fn(a.q2)
            a.opps_q=fn(a.opps_q)
            a.oppo_encoder=fn(a.oppo_encoder)
            a.target_q1=fn(a.target_q1)
            a.target_q2=fn(a.target_q2)
            a.target_opps_q=fn(a.target_opps_q)

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, agent_num=3, agent_alg="DRON", num_in_pol=364, num_out_pol=4, discrete_action=True,
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=32):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []

        for i in range(agent_num):
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': agent_alg,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location="cuda:0")
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance