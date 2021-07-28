import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from train.utils.networks import MLPNetwork
from train.utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from train.utils.agents import SACAgent

MSELoss = torch.nn.MSELoss()

class COMA(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):
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
        self.alg_types = alg_types
        self.agents = [SACAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **params)
                       for params in agent_init_params]
        self.nagents = len(self.agents)
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.actor for a in self.agents]

    @property
    def target_policies(self):
        return [a.actor for a in self.agents]

    # step single agent
    def step(self,index,obs,explore=False):
        #print(index)
        return self.agents[index].step(obs)

    # step all agents
    def stepAll(self,observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs) for a, obs in zip(self.agents,observations)]

    def update(self, sample, agent_i,alg_type='multi-agent',parallel=False, logger=None):
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
        actions=[]

        # update value network
        if alg_type == 'multi-agent':
            state=torch.cat((*obs,),dim=1)
            for a_i, pi, ob in zip(range(self.nagents),self.target_policies,obs):
                action, log_pi=pi.sample_normal(ob,reparameterize=False)
                if self.discrete_action:
                    action=onehot_from_logits(action)
                if a_i==agent_i:
                    log_prob=log_pi
                actions.append(action)
            actions=torch.cat((*actions,),dim=1)
        else:
            state=obs[agent_i]
            action,log_prob=curr_agent.actor.sample_normal(obs[agent_i],reparameterize=False)
            if self.discrete_action:
                actions=onehot_from_logits(action)
        log_prob=log_prob.view(-1)
        q_1 = curr_agent.critic_1.forward(state,actions)
        q_2 = curr_agent.critic_2.forward(state,actions)
        critic_value=torch.min(q_1,q_2)
        critic_value=critic_value.view(-1)

        value=curr_agent.value(state).view(-1)
        value_=curr_agent.target_value(state).view(-1)

        curr_agent.value.optimizer.zero_grad()
        value_target=critic_value-log_prob
        value_loss=0.5*MSELoss(value,value_target)
        value_loss.backward(retain_graph=True)
        curr_agent.value.optimizer.step()

        #update policy network
        actions=[]
        if alg_type == 'multi-agent':
            for a_i, pi, ob in zip(range(self.nagents),self.target_policies,obs):
                action, log_pi=pi.sample_normal(ob,reparameterize=True)
                if self.discrete_action:
                    action=onehot_from_logits(action)
                if a_i==agent_i:
                    log_prob=log_pi
                actions.append(action)
            actions=torch.cat((*actions,),dim=1)
        else:
            action,log_prob=curr_agent.actor.sample_normal(obs[agent_i],reparameterize=False)
            if self.discrete_action:
                actions=onehot_from_logits(action)
        log_prob=log_prob.view(-1)
        q_1=curr_agent.critic_1.forward(state,actions)
        q_2=curr_agent.critic_2.forward(state,actions)
        critic_value=torch.min(q_1,q_2)
        critic_value=critic_value.view(-1)

        actor_loss=log_prob-critic_value
        actor_loss=torch.mean(actor_loss)
        curr_agent.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        curr_agent.actor.optimizer.step()

        #update critic network
        curr_agent.critic_1.optimizer.zero_grad()
        curr_agent.critic_2.optimizer.zero_grad()
        q_hat=rews[agent_i].view(-1, 1) + self.gamma * curr_agent.target_value(state)*(1 - dones[agent_i].view(-1, 1))
        q_hat=q_hat.view(-1)
        q_1_old=curr_agent.critic_1.forward(state,actions).view(-1)
        q_2_old=curr_agent.critic_2.forward(state,actions).view(-1)
        critic_1_loss=0.5*MSELoss(q_1_old,q_hat)
        critic_2_loss=0.5*MSELoss(q_2_old,q_hat)
        critic_loss=critic_1_loss+critic_2_loss
        critic_loss.backward()
        curr_agent.critic_1.optimizer.step()
        curr_agent.critic_2.optimizer.step()

        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': value_loss,
                                'critic_loss': critic_loss,
                                'actor_loss': actor_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_value, a.value, self.tau)

        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.actor.train()
            a.critic_1.train()
            a.critic_2.train()
            a.value.train()
            a.target_value.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.actor = fn(a.actor)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic_1 = fn(a.critic_1)
                a.critic_2 = fn(a.critic_2)
                a.value = fn(a.value)
            self.critic_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_value = fn(a.target_value)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        #print('enter prep_rollouts')
        for a in self.agents:
            a.actor.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.actor = fn(a.actor)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, agent_num=4, agent_alg="multi-agent",num_in_pol=0,num_out_pol=4,discrete_action=True,
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=32):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []

        for i in range(agent_num):
            num_in_critic = num_in_pol + num_out_pol
            if agent_alg == "multi-agent":
                num_in_critic=num_in_critic*agent_num
                num_in_value = num_in_pol*agent_num
            else:
                num_in_value=num_in_pol
            agent_init_params.append({'num_in_pol': num_in_pol,
                                        'num_out_pol': num_out_pol,
                                        'num_in_critic': num_in_critic,
                                      'num_in_value':num_in_value
                                      })
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
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance