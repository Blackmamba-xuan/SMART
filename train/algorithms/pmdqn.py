import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from train.utils.networks import MLPNetwork
from train.utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from train.utils.pmagents import PMAgent
import numpy as np

MSELoss = torch.nn.MSELoss()
criterion = torch.nn.CrossEntropyLoss()

class PMDQN(object):
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
        self.agents = [PMAgent(lr=lr, discrete_action=discrete_action,
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
        for a, obs in zip(self.agents,observations):
            obs_x1=obs[:,:-2]
            obs_x2=obs[:,-2:]
            action=a.step(obs_x1,obs_x2)
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
        curr_obs_x1=curr_obs[:,:-2]
        curr_obs_x2=curr_obs[:,-2:]
        curr_acs=acs[agent_i]
        curr_rews=rews[agent_i]
        curr_next_obs=next_obs[agent_i]
        curr_next_obs_x1=curr_next_obs[:,:-2]
        curr_next_obs_x2=curr_next_obs[:,-2:]
        curr_agent.policy_optimizer.zero_grad()

        #vf_in=torch.cat((*obs,),dim=1)
        curr_acs_index=curr_acs.max(1)[1].view(-1,1)
        #print(curr_acs_index)
        # calculate the partner action prob
        action_prob1 = curr_agent.pmNet_1.forward(curr_obs_x1).detach()
        action_prob1_soft = torch.softmax(action_prob1, dim=1)
        action_prob2 = curr_agent.pmNet_2.forward(curr_obs_x1).detach()
        action_prob2_soft = torch.softmax(action_prob2, dim=1)
        partn_ac_prob = torch.cat((action_prob1_soft, action_prob2_soft), dim=1)
        actual_values=curr_agent.policy(curr_obs_x1,torch.cat((curr_obs_x2,partn_ac_prob),dim=1)).gather(1,curr_acs_index)

        # calculate the partner next action prob
        next_action_prob1 = curr_agent.pmNet_1.forward(curr_next_obs_x1).detach()
        next_action_prob1_soft = torch.softmax(next_action_prob1, dim=1)
        next_action_prob2 = curr_agent.pmNet_2.forward(curr_next_obs_x1).detach()
        next_action_prob2_soft = torch.softmax(next_action_prob2, dim=1)
        next_partn_ac_prob = torch.cat((next_action_prob1_soft, next_action_prob2_soft), dim=1)

        target_values=curr_rews.view(-1, 1)+self.gamma*(1 - dones[agent_i].view(-1, 1))*curr_agent.target_policy(curr_next_obs_x1,torch.cat((curr_next_obs_x2,next_partn_ac_prob),dim=1)).max(1)[0].unsqueeze(1).detach()
        loss = MSELoss(actual_values, target_values.detach())
        loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        for param in curr_agent.policy.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
        curr_agent.policy_optimizer.step()
        self.niter=self.niter+1
        curr_agent.EPSILON = curr_agent.EPSILON * curr_agent.EPS_DEC if curr_agent.EPSILON > \
                                                      curr_agent.EPS_MIN else curr_agent.EPS_MIN
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'loss': loss},
                               self.niter)
        #update the partner modeling
        curr_agent.pmNet_optimizer1.zero_grad()
        curr_agent.pmNet_optimizer2.zero_grad()
        other_acs=[acs[i] for i in range(self.nagents) if i!=agent_i]
        action_predic_out1=curr_agent.pmNet_1.forward(curr_obs_x1)
        _,action_label1=torch.max(other_acs[0],dim=1)
        partner_loss1=criterion(action_predic_out1, action_label1)
        action_predic_out2 = curr_agent.pmNet_2.forward(curr_obs_x1)
        _,action_label2 = torch.max(other_acs[1], dim=1)
        partner_loss2 = criterion(action_predic_out2, action_label2)
        total_loss=partner_loss1+partner_loss2
        total_loss.backward()
        for param in curr_agent.pmNet_1.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
        for param in curr_agent.pmNet_2.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
        curr_agent.pmNet_optimizer1.step()
        curr_agent.pmNet_optimizer2.step()
        if logger is not None:
            logger.add_scalars('agent%i/pmlosses1' % agent_i,
                               {'partner_loss1': partner_loss1},
                               self.niter)
            logger.add_scalars('agent%i/pmlosses2' % agent_i,
                               {'partner_loss2': partner_loss2},
                               self.niter)


    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)

        """

        for a in self.agents:
            # soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        # if self.niter % self.TARGET_UPDATE==0:
        #     print('Update Target')
        #     for a in self.agents:
        #         # soft_update(a.target_critic, a.critic, self.tau)
        #         soft_update(a.target_policy, a.policy, self.tau)
        #self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device

    def prep_rollouts(self, device='gpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        for a in self.agents:
            a.policy = fn(a.policy)
            a.target_policy=fn(a.target_policy)

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, agent_num=3, agent_alg="DQN", num_in_pol=360, attention_dim=(2+8), num_out_pol=4,num_in_pm=360,
                      discrete_action=True,gamma=0.95, tau=0.01, lr=0.01, hidden_dim=32):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []

        for i in range(agent_num):
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_pm':num_in_pm,
                                      'attention_dim':attention_dim
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