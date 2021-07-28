import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from train.utils.networks import MLPNetwork
from train.utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from train.utils.agentsPlus import DronConAgent
import numpy as np
import copy

MSELoss = torch.nn.MSELoss()
CrossEntropyLoss=torch.nn.CrossEntropyLoss()

class DronCon(object):
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
        self.agents = [DronConAgent(lr=lr, discrete_action=discrete_action,
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
        return [a.q for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_q for a in self.agents]

    def step(self, observations, oppo_obs, explore=False):
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
        for agent_i, a, obs, oppo_ob in zip(range(self.nagents),self.agents,observations, oppo_obs):
            action=a.step(obs,oppo_ob)
            actionList.append(action)
        return actionList


    def update(self, sample, agent_i, parallel=False, logger=None):
        obs, oppo_obs,acs, rews, next_obs,next_oppo_obs, dones = sample
        curr_agent = self.agents[agent_i]
        curr_obs=obs[agent_i]
        curr_oppo_obs=oppo_obs[agent_i]
        curr_acs=acs[agent_i]
        curr_rews=rews[agent_i]
        curr_next_obs=next_obs[agent_i]
        curr_next_oppo_obs=next_oppo_obs[agent_i]
        curr_agent.optimizer.zero_grad()

        curr_acs_index=curr_acs.max(1)[1].view(-1,1)
        hidd_s = curr_agent.agent_encoder.forward(curr_obs)
        hidd_o_1 = curr_agent.oppo_encoder1.forward(curr_oppo_obs)
        hidd_o_2 = curr_agent.oppo_encoder2.forward(curr_oppo_obs)
        q_in = torch.cat((hidd_s, hidd_o_1), dim=1)
        q_in = torch.cat((q_in, hidd_o_2), dim=1)
        q_values = curr_agent.q.forward(q_in)
        actual_values=q_values.gather(1,curr_acs_index)

        # calculate next_q_values
        next_hidd_s = curr_agent.agent_encoder.forward(curr_next_obs)
        next_hidd_o_1=curr_agent.oppo_encoder1.forward(curr_next_oppo_obs)
        next_hidd_o_2=curr_agent.oppo_encoder2.forward(curr_next_oppo_obs)
        next_q_in=torch.cat((next_hidd_s,next_hidd_o_1),dim=1)
        next_q_in=torch.cat((next_q_in,next_hidd_o_2),dim=1)
        next_values = self.target_policies[agent_i].forward(next_q_in).max(1)[0].unsqueeze(1).detach()

        target_values=curr_rews.view(-1, 1)+self.gamma*(1 - dones[agent_i].view(-1, 1))*next_values
        loss = MSELoss(actual_values, target_values.detach())
        loss.backward(retain_graph=True)
        for param in curr_agent.agent_encoder.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
        for param in curr_agent.q.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
        curr_agent.optimizer.step()

        # update the belief network
        curr_agent.optimizer_op.zero_grad()
        oppo_indices=[j for j in range(self.nagents) if j!=agent_i]
        pred_weight_1=curr_agent.oppo_decoder1.forward(hidd_o_1)
        oppo_ac_indics_1=acs[oppo_indices[0]].max(1)[1].squeeze()
        oppo_loss_1=CrossEntropyLoss(pred_weight_1,oppo_ac_indics_1)
        pred_weight_2 = curr_agent.oppo_decoder2.forward(hidd_o_2)
        oppo_ac_indics_2 = acs[oppo_indices[1]].max(1)[1].squeeze()
        oppo_loss_2 = CrossEntropyLoss(pred_weight_2, oppo_ac_indics_2)
        oppo_loss=oppo_loss_1+oppo_loss_2
        oppo_loss.backward()
        for param in curr_agent.oppo_encoder1.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
        for param in curr_agent.oppo_encoder2.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
        for param in curr_agent.oppo_decoder1.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
        for param in curr_agent.oppo_decoder2.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
        curr_agent.optimizer_op.step()

        self.niter=self.niter+1
        curr_agent.EPSILON = curr_agent.EPSILON * curr_agent.EPS_DEC if curr_agent.EPSILON > \
                                                      curr_agent.EPS_MIN else curr_agent.EPS_MIN
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'q_loss': loss},
                               self.niter)
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'oppo_loss1': oppo_loss_1},
                               self.niter)
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'oppo_loss2': oppo_loss_2},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)

        """

        for a in self.agents:
            # soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_q, a.q, self.tau)

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.agent_encoder.train()
            a.oppo_encoder1.train()
            a.oppo_encoder2.train()
            a.oppo_decoder1.train()
            a.oppo_decoder2.train()
            a.q.train()
            a.target_q.train()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.agent_encoder = fn(a.agent_encoder)
                a.oppo_encoder1 = fn(a.oppo_encoder1)
                a.oppo_encoder2 = fn(a.oppo_encoder2)
                a.oppo_decoder1 = fn(a.oppo_decoder1)
                a.oppo_decoder2 = fn(a.oppo_decoder2)
                a.q = fn(a.q)
            self.pol_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_q = fn(a.target_q)
            self.trgt_pol_dev = device

    def prep_rollouts(self, device='gpu'):
        for a in self.agents:
            a.agent_encoder.eval()
            a.oppo_encoder1.eval()
            a.oppo_encoder2.eval()
            a.oppo_decoder1.eval()
            a.oppo_decoder2.eval()
            a.q.eval()
            a.target_q.eval()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        for a in self.agents:
            a.agent_encoder = fn(a.agent_encoder)
            a.oppo_encoder1=fn(a.oppo_encoder1)
            a.oppo_encoder2=fn(a.oppo_encoder2)
            a.oppo_decoder1=fn(a.oppo_decoder1)
            a.oppo_decoder2=fn(a.oppo_decoder2)
            a.q=fn(a.q)
            a.target_q=fn(a.target_q)


    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, agent_num=3, agent_alg="DRON", num_in_pol=366, num_out_pol=4, num_oppo_in=360, discrete_action=True,
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=32):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []

        for i in range(agent_num):
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_oppo_in': num_oppo_in,
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
        save_dict = torch.load(filename, map_location="cuda:0")
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance