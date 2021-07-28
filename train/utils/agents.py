import torch
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork,ActorNetwork, CriticNetwork, ValueNetwork, RnnNetwork,AttentionMLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
import torch.nn.functional as F
import numpy as np

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, num_out_critic=1, hidden_dim=64,
                 lr=0.01, discrete_action=True, use_rnn=False):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.use_rnn=use_rnn
        if use_rnn:
            self.policy = RnnNetwork(num_in_pol, num_out_pol,hidden_dim=hidden_dim,
                                     constrain_out=False,
                                     discrete_action=discrete_action)
            self.target_policy = RnnNetwork(num_in_pol, num_out_pol,hidden_dim=hidden_dim,
                                            constrain_out=False,
                                            discrete_action=discrete_action)
        else:
            self.policy = MLPNetwork(num_in_pol, num_out_pol,hidden_dim=hidden_dim,
                                     constrain_out=False,
                                     discrete_action=discrete_action)
            self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                            hidden_dim=hidden_dim,constrain_out=False,
                                            discrete_action=discrete_action)
        self.critic = MLPNetwork(num_in_critic, num_out_critic,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_critic = MLPNetwork(num_in_critic, num_out_critic,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=True, hidden_state=None):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        h_out=None

        if self.use_rnn:
            action, h_out = self.policy.forward(obs,hidden_state)
        else:
            action = self.policy.forward(obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        if self.use_rnn:
            return action, h_out
        else:
            return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class TD3Agent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic_1 = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.critic_2 = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_critic_1 = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        self.target_critic_2 = MLPNetwork(num_in_critic, 1,
                                          hidden_dim=hidden_dim,
                                          constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic_1, self.critic_1)
        hard_update(self.target_critic_2, self.critic_2)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer_1 = Adam(self.critic_1.parameters(), lr=lr)
        self.critic_optimizer_2 = Adam(self.critic_2.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic_1': self.critic_1.state_dict(),
                'critic_2': self.critic_2.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic_1': self.target_critic_1.state_dict(),
                'target_critic_2': self.target_critic_2.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer_1': self.critic_optimizer_1.state_dict(),
                'critic_optimizer_2': self.critic_optimizer_2.state_dict(),
                }

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic_1.load_state_dict(params['critic_1'])
        self.critic_2.load_state_dict(params['critic_2'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic_1.load_state_dict(params['target_critic_1'])
        self.target_critic_2.load_state_dict(params['target_critic_2'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer_1.load_state_dict(params['critic_optimizer_1'])
        self.critic_optimizer_2.load_state_dict(params['critic_optimizer_2'])


class SACAgent(object):
    def __init__(self, num_in_pol, num_out_pol, num_in_critic,num_in_value, hidden_dim=64,
                 lr=0.01, discrete_action=True, alpha=0.0003, beta=0.0003,max_action=1):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.actor = ActorNetwork(alpha, num_in_pol, n_actions=num_out_pol,
                                  name='actor', max_action=max_action)
        self.critic_1 = CriticNetwork(beta, num_in_critic,
                                      name='critic_1')
        self.critic_2 = CriticNetwork(beta, num_in_critic,
                                      name='critic_2')
        self.value = ValueNetwork(beta, num_in_value, name='value')
        self.target_value = ValueNetwork(beta, num_in_value, name='target_value')
        self.discrete_action=True

    def step(self, observation):
        action, _ = self.actor.sample_normal(observation, reparameterize=False)
        if self.discrete_action:
            action = onehot_from_logits(action)
        return action

    def get_params(self):
        return {'actor': self.actor.state_dict(),
                'critic_1': self.critic_1.state_dict(),
                'critic_2': self.critic_2.state_dict(),
                'value': self.value.state_dict(),
                'target_value':self.target_value.state_dict(),
                'actor_optimizer': self.actor.optimizer.state_dict(),
                'critic_optimizer_1': self.critic_1.optimizer.state_dict(),
                'critic_optimizer_2': self.critic_2.optimizer.state_dict(),
                'value_optimizer': self.value.optimizer.state_dict(),
                'target_value_optimizer':self.target_value.optimizer.state_dict()
                }

    def load_params(self, params):
        self.actor.load_state_dict(params['actor'])
        self.critic_1.load_state_dict(params['critic_1'])
        self.critic_2.load_state_dict(params['critic_2'])
        self.value.load_state_dict(params['value'])
        self.target_value.load_state_dict(params['target_value'])
        self.actor.optimizer.load_state_dict(params['actor_optimizer'])
        self.critic_1.optimizer.load_state_dict(params['critic_optimizer_1'])
        self.critic_2.optimizer.load_state_dict(params['critic_optimizer_2'])
        self.value.optimizer.load_state_dict(params['value_optimizer'])
        self.target_value.optimizer.load_state_dict(params['target_value_optimizer'])

class DQNAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, hidden_dim=16,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False,
                                 discrete_action=discrete_action)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False,
                                        discrete_action=discrete_action)

        hard_update(self.target_policy, self.policy)
        self.num_out_pol=num_out_pol
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.EPSILON = 1.0
        self.EPS_MIN = 0.01
        self.EPS_DEC = 0.996

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        rand = np.random.random()
        if rand> self.EPSILON:
            q_values = self.policy.forward(obs)
            action_index = q_values.max(1)[1].item()
        else:
            action_index = np.random.choice(self.num_out_pol)
        action = np.array([0, 0, 0, 0])
        action[action_index] = 1
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                }

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])


class DRONAgent(object):
    def __init__(self, num_in_pol, num_out_pol, hidden_state_dim=16,hidden_dim=32,opp_num=2,
                 lr=0.01, discrete_action=True):
        self.agent_encoder=MLPNetwork(num_in_pol,hidden_state_dim,
                                      hidden_dim=hidden_dim,
                                      constrain_out=False)
        self.oppo_encoder=MLPNetwork(num_in_pol*(1+opp_num),hidden_state_dim,
                                     hidden_dim=hidden_dim,
                                     constrain_out=False)
        self.q1 = MLPNetwork(hidden_state_dim, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False,
                                 discrete_action=discrete_action)
        self.q2 = MLPNetwork(hidden_state_dim, num_out_pol,
                             hidden_dim=hidden_dim,
                             constrain_out=False,
                             discrete_action=discrete_action)
        self.opps_q=MLPNetwork(hidden_state_dim,2,
                               hidden_dim=hidden_dim,
                               constrain_out=False)

        self.target_q1 = MLPNetwork(hidden_state_dim, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_q2 = MLPNetwork(hidden_state_dim, num_out_pol,
                                    hidden_dim=hidden_dim,
                                    constrain_out=True,
                                    discrete_action=discrete_action)
        self.target_opps_q=MLPNetwork(hidden_state_dim,2,
                               hidden_dim=hidden_dim,
                               constrain_out=False)

        hard_update(self.target_q1, self.q1)
        hard_update(self.target_q2, self.q2)
        hard_update(self.target_opps_q, self.opps_q)
        self.num_out_pol = num_out_pol
        self.optimizer_1 = Adam(self.q1.parameters(), lr=lr)
        self.optimizer_2 = Adam(self.q2.parameters(), lr=lr)
        self.optimizer_op = Adam(self.opps_q.parameters(), lr=lr)
        self.EPSILON = 1.0
        self.EPS_MIN = 0.01
        self.EPS_DEC = 0.996

    def step(self,obs,opp_obs):
        rand = np.random.random()
        if rand > self.EPSILON:
            hidd_s=self.agent_encoder.forward(obs)
            hidd_o=self.oppo_encoder.forward(torch.cat((obs,*opp_obs),dim=1))
            q_1=self.q1.forward(hidd_s)
            q_2=self.q2.forward(hidd_s)
            w_out=self.opps_q.forward(hidd_o)
            w_out=F.softmax(w_out,dim=1)
            q_total=w_out.mm(torch.cat((q_1,q_2),dim=0))
            action_index = q_total.max(1)[1].item()
        else:
            action_index = np.random.choice(self.num_out_pol)
        action = np.array([0, 0, 0, 0])
        action[action_index] = 1
        return action

    def get_params(self):
        return {'agent_encoder': self.agent_encoder.state_dict(),
                'oppo_encoder': self.oppo_encoder.state_dict(),
                'q1': self.q1.state_dict(),
                'q2': self.q2.state_dict(),
                'opps_q': self.opps_q.state_dict(),
                'target_q1': self.target_q1.state_dict(),
                'target_q2': self.target_q2.state_dict(),
                'target_opps_q': self.target_opps_q.state_dict(),
                'optimizer_1': self.optimizer_1.state_dict(),
                'optimizer_2': self.optimizer_2.state_dict(),
                'optimizer_op': self.optimizer_op.state_dict(),
                }

    def load_params(self, params):
        self.agent_encoder.load_state_dict(params['agent_encoder'])
        self.oppo_encoder.load_state_dict(params['oppo_encoder'])
        self.q1.load_state_dict(params['q1'])
        self.q2.load_state_dict(params['q2'])
        self.opps_q.load_state_dict(params['opps_q'])
        self.target_q1.load_state_dict(params['target_q1'])
        self.target_q2.load_state_dict(params['target_q2'])
        self.target_opps_q.load_state_dict(params['target_opps_q'])