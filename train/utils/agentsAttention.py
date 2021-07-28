import torch
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from train.utils.networks import MLPNetwork,ActorNetwork, CriticNetwork, ValueNetwork, RnnNetwork,AttentionMLPNetwork
from train.utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from train.utils.noise import OUNoise
import torch.nn.functional as F
import numpy as np

class DQNAttentionAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol,attention_dim, hidden_dim=16,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = AttentionMLPNetwork(num_in_pol, num_out_pol,attention_dim,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False,
                                 discrete_action=discrete_action)
        self.target_policy = AttentionMLPNetwork(num_in_pol, num_out_pol,attention_dim,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False,
                                        discrete_action=discrete_action)

        hard_update(self.target_policy, self.policy)
        self.num_out_pol=num_out_pol
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.EPSILON = 1.0
        self.EPS_MIN = 0.01
        self.EPS_DEC = 0.996

    def step(self, obs_x1,obs_x2, explore=False):
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
            q_values = self.policy.forward(obs_x1,obs_x2)
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