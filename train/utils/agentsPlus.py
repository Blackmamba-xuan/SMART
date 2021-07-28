import torch
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork,ActorNetwork, CriticNetwork, ValueNetwork, RnnNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
import torch.nn.functional as F
import numpy as np

class DronConAgent(object):
    def __init__(self, num_in_pol, num_out_pol, num_oppo_in,hidden_state_dim=16,hidden_dim=32,opp_num=2,
                 lr=0.01, discrete_action=True):
        self.agent_encoder=MLPNetwork(num_in_pol,hidden_state_dim,
                                      hidden_dim=hidden_dim,
                                      constrain_out=False)
        self.oppo_encoder1=MLPNetwork(num_oppo_in,hidden_state_dim,
                                     hidden_dim=hidden_dim,
                                     constrain_out=False)
        self.oppo_encoder2 = MLPNetwork(num_oppo_in, hidden_state_dim,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        self.oppo_decoder1 = MLPNetwork(hidden_state_dim, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        self.oppo_decoder2 = MLPNetwork(hidden_state_dim, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        self.q = MLPNetwork(hidden_state_dim*3, num_out_pol,
                             hidden_dim=hidden_dim,
                             constrain_out=False,
                             discrete_action=discrete_action)
        self.target_q = MLPNetwork(hidden_state_dim * 3, num_out_pol,
                            hidden_dim=hidden_dim,
                            constrain_out=False,
                            discrete_action=discrete_action)
        hard_update(self.target_q, self.q)
        self.num_out_pol = num_out_pol
        agent_params=list(self.agent_encoder.parameters())+list(self.q.parameters())
        self.oppo_params=list(self.oppo_encoder1.parameters())+list(self.oppo_encoder2.parameters())\
                    +list(self.oppo_decoder1.parameters())+list(self.oppo_decoder2.parameters())
        self.optimizer = Adam(agent_params, lr=lr)
        self.optimizer_op = Adam(self.oppo_params, lr=lr)
        self.EPSILON = 1.0
        self.EPS_MIN = 0.01
        self.EPS_DEC = 0.996


    def step(self,obs,opp_obs):
        rand = np.random.random()
        if rand > self.EPSILON:
            hidd_s=self.agent_encoder.forward(obs)
            hidd_o_1=self.oppo_encoder1.forward(opp_obs)
            hidd_o_2=self.oppo_encoder2.forward(opp_obs)
            q_in=torch.cat((hidd_s,hidd_o_1),dim=1)
            q_in=torch.cat((q_in,hidd_o_2),dim=1)
            q_values=self.q.forward(q_in)
            action_index = q_values.max(1)[1].item()
        else:
            action_index = np.random.choice(self.num_out_pol)
        action = np.array([0, 0, 0, 0])
        action[action_index] = 1
        return action

    def get_params(self):
        return {'agent_encoder': self.agent_encoder.state_dict(),
                'oppo_encoder1': self.oppo_encoder1.state_dict(),
                'oppo_encoder2': self.oppo_encoder2.state_dict(),
                'oppo_decoder1': self.oppo_decoder1.state_dict(),
                'oppo_decoder2': self.oppo_decoder2.state_dict(),
                'q': self.q.state_dict(),
                'target_q': self.target_q.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'optimizer_op': self.optimizer_op.state_dict(),
                }

    def load_params(self, params):
        self.agent_encoder.load_state_dict(params['agent_encoder'])
        self.oppo_encoder1.load_state_dict(params['oppo_encoder1'])
        self.oppo_encoder2.load_state_dict(params['oppo_encoder2'])
        self.oppo_decoder1.load_state_dict(params['oppo_decoder1'])
        self.oppo_decoder2.load_state_dict(params['oppo_decoder2'])
        self.q.load_state_dict(params['q'])
        self.target_q.load_state_dict(params['target_q'])