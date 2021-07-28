import imp
import os.path as osp
import os
import SMART.scenarios as scenarios

class Env(object):
    def __init__(self,scenario='crosslane',state_adapter=None,action_adapter=None,reward_adapter=None):
        print('creating environment....')
        self.scenario=scenarios.load(scenario + ".py").Scenario()

    def reset(self):
        self.scenario.reset()

    def step(self):
        self.scenario.step()

