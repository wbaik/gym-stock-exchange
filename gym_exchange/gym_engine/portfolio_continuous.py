import numpy as np
from gym_exchange.gym_engine.engine_continuous import EngineContinuous
import gym


# It looks like this might not be necessary at all...
# Since we've changed a number of things...
class PortfolioContinuous(EngineContinuous):
    def __init__(self, tickers, start_date, num_days_iter,
                 today=None, seed=None, render=False,
                 action_space_min=0.0, action_space_max=1.0):
        num_action_space = len(tickers)
        super().__init__(tickers, start_date, num_days_iter,
                         today, seed, num_action_space, render,
                         action_space_min=action_space_min,
                         action_space_max=action_space_max)
        self.action_space = gym.spaces.Box(action_space_min, action_space_max,
                                           (num_action_space, ), np.float32)

    def step(self, actions):
        return super(PortfolioContinuous, self).step(actions)