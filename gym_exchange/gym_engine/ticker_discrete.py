import numpy as np
import pandas as pd

from gym_exchange.gym_engine.ticker_base import TickerBase


class TickerDiscrete(TickerBase):
    def __init__(self,
                 ticker,
                 start_date,
                 num_days_iter,
                 today=None,
                 num_actions=3,
                 test=False,
                 action_space_min=-1.0,
                 action_space_max=1.0):
        self.action_space = np.linspace(action_space_min, action_space_max, num_actions)
        super().__init__(ticker, start_date, num_days_iter, today, test)

    # Discrete is different from Continuous in Action Space
    # 1. Reward is tricky
    # 2. Should invalid action be penalized?
    def step(self, action):
        if not self.done():
            # Record pnl
            # This implementation of reward is such a hogwash!!
            #     but recall, Deepmind's DQN solution does something similar...
            #     assigning credit is always hard...
            # Pandas complain here, "A value is trying to be set on a copy of a slice from a DataFrame"
            #     but the suggested solution is actually misleading... so leaving it as is
            pd.set_option('mode.chained_assignment', None)
            self.df.pnl[self.today] = reward = 0.0 if self.today == 0 else \
                                               self.current_position * self.df.close_delta[self.today]

            # Think about accumulating the scores...
            self.accumulated_pnl += reward
            self.df.position[self.today] = self.current_position = self.action_space[action]

            self.today += 1
            # Think about how to re-allocate the reward
            return reward, False
        else:
            self.current_position = 0.0
            return 0.0, True

    def valid_action(self, action):
        current_position = self.df.position[self.today-1]
        return -1.0 <= current_position + self.action_space[action] <= 1.0
