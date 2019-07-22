import gym.spaces as spaces

from gym_exchange.envs.stock_exchange_base import StockExchangeBase
from gym_exchange.gym_engine import EngineContinuous, PortfolioContinuous
import numpy as np


class StockExchangeContinuous(StockExchangeBase):
    metadata = {'render.modes': ['human']}

    # Keep tickers in a list or an iterable...
    tickers = ['aapl',]
    start_date = '2013-09-15'
    num_days_to_iterate = 100
    num_state_space = 20
    num_action_space = len(tickers)
    # no_action_index is truly no_action only if it's not a Portfolio
    no_action_index = num_action_space//2
    today = 0
    render = False
    # set to None when not using Portfolio
    action_space_min = -1.0
    action_space_max = 1.0
    # For each ticker state: ohlc
    num_state_per_ticker = 4

    def __init__(self, seed=None):

        # Could manually throw in options eventually...
        self.portfolio = self.num_action_space > 1
        self._seed = seed

        if self.portfolio:
            assert self.action_space_min is not None
            assert self.action_space_max is not None
            self.env = PortfolioContinuous(self.tickers, self.start_date,
                                           self.num_days_to_iterate,
                                           self.today, seed, render=self.render,
                                           action_space_min=self.action_space_min,
                                           action_space_max=self.action_space_max)
        else:
            assert self.num_action_space % 2 != 0, 'NUM_ACTION_SPACE MUST BE ODD TO HAVE NO ACTION INDEX'
            self.env = EngineContinuous(self.tickers, self.start_date,
                                        self.num_days_to_iterate,
                                        self.today, seed,
                                        render=self.render)

        self.action_space = spaces.Box(self.action_space_min, self.action_space_max,
                                       (self.num_action_space, ), np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0,
                                            (self.num_state_space,
                                             self.num_action_space * self.num_state_per_ticker),
                                            dtype=np.float32)
        self.state = self.get_running_state()
        super().__init__()
