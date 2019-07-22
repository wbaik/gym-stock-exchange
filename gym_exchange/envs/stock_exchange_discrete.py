import gym.spaces as spaces

from gym_exchange.envs.stock_exchange_base import StockExchangeBase
from gym_exchange.gym_engine import EngineDiscrete, PortfolioDiscrete
import numpy as np


class StockExchangeDiscrete(StockExchangeBase):
    metadata = {'render.modes': ['human']}

    # Keep tickers in a list or an iterable...
    tickers = ['aapl', 'amd', 'msft', 'intc', 'd', 'sbux', 'atvi',
               'ibm', 'ual', 'vrsn', 't', 'mcd', 'vz']
    start_date = '2013-09-15'
    num_days_to_iterate = 1000
    num_state_space = 20
    # if Portfolio, set it to length of tickers
    # else, must be odd
    num_action_space = len(tickers)
    # no_action_index is truly no_action only if it's not a Portfolio
    no_action_index = num_action_space//2
    today = 0
    render = False
    # set to None when not using Portfolio
    action_space_min = 0.0
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
            self.env = PortfolioDiscrete(self.tickers, self.start_date,
                                         self.num_days_to_iterate,
                                         self.today, seed, render=self.render,
                                         action_space_min=self.action_space_min,
                                         action_space_max=self.action_space_max)
        else:
            assert self.num_action_space > 2, 'NUM_ACTION_SPACE SHOULD BE GREATER THAN 2'
            assert self.num_action_space % 2 != 0, 'NUM_ACTION_SPACE MUST BE ODD TO HAVE NO ACTION INDEX'
            self.env = EngineDiscrete(self.tickers, self.start_date, self.num_days_to_iterate,
                                      self.today, seed,
                                      num_action_space=self.num_action_space, render=self.render)

        self.action_space = spaces.Box(self.action_space_min, self.action_space_max, (self.num_action_space,))
        # self.action_space = spaces.Discrete(self.env.moves_available())
        self.observation_space = spaces.Box(-1.0, 1.0, (self.num_state_space, self.num_action_space), dtype=np.float)
        self.state = self.get_running_state()
        super().__init__()
