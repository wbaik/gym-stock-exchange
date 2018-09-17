import gym
import gym.spaces as spaces
from gym import error, utils
from gym.utils import seeding
from gym_stock_exchange.engine import Engine

import numpy as np
import random
import six
import sys

class StockExchange(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=None):
        self.action_space = spaces.Discrete(21)
        self._seed = seed
        self.env = None

    def create_engine(self, tickers, start_date, num_iter_days, today=None, seed=None):
        self.env = Engine(tickers, start_date, num_iter_days, today, seed)
        self.reset()

    def step(self, actions):
        assert self.env is not None

        reward, ended = self.env.step(actions)
        return self.env.get_state(), reward, ended, {'score': self.env.score}

    def reset(self):
        # if self.env is None:
        #     self.create_engine(['AAPL'], '2015-01-01', 10)
        assert self.env is not None
        self.env.reset_game()
        return self.env.get_state()

    def render(self, mode='human', close=False):
        self.env.render()

    def get_state(self, delta_t=0):
        return self.env.get_state(delta_t)

    def moves_available(self):
        return self.env.moves_available()

    def __repr__(self):
        return repr(self.env)
