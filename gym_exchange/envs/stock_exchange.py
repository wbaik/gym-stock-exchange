import gym
import gym.spaces as spaces
from gym_exchange.engine import Engine
import numpy as np
import pandas as pd


class StockExchange(gym.Env):
    metadata = {'render.modes': ['human']}

    tickers = 'aapl'
    start_date = '2014-01-01'
    num_days_to_iterate = 1000
    num_state_space = 20
    num_action_space = 3
    no_action_index = 1
    today = 0
    render = False

    def __init__(self, seed=None):
        self._seed = seed
        self.action_space = spaces.Discrete(self.num_action_space)
        self.observation_space = spaces.Box(-1.0, 10.0, (self.num_state_space, ), dtype=np.float)
        self.env = Engine(self.tickers, self.start_date, self.num_days_to_iterate,
                          self.today, seed,
                          action_space=self.num_action_space, render=self.render)
        self.state = self.get_running_state()
        self.reset()

    def step(self, actions):
        # I can fix Engine to return state from `self.env.step(action)`
        reward, ended = self.env.step(actions)
        self.state = self.add_new_state(self.env.get_state())
        return self.state, reward, ended, {'score': reward}

    def reset(self):
        self.env.reset_game()
        self.initialize_state()
        return self.state

    def render(self, mode='human', close=False):
        self.env.render()

    def initialize_state(self):
        for _ in range(self.num_state_space - 1):
            next_state, reward, done, _ = self.step(self.no_action_index)
            assert reward == 0.0, f'Reward is somehow {reward}'

    def moves_available(self):
        return self.env.moves_available()

    def __repr__(self):
        return repr(self.env)

    def get_running_state(self):
        return np.zeros(self.num_state_space) # .tolist()

    def add_new_state(self, new_state_to_add):
        if isinstance(new_state_to_add, list):
            new_state_to_add = new_state_to_add[0]
        running_state_orig = self.state
        running_state = pd.Series(running_state_orig).shift(-1)
        # Assign new price to index == last_elem - 1
        running_state.iloc[-1] = new_state_to_add.item(0)
        # running_state.iloc[-2] = new_state_to_add.item(0)
        # Assign new position to index == last_elem
        # running_state.iloc[-1] = new_state_to_add.item(1)
        assert len(running_state_orig) == len(running_state)
        return np.array(running_state)
        # return running_state.tolist()

