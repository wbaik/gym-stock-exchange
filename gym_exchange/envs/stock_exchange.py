import gym
import gym.spaces as spaces
from gym_engine import Engine, Portfolio
import numpy as np
import pandas as pd


class StockExchange(gym.Env):
    metadata = {'render.modes': ['human']}

    # Keep tickers in a list or iterables...
    tickers = ['aapl', 'amd', 'msft', 'intc', 'd', 'sbux', 'atvi', 'ibm', 'ual', 'vrsn', 't', 'mcd', 'vz']
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

        self.ticker_length = len(self.tickers)
        # Could manually throw in options eventually...
        self.portfolio = self.ticker_length > 1
        self._seed = seed

        if self.portfolio:
            assert self.num_action_space == self.ticker_length
            assert self.action_space_min is not None
            assert self.action_space_max is not None
            self.env = Portfolio(self.tickers, self.start_date, self.num_days_to_iterate,
                                 self.today, seed, render=self.render,
                                 action_space_min=self.action_space_min,
                                 action_space_max=self.action_space_max)
        else:
            assert self.num_action_space > 2, 'NUM_ACTION_SPACE SHOULD BE GREATER THAN 2'
            assert self.num_action_space % 2 != 0, 'NUM_ACTION_SPACE MUST BE ODD TO HAVE NO ACTION INDEX'
            self.env = Engine(self.tickers, self.start_date, self.num_days_to_iterate,
                              self.today, seed,
                              num_action_space=self.num_action_space, render=self.render)

        self.action_space = spaces.Discrete(self.num_action_space)
        self.observation_space = spaces.Box(-1.0, 2.0, (self.num_state_space,
                                                        self.num_state_per_ticker * self.ticker_length), dtype=np.float)
        self.state = self.get_running_state()
        self.reset()

    def step(self, actions):
        # I can fix Engine to return state from `self.env.step(action)`
        reward, ended = self.env.step(actions)
        self.state = self.add_new_state(self.env.get_state())
        return self.state, reward, ended, {'score': reward}

    def reset(self):
        self.env.reset_game()
        self._initialize_state()
        return self.state

    def render(self, mode='human', close=False):
        self.env.render()

    def _initialize_state(self):
        for _ in range(self.num_state_space - 1):
            if self.portfolio:
                random_moves = np.random.randint(0, self.moves_available())
                next_state, reward, done, _ = self.step(random_moves)
            else:
                next_state, reward, done, _ = self.step([self.no_action_index] * self.ticker_length)
                assert reward == 0.0, f'Reward is somehow {reward}'

    def moves_available(self):
        return self.env.moves_available()

    def __repr__(self):
        return repr(self.env)

    def get_running_state(self):
        return np.zeros((self.num_state_space, self.num_state_per_ticker * self.ticker_length))

    def add_new_state(self, new_states_to_add):
        assert isinstance(new_states_to_add, list), type(new_states_to_add)
        # Disregarding the last elem in each state because it's the holdings...
        # Maybe just get rid of that altogether?
        new_states = np.array([state[:-1].tolist() for state in new_states_to_add]).flatten()

        running_state_orig = self.state
        running_state = pd.DataFrame(running_state_orig).shift(-1)

        # Assign new price to index == last_elem - 1
        running_state.iloc[-1] = new_states.squeeze()

        # Deprecated...
        # running_state.iloc[-2] = new_state_to_add.item(0)
        # Assign new position to index == last_elem
        # running_state.iloc[-1] = new_state_to_add.item(1)

        assert len(running_state_orig) == len(running_state)
        return np.array(running_state)

