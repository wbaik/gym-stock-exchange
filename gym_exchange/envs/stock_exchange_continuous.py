import gym
import gym.spaces as spaces
from gym_exchange.gym_engine import EngineContinuous, PortfolioContinuous
import numpy as np
import pandas as pd


class StockExchangeContinuous(gym.Env):
    metadata = {'render.modes': ['human']}

    # Keep tickers in a list or an iterable...
    tickers = ['aapl',]
    start_date = '2013-09-15'
    num_days_to_iterate = 100
    num_days_in_state = 20
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
                                        num_action_space=self.num_action_space,
                                        render=self.render)

        self.action_space = spaces.Box(self.action_space_min, self.action_space_max,
                                       (self.num_action_space, ), np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0,
                                            (self.num_days_in_state,
                                             self.num_action_space * self.num_state_per_ticker),
                                            dtype=np.float32)
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
        for _ in range(self.num_days_in_state - 1):
            if self.portfolio:
                zero_action = [0.0] * self.num_action_space
                next_state, reward, done, _ = self.step(zero_action)
            else:
                next_state, reward, done, _ = self.step([self.no_action_index] * self.num_action_space)
                assert reward == 0.0, f'Reward is somehow {reward}'

    def __repr__(self):
        return repr(self.env)

    def get_running_state(self):
        return np.zeros((self.num_days_in_state, self.num_state_per_ticker * self.num_action_space))

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

