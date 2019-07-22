from itertools import count
import numpy as np
import pandas as pd
from gym_exchange.gym_engine import EngineDiscrete
from gym_exchange.gym_engine import iterable


class PortfolioDiscrete(EngineDiscrete):
    def __init__(self, tickers, start_date, num_days_iter,
                 today=None, seed=None, render=False,
                 action_space_min=0.0, action_space_max=1.0):
        num_action_space = len(tickers)
        super().__init__(tickers, start_date, num_days_iter,
                         today, seed, num_action_space, render,
                         action_space_min=action_space_min,
                         action_space_max=action_space_max)
        self.action_space = np.linspace(action_space_min, action_space_max, num_action_space)
        self.position_df = self._get_position_df(tickers, num_action_space, action_space_min, action_space_max)

    # Returns a dataframe of all possible position distributions
    # positions are divided into increments inversely proportional to the length of tickers
    def _get_position_df(self, tickers, num_action_spaces,
                         action_space_min, action_space_max):

        def all_possible_combinations(values, increment, proportion_remaining, idx, ret_list):
            if idx == len(values) - 1:
                values[idx] = max(0, proportion_remaining)
                ret_list.append(list(values))
                values[idx] = 0
                return
            for t in count():
                add_to_this_idx = t * increment
                if add_to_this_idx <= proportion_remaining:
                    values[idx] = add_to_this_idx
                    all_possible_combinations(values, increment,
                                              proportion_remaining - add_to_this_idx,
                                              idx + 1, ret_list)
                    values[idx] = 0.0
                else:
                    break

        ticker_length = len(tickers)
        inc = (action_space_max - action_space_min) / (num_action_spaces - 1)
        so_far = [0.0 for _ in range(ticker_length)]

        # Total positions sum to 1
        ret, remain = [], 1.0
        all_possible_combinations(so_far, inc, remain, 0, ret)
        return pd.DataFrame(ret).T

    def moves_available(self):
        return self.position_df.shape[1]

    def step(self, action_index):
        assert not iterable(action_index), f'{action_index}'
        assert 0 <= action_index < self.moves_available(), \
            f'action_index: {action_index}, moves_avail: {self.moves_available()}'

        positions = self.position_df[action_index]

        actions = [np.argwhere(np.isclose(self.action_space, position)).item()
                   for position in positions]
        return super(PortfolioDiscrete, self).step(actions)