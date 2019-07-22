import itertools
import functools
import matplotlib.pyplot as plt
import numpy as np
from gym_exchange.gym_engine import iterable

plt.ion()


class EngineBase:
    def __init__(self,
                 tickers,
                 seed=None,
                 render=False,
                 *args,
                 **kwargs):

        if seed:
            np.random.seed(seed)
        if not iterable(tickers):
            tickers = [tickers]

        self.reset_game()

        if render:
            # Somehow ax_list should be grouped in two always...
            # Or is there another way of getting one axis per row and then add?
            fig_height = 3 * len(self.tickers)
            self.fig, self.ax_list = plt.subplots(len(tickers), 2, figsize=(10, fig_height))

    def reset_game(self):
        list(map(lambda ticker: ticker.reset(), self.tickers))

    def _render(self, render):
        if render:
            if len(self.tickers) == 1:
                self.tickers[0].render(self.ax_list)
            else:
                for axis, ticker in zip(self.ax_list, self.tickers):
                    ticker.render(axis)

    def get_state(self, delta_t=0):
        # Note: np.arary(...) could also be used
        return list(map(lambda ticker: ticker.get_state(delta_t), self.tickers))

    def moves_available(self):
        raise NotImplementedError

    def step(self, actions):
        if not iterable(actions): actions = [actions]
        assert len(self.tickers) == len(actions), f'{len(self.tickers)}, {len(actions)}'

        rewards, dones = zip(*(itertools.starmap(lambda ticker, action: ticker.step(action),
                                                 zip(self.tickers, actions))))

        # This is somewhat misleading
        score = functools.reduce(lambda x, y: x + y, rewards, 0.0)
        done = functools.reduce(lambda x, y: x | y, dones, False)

        return score, done

    def render(self):
        # This is possibly unnecessary b/c of changes
        self._render(True)

    def __repr__(self):
        tickers = [f'ticker_{i}: {ticker.ticker}, ' for i, ticker in enumerate(self.tickers)]
        return str(tickers)

    def _data_valid(self):
        raise NotImplementedError

