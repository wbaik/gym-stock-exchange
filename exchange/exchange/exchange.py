from exchange.portfolio import Portfolio
from injector import inject

import gym


class Exchange(gym.Env):

    @inject
    def __init__(self, portfolio: Portfolio):
        super().__init__()
        self.portfolio = portfolio

    def step(self, actions):
        return self.portfolio.step(actions)

    def reset(self):
        self.portfolio.reset()

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        super().close()


