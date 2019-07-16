import gym
from exchange.portfolio import Portfolio


class Exchange(gym.Env):

    def __init__(self, config) -> None:
        super().__init__()
        self.portfolio = Portfolio(config)

    def step(self, action):
        self.portfolio.step(action)

    def reset(self):
        self.portfolio.reset()

    def render(self, mode='human'):
        pass

    def close(self):
        super().close()


