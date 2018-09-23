import gym
import gym.spaces as spaces
from gym_stock_exchange.engine import Engine


class StockExchange(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=None):
        self._seed = seed
        self.env = None

    def create_engine(self, tickers, start_date, num_iter_days,
                      num_action_space=3, today=None, seed=None, render=False):
        self.num_action_space = num_action_space
        self.action_space = spaces.Discrete(num_action_space)

        self.env = Engine(tickers, start_date, num_iter_days, today, seed,
                          action_space=num_action_space, render=render)
        self.reset()

    def step(self, actions):
        assert self.env is not None

        reward, ended = self.env.step(actions)
        return self.get_state(), reward, ended, {'score': reward}

    def reset(self):
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
