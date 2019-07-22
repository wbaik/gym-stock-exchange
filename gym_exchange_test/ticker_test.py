import numpy as np
import unittest
from gym_exchange.gym_engine import TickerDiscrete


class TestTicker(unittest.TestCase):

    # This is a terrible design
    def setUp(self):
        self.num_actions = 3
        self.num_iter = self.num_actions * 4
        self.ticker = TickerDiscrete('aapl', '2015-01-01', self.num_iter,
                                     num_actions=self.num_actions,
                                     test=True)

    def tearDown(self):
        del self.ticker

    def get_actions(self):
        return np.repeat(np.arange(self.num_actions), self.num_actions)

    def get_states(self):
        states = []
        for action in self.get_actions():
            self.ticker.step(action)
            states += [self.ticker.get_state()]
        self.ticker.reset()
        return np.array(states)

    # Reset required...
    def get_rewards(self):
        return list(map(lambda x: self.ticker.step(x), self.get_actions()))

    # This is a terrible design, I don't like it
    # Reset required...
    # Might as well setup decorators...
    def take_steps_yield_rewards(self):
        rewards, _ = zip(*self.get_rewards())
        return rewards

    def get_prices_delta(self):
        return np.array(self.ticker.df.close_delta[1:self.ticker.today])

    def get_positions_two_less(self):
        return np.array(self.ticker.df.position[:self.ticker.today-1])

    def get_pnl_one_less(self):
        return np.array(self.ticker.df.pnl[:self.ticker.today])

    # This is a terrible design, I don't like it
    def test_reward_agrees_positions(self):
        self.get_rewards()
        self.assertEqual(np.sum(self.get_prices_delta() *
                                self.get_positions_two_less()),
                         np.sum(self.get_pnl_one_less()))
        self.ticker.reset()

    def test_shape(self):
        self.assertEqual(self.ticker.df.shape, (self.num_iter, 12))

    def test_steps(self):
        self.assertTrue(np.array_equal(self.take_steps_yield_rewards(),
                                       self.get_pnl_one_less()))
        self.ticker.reset()

    def test_states_1(self):
        self.assertEqual(np.sum(self.get_states()[:, -1].flatten()), 0)

    def test_states_2(self):
        self.assertEqual(np.sum(self.get_states()[:, -1].flatten()),
                         np.sum(list(map(lambda x: self.ticker.action_space[x], self.get_actions()))))
        self.ticker.reset()


if __name__ == '__main__':
    unittest.main()