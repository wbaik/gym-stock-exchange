import gym
import gym_stock_exchange
import random
import time


class RandomAgent:
    def __init__(self, len, valid_action_range):
        self.len = len
        self.valid_action_range = valid_action_range

    def action(self):
        return [random.randint(*self.valid_action_range)
                for _ in range(self.len)]


if __name__ == '__main__':
    env = gym.make('game-stock-exchange-v0')

    TICKERS_OF_INTEREST = ['aapl', 'googl', 'amd', 'pg']
    env.create_engine(TICKERS_OF_INTEREST, '2014-01-01', 1000, render=True)

    VALID_ACTIONS = (0, 2)
    random_agent = RandomAgent(len(TICKERS_OF_INTEREST), VALID_ACTIONS)

    for _ in range(100):
        obs, rewards, dones, info = env.step(random_agent.action())
        env.render()

        if dones:
            break

    time.sleep(10)
