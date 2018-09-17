import gym
import gym_stock_exchange
import random
import time


if __name__ == '__main__':
    env = gym.make('game-stock-exchange-v0')
    env.create_engine(['aapl', 'googl'], '2015-01-01', 130)

    for _ in range(100):
        actions = [random.randint(0, 20) for _ in range(2)]
        env.step(actions)
        env.env.render()

    while True:
        time.sleep(100)