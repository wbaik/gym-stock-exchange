import gym
import numpy as np
import pandas as pd


class StockExchangeBase(gym.Env):
    def __init__(self):
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
                next_state, reward, done, _ = self.step([self.no_action_index] * self.num_action_space)
                assert reward == 0.0, f'Reward is somehow {reward}'

    def moves_available(self):
        return self.env.moves_available()

    def __repr__(self):
        return repr(self.env)

    def get_running_state(self):
        return np.zeros((self.num_state_space, self.num_state_per_ticker * self.num_action_space))

    def add_new_state(self, new_states_to_add):
        assert isinstance(new_states_to_add, list), type(new_states_to_add)
        # Disregarding the last elem in each state because it's the holdings...
        # Maybe just get rid of that altogether?
        new_states = np.array([state[:-1].tolist() for state in new_states_to_add]).flatten()

        running_state_orig = self.state
        running_state = pd.DataFrame(running_state_orig).shift(-1)

        # Assign new price to index == last_elem - 1
        running_state.iloc[-1] = new_states.squeeze()

        assert len(running_state_orig) == len(running_state)
        return np.array(running_state)

