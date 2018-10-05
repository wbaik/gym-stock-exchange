import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.ion()

class Ticker:
    def __init__(self, ticker, start_date, num_days_iter,
                 today=None, num_actions=3, test=False,
                 action_space_min=-1.0, action_space_max=1.0):
        self.ticker = str.upper(ticker)
        self.start_date = start_date
        self.num_days_iter = num_days_iter
        self.df, self.dates = self._load_df(test)
        self.action_space = np.linspace(action_space_min, action_space_max, num_actions)
        self.today = 0 if today is None else today
        self._data_valid()
        self.current_position = self.accumulated_pnl = 0.0

    def _load_df(self, test):
        if test:
            ticker_data = self._load_test_df()
        else:
            ticker_data = pd.read_csv(f'iexfinance/iexdata/{self.ticker}')
            ticker_data = ticker_data[ticker_data.date >= self.start_date]

        ticker_data.reset_index(inplace=True)
        # This is really cheating but...
        dates_series = ticker_data['date']
        ticker_data.drop('date', axis=1, inplace=True)
        # This part should become a function eventually
        ticker_data_delta = ticker_data.pct_change()
        add_str_delta = lambda x: x + '_delta'
        ticker_data_delta.rename(add_str_delta, axis='columns', inplace=True)
        ticker_data_delta.iloc[0, :] = 0.0

        zeros = pd.DataFrame(np.zeros((len(ticker_data), 2)),
                             columns=['position', 'pnl'])

        # It's probably better to transpose, then let columns be dates, but wtf...
        df = pd.concat([ticker_data, ticker_data_delta, zeros], axis=1)
        df.drop(['index', 'index_delta'], axis=1, inplace=True)

        return df, dates_series

    def _load_test_df(self):
        date_col = [datetime.date.today() + datetime.timedelta(days=i)
                    for i in range(self.num_days_iter)]
        aranged_values = [np.repeat(i, 6) for i in range(1, self.num_days_iter+1)]
        temp_df = pd.DataFrame(aranged_values,
                               columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        temp_df.iloc[:, 0] = date_col
        return temp_df

    def _data_valid(self):
        assert len(self.df) >= self.num_days_iter, \
                f'DataFrame shape: {self.df.shape}, num_days_iter: {self.num_days_iter}'
        assert len(self.df) == len(self.dates), \
                f'df.shape: {self.df.shape}, dates.shape:{self.dates.shape}'

    def get_state(self, delta_t=0):
        today_market_data_position = np.array(self.df.iloc[self.today+delta_t, -7:-2])
        today_market_data_position[-1] = self.current_position
        return today_market_data_position

    # 1. Reward is tricky
    # 2. Should invalid action be penalized?
    def step(self, action):
        if not self.done():
            # Record pnl
            # This implementation of reward is such a hogwash!!
            #     but recall, Deepmind's DQN solution does something similar...
            #     assigning credit is always hard...
            # Pandas complain here, "A value is trying to be set on a copy of a slice from a DataFrame"
            #     but the suggested solution is actually misleading... so leaving it as is
            pd.set_option('mode.chained_assignment', None)
            self.df.pnl[self.today] = reward = 0.0 if self.today == 0 else \
                                               self.current_position * self.df.close_delta[self.today]

            # Think about accumulating the scores...
            self.accumulated_pnl += reward

            self.df.position[self.today] = self.current_position = self.action_space[action]

            # new_position_delta = self.action_space[action] if self.today == 0 or \
            #                                                   self.valid_action(action) else 0.0
            # self.current_position = self.df.position[self.today] = \
            #                                 new_position_delta if self.today == 0 else \
            #                                 self.df.position[self.today-1] + new_position_delta

            # If we penalize invalid moves, the model learns to avoid it, but this deters
            #     learning experiences. It only focuses on position such that it avoids penalty.
            #     Penalty of -0.1 is certainly too high for this matter
            #
            # if self.valid_action(action):
            #     new_position_delta = self.action_space[action]
            #     self.current_position = self.df.position[self.today] = \
            #                                     new_position_delta if self.today == 0 else \
            #                                     self.df.position[self.today-1] + new_position_delta
            #
            # else:
            #     self.current_position = self.df.position[self.today] = self.df.position[self.today-1]
            #     reward = -0.1 # This is tricky, should consider the distribution of the returns

            self.today += 1
            # Think about how to re-allocate the reward
            return reward, False
        else:
            self.current_position = 0.0
            return 0.0, True

    def valid_action(self, action):
        if self.today == 0: return True
        # current_position = self.df.position[self.today-1]
        # return -1.0 <= current_position + self.action_space[action] <= 1.0
        # The above approach causes shitty troubles...

    def reset(self):
        self.today = 0
        self.df.position = self.df.pnl = 0.0
        self.current_position = self.accumulated_pnl = 0.0

    # NOT THE MOST EFFICIENT...
    def done(self):
        return self.today > self.num_days_iter

    def render(self, axis):
        # market_data, position = self.get_state()
        # axis[0].scatter(self.today, self.df.pnl[self.today-1])
        axis[0].set_ylabel(f'Daily price: {self.ticker}')
        axis[0].set_xlabel('Time step')
        axis[0].plot(np.arange(self.today), self.df.close[:self.today])
        # axis[1].scatter(self.today, position)
        # axis[2].scatter(self.today, self.accumulated_pnl)
        axis[1].set_ylabel(f'Daily return from Agent')
        axis[1].set_xlabel('Time step')
        axis[1].scatter(self.today, self.accumulated_pnl)
        plt.pause(0.0001)
