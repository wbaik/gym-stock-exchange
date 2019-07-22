import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.ion()


class TickerBase:
    def __init__(self,
                 ticker,
                 start_date,
                 num_days_iter,
                 today=None,
                 test=False):
        self.ticker = str.upper(ticker)
        self.start_date = start_date
        self.num_days_iter = num_days_iter
        self.df, self.dates = self._load_df(test)
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

    def step(self, action):
        raise NotImplementedError

    def valid_action(self, action):
        raise NotImplementedError

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
