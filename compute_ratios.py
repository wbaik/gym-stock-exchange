from dateutil import parser
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from postgres_utils import PostgresPandas


compound = lambda x: (1 + x).prod() - 1
daily_sr = lambda x: x.mean() / x.std()


def create_dataframes(symbols):

    to_merge = []
    for s in symbols:
        try:
            queries = '''
                select date, close
                from {}
                '''.format(s)
            ret = ppd.run_query(queries)
            ret.columns = ['date', str(s)]
            to_merge.append(ret)
        except:
            print('*** Error occurred on {}'.format(s))

    merged_date = reduce(lambda x, y:
                         pd.merge(x, y, on='date', how='outer'), to_merge)
    # When merged, sort by the dates
    merged_date.sort_values('date', inplace=True)
    merged_date.date = merged_date.date.map(lambda x: parser.parse(x))
    merged_date.set_index('date', inplace=True, drop=True)

    return merged_date


def momentum_signal(df, lookback, lag):

    def momentum_function(given_series):
        # can be lambda x: , or .apply(np.sum) ...
        return np.sum(given_series)

    # applymap could be necessary
    signal = df.rolling(lookback).apply(momentum_function)

    return signal.shift(lag)


def rolling_example():

    def to_index(rets):
        index = (1+rets).cumprod()

        first_loc = index.notnull().idxmax()
        index[first_loc] = 1
        return index

    def sharpe(ret, ann=250):
        factor = np.sqrt(ann)
        return ret.rolling(ann, ann).apply(lambda x:
                                           x.mean() / x.std() * factor)

    queries = '''
                select date, close
                from {};
                '''.format('NOC')
    ret = ppd.run_query(queries)
    ret.close = ret.close.pct_change()

    m_signal = momentum_signal(ret, 5, 5)
    m_signal.index = ret.index = ret['date'].map(lambda x: parser.parse(x))
    every_friday_at_close = m_signal.resample('W-FRI').resample('B', fill_method='ffill').shift(1)

    trade_rets = every_friday_at_close.close * ret.close
    to_index(trade_rets).plot(); plt.show()
    vol = ret.close.rolling(252, 200).std()*np.sqrt(252)
    vol.plot(); plt.show()
    sharpe(ret.close).plot(); plt.show()


def calc_mom(price, lookback, lag):
    # lag time is how many days to wait for, till the execution
    mom_ret = price.shift(lag).pct_change(lookback)
    ranks = mom_ret.rank(axis=1, ascending=False)
    demeaned = ranks.subtract(ranks.mean(axis=1), axis=0)
    return demeaned.divide(demeaned.std(axis=1), axis=0)


def strat_sr(prices, lb, hold):
    # Compute portfolio weights
    freq = '%dB' % hold
    port = calc_mom(prices, lb, lag=1)

    daily_rets = prices.pct_change()

    # `resample` merely loops over for each freq
    # Takes the first value of the resampled
    port = port.shift(1).resample(freq).first() # shift 1 for trading at the close
    returns = daily_rets.resample(freq).apply(compound)
    port_rets = (port * returns).sum(axis=1)

    return daily_sr(port_rets) * np.sqrt(252 / hold)


def some_strategy(prices, lb, hold):

    def get_portfolio_logic(price, lookback, lag):
        momentum = price.shift(lag).pct_change(lookback)
        ranks = momentum.rank(axis=1, ascending=False)
        best_ten = lambda x: list(map(lambda y: y
                                      if y > price.shape[1] - 10
                                      else 0, x))
        ranks = ranks.apply(best_ten, axis=1)
        return ranks

    compound = lambda x: (1 + x).prod() - 1
    daily_sr = lambda x: x.mean() / x.std()

    # Compute portfolio weights
    freq = '%dB' % hold
    port = get_portfolio_logic(prices, lb, lag=0)

    daily_rets = prices.pct_change()

    # `resample` merely loops over for each freq
    # Takes the first value of the resampled
    port = port.shift(1).resample(freq).first() # shift 1 for trading at the close
    returns = daily_rets.resample(freq).apply(compound)
    port_rets = (port * returns).sum(axis=1)

    return daily_sr(port_rets) * np.sqrt(252 / hold)


def heatmap(df, cmap=plt.cm.gray_r):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    axim = ax.imshow(df.values, cmap=cmap, interpolation='nearest')
    ax.set_xlabel(df.columns.name)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(list(df.columns))
    ax.set_ylabel(df.index.name)
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(list(df.index))
    plt.colorbar(axim)
    plt.show()


def example_combine_lookback_holding():
    from collections import defaultdict

    queries = '''
                select *
                from spx_table
                order by "Sector";
                '''

    ret = ppd.run_query(queries)

    by_sectors = ret.groupby('Sector')
    sectors = list(set(ret['Sector']))

    IT = by_sectors.get_group(sectors[3])
    tech_symbols = IT.Symbol.values
    tech = create_dataframes(tech_symbols)

    lookbacks = range(20, 90, 5)
    holdings = range(20, 90, 5)
    dd = defaultdict(dict) # dictionary of dictionary

    for lb in lookbacks:
        for hold in holdings:
            dd[lb][hold] = strat_sr(tech.tail(1000), lb, hold)

    ddf = pd.DataFrame(dd) # This is awesome...
    ddf.index.name = 'Holding Period'
    ddf.columns.name = 'Lookback Period'

    heatmap(ddf)


if __name__ == '__main__':

    ppd = PostgresPandas()

    rolling_example()
    example_combine_lookback_holding()


