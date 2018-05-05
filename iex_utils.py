from datetime import datetime
from iexfinance import get_historical_data
from functools import reduce
from dateutil import parser

import pandas as pd
import numpy as np
import sys
import time


def create_df_from_symbols(*symbols):

    to_merge = []

    for s in symbols:
        try:
            ret = pd.read_csv('./iexdata/{}'.format(s))
            ret.columns = list(map(lambda x: x if x=='date'
                                            else s+'_'+x, ret.columns))
            to_merge.append(ret)
        except FileNotFoundError:
            print('*** File Not Found Error in {}'.format(s))
        except:
            print('*** Error in {}: {}'.format(s, sys.exc_info()[0]))


    merged_date = reduce(lambda x, y:
                         pd.merge(x, y, on='date', how='outer'), to_merge)
    merged_date.sort_values('date', inplace=True)
    merged_date.date = merged_date.date.map(lambda x: parser.parse(x))
    merged_date.set_index('date', inplace=True, drop=True)

    return merged_date


def collect_data_from_iex(start, end, *symbols):

    symbols = list(map(str.upper, symbols))
    for s in symbols:
        try:
            time.sleep(2)
            df = get_historical_data(s, start, end, 'pandas')
            df.to_csv('./iexdata/{}'.format(s))
        except:
            print('*** Error in the symbol {}: {}'.format(s, sys.exc_info()[0]))


def collect_data_for_each_sectors(start, end):

    spx_table = pd.read_csv('./iexdata/10K_data.csv')
    sectors_list = list(set(spx_table['Sector']))
    by_sectors = spx_table.groupby('Sector')

    for sector in sectors_list:
        collect_data_from_iex(start, end,
                              *(by_sectors.get_group(sector)['Symbol']))


if __name__ == '__main__':
    start = datetime(2013, 2, 9)
    end = datetime(2018, 5, 2)

    # collect_data_for_each_sectors(start, end)
    # collect_data_from_iex(start, end, *interested)
    # create_df_from_symbols(start, end, *interested)




