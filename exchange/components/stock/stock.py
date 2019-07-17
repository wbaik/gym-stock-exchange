import yaml

from exchange.components.action_space import ActionSpace, ActionModule
from exchange.components.dates_trading import DatesTrading, DatesModule
from exchange.utils import YAML_FILE_PATH
from injector import inject, Injector, Module


class Stock:
    @inject
    def __init__(self,
                 ticker: str,
                 action_space: ActionSpace,
                 dates: DatesTrading):
        self.ticker = ticker
        self.action_space = action_space
        self.dates = dates
        self.position = 0.0

    def __repr__(self) -> str:
        return self.ticker


if __name__ == '__main__':
    # THIS LOGIC IS USED IN PORTFOLIO
    inj = Injector([ActionModule, DatesModule])

    ticker_list = yaml.load(open(YAML_FILE_PATH, 'r'), Loader=yaml.Loader)['ticker']
    assert ticker_list, "Please add tickers into {}".format(YAML_FILE_PATH)

    list_of_stocks_for_portfolio = [inj.create_object(Stock, {'ticker': ticker})
                                    for ticker in ticker_list]

    print(list_of_stocks_for_portfolio)

