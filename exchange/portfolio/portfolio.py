from exchange.components import ActionModule, ActionResult, DatesModule, Stock
from exchange.utils import get_tickers
from injector import Module, Injector, inject
from typing import List, NewType


class PortfolioModule(Module):
    def configure(self, binder):
        binder.multibind(List[Stock], self.__get_list_of_stocks())

    def __get_list_of_stocks(self):
        ticker_list = get_tickers()

        list_of_stocks_for_portfolio = [self.__injector__.create_object(Stock, {'ticker': ticker})
                                        for ticker in ticker_list]

        return list_of_stocks_for_portfolio


class Portfolio:
    @inject
    def __init__(self, stocks: List[Stock]):
        self.stocks = stocks

    def step(self, actions) -> List[ActionResult]:
        return [stock.step(action) for stock, action in zip(self.stocks, actions)]

    def reset(self):
        pass


if __name__ == '__main__':

    inj = Injector([ActionModule, DatesModule, PortfolioModule])

    print(inj.get(Portfolio).stocks)
