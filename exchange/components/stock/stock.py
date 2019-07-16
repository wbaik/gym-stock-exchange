from exchange.components.action_space import ActionSpace, ActionModule
from exchange.components.dates_trading import DatesTrading, DatesModule
from injector import inject, Injector, Module
from typing import NewType


Ticker = NewType('ticker', str)


class TickerModule(Module):
    def configure(self, binder):
        binder.bind(Ticker, to='AAPL')


class Stock:
    @inject
    def __init__(self,
                 ticker: Ticker,
                 action_space: ActionSpace,
                 dates: DatesTrading):
        self.ticker = ticker
        self.action_space = action_space
        self.dates = dates


if __name__ == '__main__':

    inj = Injector([TickerModule, ActionModule, DatesModule])
    aapl = inj.create_object(Stock)

    print(aapl.ticker)
    print(aapl.action_space.min_action)
    print(aapl.action_space.max_action)
    print(aapl.dates.start_date)

