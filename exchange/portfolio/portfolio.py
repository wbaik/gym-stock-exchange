from collections import Iterable

from exchange.components.stock import Ticker


class Portfolio:

    def __init__(self, config) -> None:
        super().__init__()
        self._portfolio = self._get_a_portfolio(config)

    def step(self, action):

    def reset(self):
        pass

    def _get_a_portfolio(self, config) -> Iterable:
        # return [Ticker(ticker) for ticker in config.TICKERS]
        raise NotImplementedError

