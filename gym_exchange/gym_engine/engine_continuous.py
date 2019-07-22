from gym_exchange.gym_engine import TickerContinuous
from gym_exchange.gym_engine.engine_base import EngineBase


class EngineContinuous(EngineBase):
    def __init__(self,
                 tickers,
                 start_date,
                 num_days_iter,
                 today=None,
                 seed=None,
                 render=False,
                 *args,
                 **kwargs):

        self.tickers = self._get_tickers(tickers, start_date, num_days_iter, today, *args, **kwargs)
        super().__init__(tickers, seed, render, *args, **kwargs)

    def _get_tickers(self,
                     tickers,
                     start_date,
                     num_days_iter,
                     today,
                     *args,
                     **kwargs):
        return [TickerContinuous(ticker, start_date, num_days_iter, today, *args, **kwargs)
                for ticker in tickers]
