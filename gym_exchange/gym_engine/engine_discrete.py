from gym_exchange.gym_engine import TickerDiscrete
from gym_exchange.gym_engine.engine_base import EngineBase


class EngineDiscrete(EngineBase):
    def __init__(self,
                 tickers,
                 start_date,
                 num_days_iter,
                 today=None,
                 seed=None,
                 num_action_space=3,
                 render=False,
                 *args,
                 **kwargs):

        self.tickers = self._get_tickers(tickers, start_date, num_days_iter,
                                         today, num_action_space, *args, **kwargs)
        super().__init__(tickers, seed, render, *args, **kwargs)

    def _get_tickers(self, tickers, start_date, num_days_iter,
                     today, num_action_space, *args, **kwargs):
        return [TickerDiscrete(ticker, start_date, num_days_iter, today, num_action_space, *args, **kwargs)
                for ticker in tickers]
