from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


class ActionConfig:

    def __init__(self, min_action: float, max_action: float) -> object:
        self.min_action = min_action
        self.max_action = max_action


class DiscreteActionConfig(ActionConfig):

    def __init__(self, min_action: float, max_action: float, no_action: float = 0.0) -> object:
        super().__init__(min_action, max_action)
        self.action_space = [self.min_action, no_action, self.max_action]


class ContinuousActionConfig(ActionConfig):

    def __init__(self, min_action: float, max_action: float) -> object:
        super().__init__(min_action, max_action)
        self.action_space = [self.min_action, self.max_action]


@dataclass
class DatesConfig:
    dates_trading: Iterable[datetime]


@dataclass
class TickerConfig:
    tickers: Iterable[str]


@dataclass
class Config(ActionConfig, DatesConfig, TickerConfig):
    pass


if __name__ == '__main__':
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2019, 3, 1)
    d = DatesConfig([start_date, end_date])
    print(d.dates_trading)
