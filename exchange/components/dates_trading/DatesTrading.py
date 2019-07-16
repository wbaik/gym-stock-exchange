from datetime import datetime
from injector import inject, Module
from typing import NewType

StartDate = NewType('start_date', datetime)
EndDate = NewType('end_date', datetime)


class DatesModule(Module):
    def configure(self, binder):
        binder.bind(StartDate, to=datetime(2000, 1, 1))
        binder.bind(EndDate, to=datetime(2000, 1, 10))


class DatesTrading:
    @inject
    def __init__(self,
                 start_date: StartDate,
                 end_date: EndDate):
        self.start_date = start_date
        self.end_date = end_date
