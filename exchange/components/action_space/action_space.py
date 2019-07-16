from injector import inject, Module
from typing import NewType


MinActionValue = NewType('min_action_value', float)
MaxActionValue = NewType('max_action_value', float)


class ActionModule(Module):
    def configure(self, binder):
        binder.bind(MinActionValue, to=-1.0)
        binder.bind(MaxActionValue, to=1.0)


class ActionSpace:
    @inject
    def __init__(self,
                 min_action: MinActionValue,
                 max_action: MaxActionValue):
        self.min_action = min_action
        self.max_action = max_action
