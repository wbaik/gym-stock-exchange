from unittest import TestCase

from exchange.exchange import Exchange
from config import BaseConfig


class TestExchange(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.exchange = Exchange()

    def test_step(self):
        self.fail()

    def test_reset(self):
        self.fail()

    def test_render(self):
        self.fail()

    def test_close(self):
        self.fail()