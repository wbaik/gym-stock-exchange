from gym.envs.registration import register

register(
    id='game-stock-exchange-v0',
    entry_point='gym_stock_exchange.envs:StockExchange',
)
