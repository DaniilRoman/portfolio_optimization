import time

from logic import logic
from logic.data.data import StockLimit, StockLimitType
from stock_names import sp500, russian_stocks

if __name__ == '__main__':
    stock_names = russian_stocks[:10]
    # stock_names = get_sp500_stocks()
    stock_names = list(
        filter(lambda x: x not in ["BRK.B", "BF.B", "MMM", "AES", "AFL", "A", "ABT", "ADBE", "RENI.ME", "SLEN.ME", "MDMG.ME"], stock_names[:50]))
    stock_limit = StockLimit(StockLimitType.PERCENT, common_limit=30)

    logic.run(
        stock_names, 
        stock_limit, 
        budget=100000, 
        predict_period_days=30,
        parallelism=50, 
        is_backtest=False)