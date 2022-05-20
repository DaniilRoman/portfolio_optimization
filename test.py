import time

from etl import run_etl
from optimization_job_repo import OptimizationRepository
from data import StockLimit, StockLimitType
from stock_names import sp500, russian_stocks

if __name__ == '__main__':
    start = time.time()
    # stock_names = ["DIDI", "MSFT", "AMD", "BABA", "NVDA", "AAL"]
    # stock_names = sp500
    stock_names = russian_stocks[:10]
    # stock_names = get_sp500_stocks()
    stock_names = list(
        filter(lambda x: x not in ["BRK.B", "BF.B", "MMM", "AES", "AFL", "A", "ABT", "ADBE", "RENI.ME", "SLEN.ME", "MDMG.ME"], stock_names[:50]))
    stock_limit = StockLimit(StockLimitType.PERCENT, common_limit=30)
    BUDGET = 100000
    PREDICT_PERIOD_DAYS = 30

    repo = OptimizationRepository()
    run_etl(stock_names, stock_limit, BUDGET, PREDICT_PERIOD_DAYS, repo, parallelism=50, is_backtest=False)
    print(f"Time: {time.time() - start}")