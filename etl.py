import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

import pandas as pd
from typing import List

from optimization_job_repo import OptimizationRepository
from services.analyze import construct_result, print_result
from data import StockOptimizationJob, OptimizationJobStatus, StockLimit, StockLimitType
from services.stock_data_downloader import get_current_and_predict_prices, download_current_price
from services.stock_limit_transformer import transform_stock_limit
from services.stock_optimize import optimize
from utils import get_prev_day, OnlyPutBlockingQueue


def get_sp500_stocks():
    # There are 2 tables on the Wikipedia page
    # we want the first table

    payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    first_table = payload[0]
    second_table = payload[1]

    df = first_table

    symbols = df['Symbol'].values.tolist()
    excluded_stocks = ["BRK.B", "BF.B", "MMM", "AES", "AFL", "A", "ABT", "ADBE"]
    return list(filter(lambda s: s not in excluded_stocks, symbols))


def create_optimization_task_step(stock_names: List[str], stock_limit: StockLimit, budget: int,
                                  predict_period_days: int, is_backtest: bool,
                                  repo: OptimizationRepository) -> StockOptimizationJob:
    job = StockOptimizationJob(stock_names, stock_limit, budget, predict_period_days, is_backtest)
    repo.save_job(job)
    return job


class PriceGetter(object):
    def __init__(self, predict_days, repo, is_backtest):
        self.tmp_dict = OnlyPutBlockingQueue()
        self.predict_days = predict_days
        self.repo = repo
        self.is_backtest = is_backtest

    def __call__(self, src):
        res = get_current_and_predict_prices(src, self.predict_days, self.repo, self.is_backtest)
        self.tmp_dict.put(src, res)


def download_and_predict_step(job: StockOptimizationJob, repo: OptimizationRepository, parallelism: int = 50):
    print("STEP: download and predict")
    job.status = OptimizationJobStatus.PREPARING_DATA
    repo.save_job(job)

    task_object = PriceGetter(job.predict_period_days, repo, job.is_backtest)
    with ThreadPoolExecutor(parallelism) as executor:
        executor.map(task_object, tqdm(job.stock_names))
    current_predict_prices = [task_object.tmp_dict.queue[i] for i in job.stock_names]

    current_prices = [prices_pair.current_price for prices_pair in current_predict_prices]
    predict_prices = [prices_pair.predict_price for prices_pair in current_predict_prices]
    real_future_prices = [prices_pair.real_future_price for prices_pair in current_predict_prices]

    job.current_prices = current_prices
    job.predicted_prices = predict_prices
    job.real_prices = real_future_prices


def optimization_step(job: StockOptimizationJob, repo: OptimizationRepository):
    print("STEP: Optimization")
    job.status = OptimizationJobStatus.OPTIMIZATION
    repo.save_job(job)

    best_set = optimize(job)
    job.best_set = best_set
    job.status = OptimizationJobStatus.FINISHED
    repo.save_job(job)


def construct_result_step(job: StockOptimizationJob, repo: OptimizationRepository):
    print("STEP: Construct result")
    opt_result = construct_result(job)
    repo.save_opt_result(opt_result)
    print_result(opt_result)

    if job.is_backtest:
        finish_sp500_price = download_current_price("SPY", repo)
        start_sp500_price = download_current_price("SPY", repo, get_prev_day(job.predict_period_days))
        print("S&P500 price: " + str(round(finish_sp500_price, 5)))
        print("S&P500 price change: " + str(round(finish_sp500_price - start_sp500_price, 5)))
        print(
            "S&P500 price change: " + str(
                round((finish_sp500_price - start_sp500_price) / start_sp500_price * 100, 5)) + "%")

    return opt_result


def stock_limit_transform_step(job: StockOptimizationJob, repo: OptimizationRepository):
    print("STEP: Stock limit transformation")
    transform_stock_limit(job, repo)
    repo.save_job(job)


def run_etl(stock_names: List[str], stock_limit: StockLimit, budget: int, predict_period_days: int,
            repo: OptimizationRepository, parallelism: int):
    job = create_optimization_task_step(stock_names, stock_limit, budget, predict_period_days, True, repo)
    print(f"Job `{job._id}` created")
    download_and_predict_step(job, repo, parallelism)
    stock_limit_transform_step(job, repo)
    optimization_step(job, repo)
    construct_result_step(job, repo)

if __name__ == '__main__':
    start = time.time()
    # stock_names = ["DIDI", "MSFT", "AMD", "BABA", "NVDA", "AAL"]
    # stock_names = ["DIDI", "MSFT"]
    stock_names = get_sp500_stocks()
    stock_limit = StockLimit(StockLimitType.PERCENT, common_limit=30)
    BUDGET = 20000
    PREDICT_PERIOD_DAYS = 30

    repo = OptimizationRepository()
    run_etl(stock_names, stock_limit, BUDGET, PREDICT_PERIOD_DAYS, repo, parallelism=100)
    print(f"Time: {time.time()-start}")


