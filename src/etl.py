import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from typing import List

from tqdm import tqdm

from repository.optimization_job_repo import OptimizationRepository
from services.analyze import construct_result, print_result
from data.data import StockOptimizationJob, OptimizationJobStatus, StockLimit, PriceData
from services.stock_data_downloader import construct_price_data, download_current_price
from services.stock_limit_transformer import transform_stock_limit
from adapter.out.optimization.stock_optimize import optimize
from utils.utils import get_prev_day, OnlyPutBlockingQueue, round_precise, get_current_date_str


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
        res = construct_price_data(src, self.predict_days, self.repo, self.is_backtest)
        self.tmp_dict.put(src, res)


def filter_negative_stock(stock_price: PriceData) -> bool:
    if stock_price.predict_price < stock_price.current_price:
        print(f"Filter: {stock_price}")
        return False
    else:
        return True


def download_and_predict_step(job: StockOptimizationJob, repo: OptimizationRepository, parallelism: int = 50):
    print("STEP: download and predict")
    job.status = OptimizationJobStatus.PREPARING_DATA
    repo.save_job(job)

    task_object = PriceGetter(job.predict_period_days, repo, job.is_backtest)

    with ThreadPoolExecutor(parallelism) as executor:
        stock_price_futures = [executor.submit(task_object, stock_name) for stock_name in tqdm(job.stock_names)]
        for future in as_completed(stock_price_futures):
            # url = future_to_url[future]
            future.result()

    prices = [task_object.tmp_dict.queue[i] for i in job.stock_names]
    prices = list(filter(filter_negative_stock, prices))

    job.current_prices = [prices_pair.current_price for prices_pair in prices]
    job.predicted_prices = [prices_pair.predict_price for prices_pair in prices]
    job.real_prices = [prices_pair.real_future_price for prices_pair in prices]

    filtered_stock_names = [prices_pair.stock_name for prices_pair in prices]

    def filter_max_stock_count(stock_name, stock_count) -> bool:
        if stock_name in filtered_stock_names:
            return True
        else:
            return False

    job.max_stock_count_list = [y for x, y in
                                list(filter(filter_max_stock_count, zip(job.stock_names, job.max_stock_count_list)))]
    job.stock_names = filtered_stock_names


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
        index_name = "IMOEX.ME"  # "SPY"
        finish_sp500_price = download_current_price(index_name, repo)
        start_sp500_price = download_current_price(index_name, repo, get_prev_day(job.predict_period_days))
        print("S&P500 start price: " + str(round(start_sp500_price, round_precise)))
        print("S&P500 finish price: " + str(round(finish_sp500_price, round_precise)))
        print("S&P500 price change: " + str(round(finish_sp500_price - start_sp500_price, round_precise)))
        print(
            "S&P500 price change: " + str(
                round((finish_sp500_price - start_sp500_price) / start_sp500_price * 100, round_precise)) + "%")
        print(f"From {get_prev_day(job.predict_period_days)} to {get_current_date_str()}")
        print(f"Stock limit: {job.stock_limit}")

    return opt_result


def stock_limit_transform_step(job: StockOptimizationJob, repo: OptimizationRepository):
    print("STEP: Stock limit transformation")
    transform_stock_limit(job, repo)
    repo.save_job(job)


def filter_zero_limit_step(job: StockOptimizationJob, repo: OptimizationRepository):
    print("STEP: Stock limit filtering")
    filter_indices = set()
    for i, val in enumerate(job.max_stock_count_list):
        if val == 0:
            filter_indices.add(i)
    for i, val in enumerate(job.current_prices):
        if val == 0.0 or math.isnan(val):
            filter_indices.add(i)

    def except_filter_indices(list_values):
        return [x for i, x in enumerate(list_values) if i not in filter_indices]

    job.max_stock_count_list = except_filter_indices(job.max_stock_count_list)
    job.stock_names = except_filter_indices(job.stock_names)
    job.current_prices = except_filter_indices(job.current_prices)
    job.best_set = except_filter_indices(job.best_set)
    job.real_prices = except_filter_indices(job.real_prices)
    job.predicted_prices = except_filter_indices(job.predicted_prices)
    job.max_stock_count_list = job.max_stock_count_list[:len(job.stock_names)] # TODO

    repo.save_job(job)


def run_etl(stock_names: List[str], stock_limit: StockLimit, budget: int, predict_period_days: int,
            repo: OptimizationRepository, parallelism: int, is_backtest: bool):
    job = create_optimization_task_step(stock_names, stock_limit, budget, predict_period_days, is_backtest, repo)
    print(f"Job `{job._id}` created")
    run_etl_internal(job, repo, parallelism)
    return job._id


def run_etl_async(stock_names: List[str], stock_limit: StockLimit, budget: int, predict_period_days: int,
                  repo: OptimizationRepository, parallelism: int, is_backtest: bool, executor: ThreadPoolExecutor):
    job = create_optimization_task_step(stock_names, stock_limit, budget, predict_period_days, is_backtest, repo)
    print(f"Job `{job._id}` created")
    executor.submit(lambda: run_etl_internal(job, repo, parallelism))
    return job._id


def run_etl_internal(job: StockOptimizationJob, repo: OptimizationRepository, parallelism: int):
    try:
        download_and_predict_step(job, repo, parallelism)
        filter_zero_limit_step(job, repo)
        stock_limit_transform_step(job, repo)
        filter_zero_limit_step(job, repo)
        optimization_step(job, repo)
        construct_result_step(job, repo)
    except Exception as e:
        print(e)

