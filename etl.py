from typing import List

from optimization_job_repo import OptimizationResult
from services.analyze import construct_result, print_result
from data import StockOptimizationJob, OptimizationJobStatus
from services.stock_data_downloader import get_current_and_predict_prices
from services.stock_optimize import optimize


def create_optimization_task_step(stock_names: List[str], max_stock_count_list: List[int], budget: int,
                                  predict_period_days: int, is_backtest: bool) -> StockOptimizationJob:
    job = StockOptimizationJob(stock_names, max_stock_count_list, budget, predict_period_days, is_backtest)
    return job


def download_and_predict_step(job: StockOptimizationJob):
    job.status = OptimizationJobStatus.PREPARING_DATA

    current_predict_prices = [get_current_and_predict_prices(stock_name, job.predict_period_days, job.is_backtest) for stock_name in
                              job.stock_names]
    current_prices = [prices_pair[0] for prices_pair in current_predict_prices]
    predict_prices = [prices_pair[1] for prices_pair in current_predict_prices]
    real_future_prices = [prices_pair[2] for prices_pair in current_predict_prices]

    job.current_prices = current_prices
    job.predicted_prices = predict_prices
    job.real_prices = real_future_prices


def optimization_step(job: StockOptimizationJob):
    job.status = OptimizationJobStatus.OPTIMIZATION

    best_set = optimize(job)
    job.best_set = best_set
    job.status = OptimizationJobStatus.FINISHED


def construct_result_step(job: StockOptimizationJob):
    opt_result = construct_result(job)
    print(job)
    print("=================================================================")
    print_result(opt_result)
    return opt_result


def run_etl(stock_names: List[str], max_stock_count_list: List[int], budget: int, predict_period_days: int):
    job = create_optimization_task_step(stock_names, max_stock_count_list, budget, predict_period_days, True)
    print(f"Job `{job._id}` created")
    download_and_predict_step(job)
    optimization_step(job)
    construct_result_step(job)


if __name__ == '__main__':
    # stock_names = ["DIDI", "MSFT", "AMD", "BABA", "NVDA", "AAL"]
    stock_names = ["DIDI", "MSFT"]
    max_stock_count_list = [10 for i in range(len(stock_names))]
    BUDGET = 2000
    PREDICT_PERIOD_DAYS = 30

    run_etl(stock_names, max_stock_count_list, BUDGET, PREDICT_PERIOD_DAYS)
