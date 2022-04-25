from typing import List

from optimization_job_repo import OptimizationRepository
from services.analyze import construct_result, print_result
from data import StockOptimizationJob, OptimizationJobStatus, StockLimit, StockLimitType
from services.stock_data_downloader import get_current_and_predict_prices
from services.stock_limit_transformer import transform_stock_limit
from services.stock_optimize import optimize


def create_optimization_task_step(stock_names: List[str], stock_limit: StockLimit, budget: int,
                                  predict_period_days: int, is_backtest: bool,
                                  repo: OptimizationRepository) -> StockOptimizationJob:
    job = StockOptimizationJob(stock_names, stock_limit, budget, predict_period_days, is_backtest)
    repo.save_job(job)
    return job


def download_and_predict_step(job: StockOptimizationJob, repo: OptimizationRepository):
    job.status = OptimizationJobStatus.PREPARING_DATA
    repo.save_job(job)

    current_predict_prices = [get_current_and_predict_prices(stock_name, job.predict_period_days, repo, job.is_backtest)
                              for stock_name in
                              job.stock_names]
    current_prices = [prices_pair[0] for prices_pair in current_predict_prices]
    predict_prices = [prices_pair[1] for prices_pair in current_predict_prices]
    real_future_prices = [prices_pair[2] for prices_pair in current_predict_prices]

    job.current_prices = current_prices
    job.predicted_prices = predict_prices
    job.real_prices = real_future_prices


def optimization_step(job: StockOptimizationJob, repo: OptimizationRepository):
    job.status = OptimizationJobStatus.OPTIMIZATION
    repo.save_job(job)

    best_set = optimize(job)
    job.best_set = best_set
    job.status = OptimizationJobStatus.FINISHED
    repo.save_job(job)


def construct_result_step(job: StockOptimizationJob, repo: OptimizationRepository):
    opt_result = construct_result(job)
    repo.save_opt_result(opt_result)
    print_result(opt_result)
    return opt_result


def stock_limit_transform_step(job: StockOptimizationJob, repo: OptimizationRepository):
    transform_stock_limit(job, repo)
    repo.save_job(job)


def run_etl(stock_names: List[str], stock_limit: StockLimit, budget: int, predict_period_days: int,
            repo: OptimizationRepository):
    job = create_optimization_task_step(stock_names, stock_limit, budget, predict_period_days, True, repo)
    print(f"Job `{job._id}` created")
    download_and_predict_step(job, repo)
    stock_limit_transform_step(job, repo)
    optimization_step(job, repo)
    construct_result_step(job, repo)


if __name__ == '__main__':
    # stock_names = ["DIDI", "MSFT", "AMD", "BABA", "NVDA", "AAL"]
    stock_names = ["DIDI", "MSFT"]
    stock_limit = StockLimit(StockLimitType.PERCENT, common_limit=30)
    BUDGET = 2000
    PREDICT_PERIOD_DAYS = 30

    repo = OptimizationRepository()
    run_etl(stock_names, stock_limit, BUDGET, PREDICT_PERIOD_DAYS, repo)

