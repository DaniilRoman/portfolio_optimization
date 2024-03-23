from data.data import StockOptimizationJob, StockLimitType
from repository.optimization_job_repo import OptimizationRepository


def transform_stock_limit(job: StockOptimizationJob, repo: OptimizationRepository):
    stock_limits = []
    stock_limit = job.stock_limit
    if stock_limit.type == StockLimitType.COUNT and stock_limit.common_limit is None:
        stock_limits = stock_limit.limits
    elif stock_limit.type == StockLimitType.COUNT and stock_limit.common_limit is not None:
        stock_limits = [stock_limit.common_limit for i in range(len(job.stock_names))]
    elif stock_limit.type == StockLimitType.PRICE and stock_limit.common_limit is None:
        stock_limits = [int(total_price / price) for total_price, price in zip(stock_limit.limits, job.current_prices)]
    elif stock_limit.type == StockLimitType.PRICE and stock_limit.common_limit is not None:
        stock_limits = [int(stock_limit.common_limit / price) for price in job.current_prices]

    elif stock_limit.type == StockLimitType.PERCENT and stock_limit.common_limit is None:
        stock_limits = [int((job.budget * percent / 100) / price) for percent, price in zip(stock_limit.limits, job.current_prices)]
    elif stock_limit.type == StockLimitType.PERCENT and stock_limit.common_limit is not None:
        total_price = (job.budget * stock_limit.common_limit / 100)
        stock_limits = [int(total_price / price) for price in job.current_prices]

    job.max_stock_count_list = stock_limits
    repo.save_job(job)
