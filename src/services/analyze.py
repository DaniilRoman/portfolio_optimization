import pandas as pd

from data import StockOptimizationJob, OptimizationResult


def print_result(result: OptimizationResult):
    append_df = pd.DataFrame.from_records(
        [[x, y, z, w, q] for x, y, z, w, q in
         zip(result.best_set, result.max_stock_count_list, result.by_stock_cost, result.by_stock_predicted_cost, result.by_stock_real_cost)]
    )
    append_df.columns = ['Count', 'LimitCount', 'TotalCost', 'TotalPredictedCost', 'TotalRealCost']

    summary_df = pd.DataFrame.from_records(
        [[x, y, z, w] for x, y, z, w in zip(result.stock_names, result.current_prices, result.predicted_prices, result.real_prices)]
    )
    summary_df.columns = ['Name', 'CurrentPrice', 'PredictedPrice', 'RealPrice']

    summary = summary_df.join(append_df)
    print(summary.to_string())
    print("Limit budget: " + str(result.budget))
    print("Final cost: " + str(result.total_cost))
    print("Future price: " + str(result.total_predicted_cost))
    print("Real future price: " + str(result.total_real_cost))
    print("Profit: " + str(result.profit) + " / " + str(result.profit_percent) + "%")
    print("Real profit: " + str(result.real_profit) + " / " + str(result.real_profit_percent) + "%")


def construct_result(job: StockOptimizationJob) -> OptimizationResult:
    by_stock_cost = [round(x * y, 2) for x, y in zip(job.current_prices, job.best_set)]
    by_stock_predicted_cost = [round(x * y, 2) for x, y in zip(job.predicted_prices, job.best_set)]
    by_stock_real_cost = [round(x * y, 2) for x, y in zip(job.real_prices, job.best_set)]

    total_predicted_cost = round(sum(x * y for x, y in zip(job.predicted_prices, job.best_set)), 2)
    total_cost = round(sum(x * y for x, y in zip(job.current_prices, job.best_set)), 2)
    total_real_cost = round(sum(x * y for x, y in zip(job.real_prices, job.best_set)), 2)

    profit = round(total_predicted_cost - total_cost, 2)
    profit_percent = round((total_predicted_cost - total_cost) / total_cost * 100, 2)
    real_profit = total_real_cost - total_cost
    real_profit_percent = round((total_real_cost - total_cost) / total_cost * 100, 2)

    real_prices = job.real_prices
    if not job.is_backtest:
        real_prices = [0 for p in job.current_prices]
        by_stock_real_cost = [0 for p in job.current_prices]
        total_real_cost = 0
        real_profit = 0
        real_profit_percent = 0

    result = OptimizationResult(job._id, job.stock_names, job.max_stock_count_list, job.budget, job.predict_period_days,
                                total_cost, total_predicted_cost, total_real_cost,
                                profit, profit_percent, real_profit, real_profit_percent,
                                job.current_prices, job.predicted_prices, real_prices, job.best_set,
                                by_stock_cost, by_stock_predicted_cost, by_stock_real_cost)

    return result
