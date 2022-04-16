from stock_data_downloader import construct_data_for_optimization, construct_data_for_backtest
from stock_optimize import analyze, evaluate, gen_one_individual
from utils import create_toolbox, optimize

stock_names = ["DIDI", "MSFT", "AMD", "BABA", "NVDA", "AAL"]
max_stock_count_list = [10, 10, 10, 10, 10, 10]
BUDGET = 2000
PREDICT_PERIOD_DAYS = 30

data = construct_data_for_backtest(stock_names, max_stock_count_list, PREDICT_PERIOD_DAYS)
VALUE_DATA = list(data['Value'])
PRICE_DATA = list(data['Price'])
MAX_COUNT_DATA = list(data['MaxCount'])
FUN_WEIGHTS = {"profit_func": 1.0, "cost_func": -1.0}


def evaluate_func_wrapper(i):
    return evaluate(i, VALUE_DATA, PRICE_DATA, BUDGET)


def gen_one_individual_wrapper():
    return gen_one_individual(MAX_COUNT_DATA)


toolbox = create_toolbox(evaluate_func_wrapper, gen_one_individual_wrapper, tuple(FUN_WEIGHTS.values()))
best_solution = optimize(toolbox)

analyze(best_solution, data, BUDGET)
