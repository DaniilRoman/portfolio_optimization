import pandas as pd
import random


from utils import create_toolbox
from utils import optimize


BUDGET = 50


products_table = pd.DataFrame.from_records([
    ['Banana 1u', 23, 1, 40],
    ['Mandarin 1u', 10, 2, 3],
    ['Ananas 100g', 13, 3, 2],
    ['Grapes 100g', 17, 4, 1],
    ['Chocolate 1 bar', 25, 1, 8]
])
products_table.columns = ['Name', 'Value', 'Price', 'MaxCount']

VALUE_DATA = list(products_table['Value'])
PRICE_DATA = list(products_table['Price'])
MAX_COUNT_DATA = list(products_table['MaxCount'])

FUN_WEIGHTS = {"profit_func": 1.0, "cost_func": -1.0}


def gen_one_individual():
    return [random.randint(0, max_count) for max_count in MAX_COUNT_DATA]


def evaluate(individual):
    individual = individual[0]
    profit = sum(x * y for x, y in zip(VALUE_DATA, individual))
    sum_price = sum(x * y for x, y in zip(PRICE_DATA, individual))

    return profit, abs(BUDGET - sum_price)


toolbox = create_toolbox(evaluate, gen_one_individual, tuple(FUN_WEIGHTS.values()))
best_solution = optimize(toolbox)

print(best_solution)



best_set = best_solution[0][0]
print(best_set)
cost_by_stock = [x * y for x, y in zip(PRICE_DATA, best_set)]
profit_by_srock = [x * y for x, y in zip(VALUE_DATA, best_set)]

append_df = pd.DataFrame.from_records(
    [[x, y, z] for x, y, z in zip(best_set, cost_by_stock, profit_by_srock)]
)
append_df.columns = ['Count', 'TotalCost', 'TotalProfit']

summary = products_table.join(append_df)
profit = sum(x * y for x, y in zip(VALUE_DATA, best_set))
cost = sum(x * y for x, y in zip(PRICE_DATA, best_set))

print(summary)
print("Profit: " + str(profit))
print("Final cost: " + str(cost))
print("Limit budget: " + str(BUDGET))
