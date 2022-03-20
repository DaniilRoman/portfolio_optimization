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



# products_table['univariate_choice'] = pd.Series(best_solution[0])
# print(products_table.head())
#
# products_table['univariate_gr_prot'] = products_table['univariate_choice'] * products_table['Gram_Prot']
# products_table['univariate_gr_fat'] = products_table['univariate_choice'] * products_table['Gram_Fat']
# products_table['univariate_gr_carb'] = products_table['univariate_choice'] * products_table['Gram_Carb']
# products_table['univariate_cal'] = products_table['univariate_choice'] * products_table['Calories']

summary = pd.DataFrame.from_records(
    [
        [products_table['univariate_gr_prot'].sum(), gram_prot],
        [products_table['univariate_gr_carb'].sum(), gram_carb],
        [products_table['univariate_gr_fat'].sum(), gram_fat],
        [products_table['univariate_cal'].sum(), sum((cal_prot, cal_fat, cal_carb))]
    ])
summary.columns = ['univariate', 'goal']
summary.index = ['prot', 'carb', 'fat', 'cal']

print(summary)
