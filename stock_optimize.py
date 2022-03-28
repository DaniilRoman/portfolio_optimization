import pandas as pd
import random


def gen_one_individual(max_count_data):
    return [random.randint(0, max_count) for max_count in max_count_data]


def evaluate(individual, value_data, price_data, budget):
    individual = individual[0]
    profit = sum(x * y for x, y in zip(value_data, individual))
    sum_price = sum(x * y for x, y in zip(price_data, individual))

    if sum_price > budget:
        return 0, 100000000000000000000
    return profit, abs(budget - sum_price)


def analyze(best_solution, stock_table, budget):
    value_data = list(stock_table['Value'])
    price_data = list(stock_table['Price'])

    best_set = best_solution[0][0]
    print(best_set)
    cost_by_stock = [x * y for x, y in zip(price_data, best_set)]
    profit_by_stock = [x * y for x, y in zip(value_data, best_set)]

    append_df = pd.DataFrame.from_records(
        [[x, y, z] for x, y, z in zip(best_set, cost_by_stock, profit_by_stock)]
    )
    append_df.columns = ['Count', 'TotalCost', 'TotalProfit']

    summary = stock_table.join(append_df)
    profit = sum(x * y for x, y in zip(value_data, best_set))
    cost = sum(x * y for x, y in zip(price_data, best_set))

    print(summary)
    print("Future price: " + str(profit))
    print("Final cost: " + str(cost))
    print("Limit budget: " + str(budget))
    print("Profit: " + str(profit - cost))
