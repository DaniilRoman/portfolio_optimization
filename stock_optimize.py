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
    real_price_data = list(stock_table['RealPrice'])

    best_set = best_solution[0][0]
    print(best_set)
    cost_by_stock = [x * y for x, y in zip(price_data, best_set)]
    profit_by_stock = [x * y for x, y in zip(value_data, best_set)]
    real_price_by_stock = [x * y for x, y in zip(real_price_data, best_set)]

    append_df = pd.DataFrame.from_records(
        [[x, y, z, w] for x, y, z, w in zip(best_set, cost_by_stock, profit_by_stock, real_price_by_stock)]
    )
    append_df.columns = ['Count', 'TotalCost', 'TotalPrice', 'TotalRealPrice']

    summary = stock_table.join(append_df)
    future_price = sum(x * y for x, y in zip(value_data, best_set))
    cost = sum(x * y for x, y in zip(price_data, best_set))
    real_future_price = sum(x * y for x, y in zip(real_price_data, best_set))

    print(summary)
    print("Limit budget: " + str(budget))
    print("Final cost: " + str(cost))
    print("Future price: " + str(future_price))
    print("Real future price: " + str(real_future_price))
    print("Profit: " + str(future_price - cost) + " / " + str(round((future_price - cost) / cost * 100, 2)) + "%")
    print("Real profit: " + str(real_future_price - cost) + " / " + str(round((real_future_price - cost) / cost * 100, 2)) + "%")
