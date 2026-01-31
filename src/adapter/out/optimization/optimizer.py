import requests
import configuration
from deap import base, creator, tools, algorithms
from deap.base import Toolbox
from typing import List, Dict, Tuple, Any
import random

# CXPB  is the probability with which two individuals are crossed
# MUTPB is the probability for mutating an individual
from src.logic.data.data import StockData

CXPB, MUTPB, NUMBER_OF_ITERATIONS, NUMBER_OF_POPULATION = 0.3, 0.7, 100, 70 # 200
FUN_WEIGHTS = {"cost_func": -1.0, "profit_func": 1.0}


def __gen_one_individual(max_count_data):
    return [random.randint(0, max_count) for max_count in max_count_data]


def __evaluate(individual, predicted_prices, prices, budget):
    predicted_cost = sum(x * y for x, y in zip(predicted_prices, individual))
    cost = sum(x * y for x, y in zip(prices, individual))

    if cost > budget:
        return 100000000000, -10000000000
    return abs(budget - cost), predicted_cost - cost

def __create_toolbox(eval_func, weights: tuple, mutFlipBit) -> Toolbox:
    # Check if creator classes already exist to avoid warnings
    if not hasattr(creator, "FitnessFunc"):
        creator.create("FitnessFunc", base.Fitness, weights=weights)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessFunc)

    toolbox = Toolbox()
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", mutFlipBit, indpb=0.4)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def __optimize_internal(toolbox, gen_individual_func):
    pop = [creator.Individual(gen_individual_func()) for i in range(NUMBER_OF_POPULATION)]
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    return algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NUMBER_OF_ITERATIONS)


def optimize(stocks: List[StockData], budget: float = 50.0, max_per_etf_budget: float = 150.0) -> List[Tuple[str, int, float, float]]:
    """
    Optimize portfolio to suggest what ETFs to buy next.
    
    Args:
        stocks: List of StockData objects containing current and predicted prices
        budget: Ideal budget to spend (default 50 EUR)
        max_per_etf_budget: Maximum to spend on a single ETF if expensive (default 150 EUR)
        
    Returns:
        List of tuples (ticker_symbol, shares_to_buy, total_cost, expected_profit)
    """
    # Extract prices and tickers
    tickers = [stock.ticker_symbol for stock in stocks]
    current_prices = [stock.current_price for stock in stocks]
    predicted_prices = [stock.predict_price for stock in stocks]
    
    # Get current ETF ownership
    etf_map = __get_etf_map()
    
    # Calculate max shares based on budget constraints
    # For each ETF, we can spend up to max_per_etf_budget, but prefer to stay near budget
    max_shares_per_stock = []
    for price in current_prices:
        if price <= 0:
            max_shares = 0
        else:
            # Calculate maximum shares based on max_per_etf_budget
            max_by_budget = int(max_per_etf_budget / price)
            # Also consider reasonable upper limit
            max_shares = min(max_by_budget, 100)  # Cap at 100 shares max
        max_shares_per_stock.append(max_shares)
    
    # Adjust fitness based on current ownership - prefer ETFs with lower current ownership
    ownership_weights = []
    for ticker in tickers:
        current_count = etf_map.get(ticker, 0)
        # Lower weight for ETFs we already own many of (encourage diversification)
        # Weight = 1 / (1 + current_count) - more ownership = lower weight
        weight = 1.0 / (1.0 + current_count)
        ownership_weights.append(weight)
    
    def evaluate_func_wrapper(individual):
        # Calculate cost and predicted profit
        cost = sum(x * y for x, y in zip(current_prices, individual))
        predicted_profit = sum(x * y * w for x, y, w in zip(predicted_prices, individual, ownership_weights))
        
        # Penalize if cost exceeds budget
        if cost > budget:
            # Heavy penalty for exceeding budget
            return 100000000000, -10000000000
        
        # Objective 1: Minimize deviation from budget (prefer spending close to budget)
        # Objective 2: Maximize weighted predicted profit
        budget_deviation = abs(budget - cost)
        return budget_deviation, predicted_profit
    
    def gen_one_individual_wrapper():
        return __gen_one_individual(max_shares_per_stock)
    
    def mutFlipBit(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                # Flip within bounds of max_shares_per_stock
                individual[i] = max_shares_per_stock[i] - individual[i]
        return individual,
    
    toolbox = __create_toolbox(evaluate_func_wrapper, tuple(FUN_WEIGHTS.values()), mutFlipBit)
    best_solution = __optimize_internal(toolbox, gen_one_individual_wrapper)
    best_individual = tools.selBest(best_solution[0], 1)[0]
    
    # Format results: only include ETFs with positive share count
    results = []
    for i, (ticker, shares) in enumerate(zip(tickers, best_individual)):
        if shares > 0:
            cost = shares * current_prices[i]
            expected_profit = shares * (predicted_prices[i] - current_prices[i])
            results.append((ticker, shares, cost, expected_profit))
    
    # Sort by expected profit descending
    results.sort(key=lambda x: x[3], reverse=True)
    return results

def __get_etf_map():
    etf_to_count = requests.get(configuration.GET_AND_INCREMENT_COUNTER_URL, params={"etf": "true"})    
    return etf_to_count.json()
  
