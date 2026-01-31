import requests
import config.configuration as configuration
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


def optimize(stocks: List[StockData], budget: float = 50.0, max_per_etf_budget: float = 50.0) -> str:
    """
    Optimize portfolio to suggest what ETFs to buy next.
    
    Args:
        stocks: List of StockData objects containing current and predicted prices
        budget: Ideal budget to spend (default 50 EUR)
        max_per_etf_budget: Maximum to spend on a single ETF if expensive (default 150 EUR)
        
    Returns:
        Formatted string for Telegram with optimization results
    """
    # Extract prices and tickers
    tickers = [stock.ticker_symbol for stock in stocks]
    current_prices = [stock.current_price for stock in stocks]
    predicted_prices = [stock.predict_price for stock in stocks]
    dividend_yields = [stock.dividend_yield for stock in stocks]
    expense_ratios = [stock.expense_ratio for stock in stocks]
    
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
        
        # Calculate net profit per stock: (capital_gain + dividend_income) * (1 - expense_ratio)
        total_net_profit = 0.0
        for i, shares in enumerate(individual):
            if shares == 0:
                continue
                
            current_price = current_prices[i]
            predicted_price = predicted_prices[i]
            dividend_yield = dividend_yields[i]
            expense_ratio = expense_ratios[i]
            ownership_weight = ownership_weights[i]
            
            # Capital gain from price appreciation
            capital_gain = (predicted_price - current_price) * shares
            
            # Dividend income (assuming annual yield)
            dividend_income = current_price * dividend_yield * shares
            
            # Total gross profit
            gross_profit = capital_gain + dividend_income
            
            # Apply expense ratio penalty (net after fees)
            net_profit = gross_profit * (1 - expense_ratio)
            
            # Apply ownership weight for diversification
            weighted_profit = net_profit * ownership_weight
            total_net_profit += weighted_profit
        
        # Penalize if cost exceeds budget
        if cost > budget:
            # Heavy penalty for exceeding budget
            return 100000000000, -10000000000
        
        # Objective 1: Minimize deviation from budget (prefer spending close to budget)
        # Objective 2: Maximize weighted net profit (after expenses and dividends)
        budget_deviation = abs(budget - cost)
        return budget_deviation, total_net_profit
    
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
            # Calculate detailed profit breakdown
            capital_gain = (predicted_prices[i] - current_prices[i]) * shares
            dividend_income = current_prices[i] * dividend_yields[i] * shares
            gross_profit = capital_gain + dividend_income
            net_profit = gross_profit * (1 - expense_ratios[i])
            stock_name = stocks[i].stock_name
            results.append((ticker, stock_name, shares, cost, net_profit, capital_gain, dividend_income, expense_ratios[i]))
    
    # Sort by net profit descending
    results.sort(key=lambda x: x[4], reverse=True)
    
    # Create formatted string for Telegram
    if not results:
        return "📊 *Portfolio Optimization Results*\n\nNo ETFs to buy with current budget and constraints."
    
    total_cost = sum(r[3] for r in results)
    total_net_profit = sum(r[4] for r in results)
    total_capital_gain = sum(r[5] for r in results)
    total_dividend_income = sum(r[6] for r in results)
    
    message_lines = [
        "📈 *Recommended Buys:*",
        ""
    ]
    
    for ticker, stock_name, shares, cost, net_profit, capital_gain, dividend_income, expense_ratio in results:
        net_profit_percentage = (net_profit / cost * 100) if cost > 0 else 0
        expense_amount = (capital_gain + dividend_income) * expense_ratio
        
        message_lines.append(f"• *{ticker}*: {shares} shares")
        message_lines.append(f"  {stock_name}")
        message_lines.append(f"  Cost: €{cost:.2f}")
        message_lines.append(f"  Net Profit: €{net_profit:.2f} ({net_profit_percentage:.1f}%)")
        message_lines.append(f"    - Capital Gain: €{capital_gain:.2f}")
        message_lines.append(f"    - Dividend Income: €{dividend_income:.2f}")
        message_lines.append(f"    - Expenses ({(expense_ratio*100):.2f}%): €{expense_amount:.2f}")
        message_lines.append("")
    
    message_lines.append(f"💰 *Total Investment:* €{total_cost:.2f}")
    message_lines.append(f"📈 *Total Net Profit:* €{total_net_profit:.2f}")
    message_lines.append(f"   - Capital Gains: €{total_capital_gain:.2f}")
    message_lines.append(f"   - Dividend Income: €{total_dividend_income:.2f}")
    message_lines.append(f"   - Total Gross Profit: €{(total_capital_gain + total_dividend_income):.2f}")
    
    return "\n".join(message_lines)

def __get_etf_map():
    etf_to_count = requests.get(configuration.GET_AND_INCREMENT_COUNTER_URL, params={"etf": "true"})    
    return etf_to_count.json()
  
