import requests
import config.configuration as configuration
from deap import base, creator, tools, algorithms
from deap.base import Toolbox
from typing import List, Dict, Tuple, Any, Callable
import random
import numpy as np

# CXPB  is the probability with which two individuals are crossed
# MUTPB is the probability for mutating an individual
from src.logic.data.data import StockData

# Recommended GA parameters for 20 ETFs portfolio optimization
CXPB = 0.35  # Crossover probability
MUTPB = 0.55  # Mutation probability
NUMBER_OF_ITERATIONS = 350  # Increased number of generations for better convergence
NUMBER_OF_POPULATION = 120  # Increased population size for better search space coverage
TOURNSIZE = 5  # Increased tournament size for better selection pressure
MUTATION_INDPB = 0.4  # Probability of each gene to be mutated
MATE_INDPB = 0.1  # Probability of each gene to be exchanged during crossover
MAX_SECTOR_CONCENTRATION = 0.40  # Max 40% in any single sector
FUN_WEIGHTS_RISK_AWARE = (-1.0, 1.0, 1.0)  # min deviation, max profit, min risk
FUN_WEIGHTS_PROFIT_ONLY = (0.0, 1.0, 0.0)  # ignore deviation, max profit, ignore risk


def __gen_one_individual(max_count_data):
    return [random.randint(0, max_count) for max_count in max_count_data]


def __evaluate(individual, predicted_prices, prices, budget):
    predicted_cost = sum(x * y for x, y in zip(predicted_prices, individual))
    cost = sum(x * y for x, y in zip(prices, individual))

    if cost > budget:
        return 100000000000, -10000000000
    return abs(budget - cost), predicted_cost - cost


def __calculate_volatility_risk(individual, stocks, current_prices):
    """
    Calculate portfolio volatility risk using ETF standard deviations.
    Returns normalized risk score (higher = more volatile = worse).
    """
    total_value = sum(individual[i] * current_prices[i] for i in range(len(individual)))
    
    if total_value == 0:
        return 0.0
    
    # Calculate weighted average of ETF standard deviations
    weighted_volatility = 0.0
    for i, shares in enumerate(individual):
        if shares == 0:
            continue
        
        etf_value = shares * current_prices[i]
        weight = etf_value / total_value
        std_dev = stocks[i].standard_deviation
        
        # Standard deviation is already annualized (from yfinance)
        weighted_volatility += weight * std_dev
    
    return weighted_volatility


def __calculate_sector_concentration_risk(individual, stocks, current_prices):
    """
    Calculate sector concentration risk.
    Returns a penalty score (higher = more concentrated = worse)
    """
    # Calculate total portfolio value
    total_value = sum(individual[i] * current_prices[i] for i in range(len(individual)))
    
    if total_value == 0:
        return 0.0
    
    # Aggregate sector exposure across all ETFs
    sector_exposure = {}
    
    for i, shares in enumerate(individual):
        if shares == 0:
            continue
        
        stock = stocks[i]
        etf_value = shares * current_prices[i]
        etf_weight = etf_value / total_value
        
        # Add this ETF's sector allocations to total exposure
        for sector, allocation in stock.sector_allocation.items():
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + (etf_weight * allocation)
    
    if not sector_exposure:
        return 0.0
    
    # Calculate concentration penalties
    max_sector_exposure = max(sector_exposure.values())
    
    # Herfindahl index (sum of squared concentrations)
    herfindahl = sum(exp**2 for exp in sector_exposure.values())
    
    # Penalty for exceeding max concentration
    excess_concentration = max(0, max_sector_exposure - MAX_SECTOR_CONCENTRATION)
    
    # Combined risk score (normalized to 0-1 scale)
    concentration_risk = (herfindahl + excess_concentration * 10)
    
    return concentration_risk


def __calculate_company_overlap_risk(individual, stocks, current_prices):
    """
    Calculate risk from overlapping company holdings across ETFs.
    Uses top_holdings data (np.ndarray format) to identify concentration.
    """
    total_value = sum(individual[i] * current_prices[i] for i in range(len(individual)))
    
    if total_value == 0:
        return 0.0
    
    # Aggregate company exposure across all ETFs
    company_exposure = {}
    
    for i, shares in enumerate(individual):
        if shares == 0:
            continue
        
        stock = stocks[i]
        etf_value = shares * current_prices[i]
        etf_weight = etf_value / total_value
        
        # Parse top_holdings (np.ndarray format: rows of [company_name, weight])
        for holding in stock.top_holdings:
            try:
                # holding is a numpy array row: [company_name, weight]
                company = holding[0]
                weight = float(holding[1])
                
                # Total exposure to this company
                company_exposure[company] = company_exposure.get(company, 0.0) + (etf_weight * weight)
            except (IndexError, ValueError, TypeError):
                continue
    
    if not company_exposure:
        return 0.0
    
    # Calculate risk metrics
    max_company_exposure = max(company_exposure.values())
    
    # Count companies with >5% exposure (high concentration)
    high_concentration_count = sum(1 for exp in company_exposure.values() if exp > 0.05)
    
    # Herfindahl index for company concentration
    herfindahl = sum(exp**2 for exp in company_exposure.values())
    
    # Combined risk (normalized)
    overlap_risk = max_company_exposure + (high_concentration_count * 0.01) + herfindahl
    
    return overlap_risk


def __create_toolbox(eval_func, weights: tuple, mutFlipBit) -> Toolbox:
    # Check if creator classes already exist to avoid warnings
    if not hasattr(creator, "FitnessFunc"):
        creator.create("FitnessFunc", base.Fitness, weights=weights)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessFunc)

    toolbox = Toolbox()
    toolbox.register("evaluate", eval_func)    
    toolbox.register("mate", tools.cxUniform, indpb=MATE_INDPB)
    toolbox.register("mutate", mutFlipBit, indpb=MUTATION_INDPB)
    toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)  # Increased for better selection pressure
    return toolbox


def __optimize_internal(toolbox, gen_individual_func):
    pop = [creator.Individual(gen_individual_func()) for i in range(NUMBER_OF_POPULATION)]
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    return algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NUMBER_OF_ITERATIONS)


def _prepare_stock_data(stocks: List[StockData]):
    """Extract and prepare stock data for optimization."""
    tickers = [stock.ticker_symbol for stock in stocks]
    current_prices = [stock.current_price for stock in stocks]
    predicted_prices = [stock.predict_price for stock in stocks]
    dividend_yields = [stock.dividend_yield for stock in stocks]
    expense_ratios = [stock.expense_ratio for stock in stocks]
    
    return tickers, current_prices, predicted_prices, dividend_yields, expense_ratios


def _calculate_max_shares(current_prices: List[float], max_per_etf_budget: float = 50.0):
    """Calculate maximum shares per ETF based on budget constraints."""
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
    
    return max_shares_per_stock


def _create_ownership_weights(tickers: List[str], etf_map: Dict[str, int]):
    """Calculate diversification weights based on current ownership."""
    ownership_weights = []
    for ticker in tickers:
        current_count = etf_map.get(ticker, 0)
        # Lower weight for ETFs we already own many of (encourage diversification)
        # Weight = 1 / (1 + current_count) - more ownership = lower weight
        weight = 1.0 / (1.0 + current_count)
        ownership_weights.append(weight)
    
    return ownership_weights


def _create_evaluator_factory(
    current_prices: List[float],
    predicted_prices: List[float],
    dividend_yields: List[float],
    expense_ratios: List[float],
    ownership_weights: List[float],
    stocks: List[StockData],
    budget: float,
    include_risk: bool = True
) -> Callable:
    """Create an evaluator function for the genetic algorithm."""
    
    def evaluate_func(individual):
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
            return 100000000000, -10000000000, 100000000000
        
        if include_risk:
            budget_deviation = abs(budget - cost)
            # Calculate risk components
            volatility_risk = __calculate_volatility_risk(individual, stocks, current_prices)
            sector_risk = __calculate_sector_concentration_risk(individual, stocks, current_prices)
            overlap_risk = __calculate_company_overlap_risk(individual, stocks, current_prices)
            
            # Combined risk score with weights: 40% volatility, 35% sector, 25% overlap
            total_risk = (
                0.25 * volatility_risk +
                0.4 * sector_risk +
                0.35 * overlap_risk
            )
            
            # Return tuple: (minimize deviation, maximize profit, minimize risk)
            return budget_deviation, total_net_profit, -total_risk  # Negative risk to minimize
        else:
            # Profit-only optimization: ignore risk
            return 0.0, total_net_profit, 0.0
    
    return evaluate_func


def _run_genetic_algorithm(
    stocks: List[StockData],
    budget: float,
    max_per_etf_budget: float,
    include_risk: bool = True
) -> Tuple[List[int], List[StockData], List[float], List[float], List[float], List[float], List[float]]:
    """Run genetic algorithm optimization with specified risk inclusion."""
    # Prepare data
    tickers, current_prices, predicted_prices, dividend_yields, expense_ratios = _prepare_stock_data(stocks)
    
    # Get current ETF ownership
    etf_map = __get_etf_map()
    
    # Calculate constraints and weights
    max_shares_per_stock = _calculate_max_shares(current_prices, max_per_etf_budget)
    ownership_weights = _create_ownership_weights(tickers, etf_map)
    
    # Create evaluator
    evaluator = _create_evaluator_factory(
        current_prices, predicted_prices, dividend_yields, expense_ratios,
        ownership_weights, stocks, budget, include_risk
    )
    
    # Create mutation function
    def mutFlipBit(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                # Flip within bounds of max_shares_per_stock
                individual[i] = max_shares_per_stock[i] - individual[i]
        return individual,
    
    # Create individual generator
    def gen_one_individual_wrapper():
        return __gen_one_individual(max_shares_per_stock)
    
    # Select weights based on risk inclusion
    weights = FUN_WEIGHTS_RISK_AWARE if include_risk else FUN_WEIGHTS_PROFIT_ONLY
    
    # Run optimization
    toolbox = __create_toolbox(evaluator, weights, mutFlipBit)
    best_solution = __optimize_internal(toolbox, gen_one_individual_wrapper)
    best_individual = tools.selBest(best_solution[0], 1)[0]
    
    return best_individual, stocks, current_prices, predicted_prices, dividend_yields, expense_ratios, tickers


def _format_portfolio_results(
    best_individual: List[int],
    stocks: List[StockData],
    current_prices: List[float],
    predicted_prices: List[float],
    dividend_yields: List[float],
    expense_ratios: List[float],
    tickers: List[str],
    include_risk: bool = True
) -> str:
    """Format portfolio optimization results into a string."""
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
    
    # Create formatted string
    if not results:
        return "No ETFs to buy with current budget and constraints."
    
    total_cost = sum(r[3] for r in results)
    total_net_profit = sum(r[4] for r in results)
    total_capital_gain = sum(r[5] for r in results)
    total_dividend_income = sum(r[6] for r in results)
    
    message_lines = []
    
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
    
    # Always calculate and display risk metrics, regardless of whether they were used in optimization
    # Calculate risk metrics for the final portfolio
    final_volatility = __calculate_volatility_risk(best_individual, stocks, current_prices)
    final_sector_risk = __calculate_sector_concentration_risk(best_individual, stocks, current_prices)
    final_overlap_risk = __calculate_company_overlap_risk(best_individual, stocks, current_prices)
    
    message_lines.append("")
    message_lines.append(f"⚠️ *Risk Metrics:*")
    message_lines.append(f"   - Volatility: {(final_volatility*100):.1f}%")
    message_lines.append(f"   - Sector Concentration: {final_sector_risk:.3f}")
    message_lines.append(f"   - Company Overlap: {final_overlap_risk:.3f}")
    
    return "\n".join(message_lines)


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
    # Run risk-aware optimization
    risk_aware_individual, risk_stocks, risk_current_prices, risk_predicted_prices, risk_dividend_yields, risk_expense_ratios, risk_tickers = _run_genetic_algorithm(
        stocks, budget, max_per_etf_budget, include_risk=True
    )
    
    # Run profit-only optimization
    profit_only_individual, profit_stocks, profit_current_prices, profit_predicted_prices, profit_dividend_yields, profit_expense_ratios, profit_tickers = _run_genetic_algorithm(
        stocks, budget, max_per_etf_budget, include_risk=False
    )
    
    # Format both results
    risk_aware_results = _format_portfolio_results(
        risk_aware_individual, risk_stocks, risk_current_prices, risk_predicted_prices,
        risk_dividend_yields, risk_expense_ratios, risk_tickers, include_risk=True
    )
    
    profit_only_results = _format_portfolio_results(
        profit_only_individual, profit_stocks, profit_current_prices, profit_predicted_prices,
        profit_dividend_yields, profit_expense_ratios, profit_tickers, include_risk=False
    )
    
    # Combine results with headers
    message_lines = [
        "📊 *Portfolio Optimization Results*",
        "",
        "⚠️ **Risk-Aware Optimization** (Minimizes risk while maximizing profit)",
        "",
        risk_aware_results,
        "",
        "📈 **Profit-Only Optimization** (Maximizes profit only)",
        "",
        profit_only_results
    ]
    
    return "\n".join(message_lines)

def __get_etf_map():
    etf_to_count = requests.get(configuration.GET_AND_INCREMENT_COUNTER_URL, params={"etf": "true"})    
    return etf_to_count.json()
  
