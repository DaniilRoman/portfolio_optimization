import random
from collections.abc import Callable

import requests
from deap import algorithms, base, creator, tools
from deap.base import Toolbox

from config.configuration import settings
from src.logic.data.data import StockData

FUN_WEIGHTS_RISK_AWARE = (-1.0, 1.0)
FUN_WEIGHTS_PROFIT_ONLY = (-1.0, 1.0)


def __gen_one_individual(max_count_data, current_prices=None, budget=None):
    """Generate a random individual that respects budget constraints."""
    if current_prices is not None and budget is not None:
        # Try to generate individuals that respect budget
        individual = [0] * len(max_count_data)
        remaining_budget = budget
        
        # Try to add shares for each ETF, starting with random order
        indices = list(range(len(max_count_data)))
        random.shuffle(indices)
        
        for i in indices:
            if max_count_data[i] == 0:
                continue
                
            max_shares = max_count_data[i]
            price = current_prices[i]
            
            if price <= 0:
                continue
                
            # Calculate maximum shares we can afford with remaining budget
            max_affordable = min(max_shares, int(remaining_budget / price))
            
            if max_affordable > 0:
                # Randomly choose number of shares (could be 0)
                shares = random.randint(0, max_affordable)
                individual[i] = shares
                remaining_budget -= shares * price
                
                if remaining_budget <= 0:
                    break
        return individual
    else:
        # Fallback to simple random generation
        return [random.randint(0, max_count) for max_count in max_count_data]


def __evaluate(individual, predicted_prices, prices, budget):
    predicted_cost = sum(x * y for x, y in zip(predicted_prices, individual, strict=False))
    cost = sum(x * y for x, y in zip(prices, individual, strict=False))

    if cost > budget:
        return 100000000000, -10000000000
    return abs(budget - cost), predicted_cost - cost


def __calculate_volatility_risk(individual, stocks, current_prices):
    """
    Calculate portfolio volatility risk using historical volatility and GARCH forecasts.
    Returns normalized risk score (higher = more volatile = worse).
    """
    total_value = sum(individual[i] * current_prices[i] for i in range(len(individual)))
    
    if total_value == 0:
        return 0.0
    
    # Calculate weighted average of ETF standard deviations and forecast volatility
    weighted_volatility = 0.0
    weighted_forecast_volatility = 0.0
    
    for i, shares in enumerate(individual):
        if shares == 0:
            continue
        
        etf_value = shares * current_prices[i]
        weight = etf_value / total_value
        
        # Historical volatility (standard deviation)
        std_dev = stocks[i].standard_deviation
        
        # If standard deviation is 0 or not available, use beta as fallback
        if std_dev <= 0:
            # Use beta as proxy for volatility (beta * market_volatility)
            # Assume market volatility of 0.15 (15% annualized)
            market_volatility = 0.15
            std_dev = stocks[i].beta * market_volatility
        
        # Standard deviation is already annualized (from yfinance)
        weighted_volatility += weight * std_dev

        forecast_volatility = float(getattr(stocks[i], "forecast_volatility", 0.0))
        if forecast_volatility <= 0:
            # Prophet path: convert absolute forecast-band width to relative uncertainty.
            pred_unc = float(getattr(stocks[i], "prediction_uncertainty", 0.0))
            if current_prices[i] > 0 and pred_unc > 0:
                forecast_volatility = pred_unc / current_prices[i]

        weighted_forecast_volatility += weight * forecast_volatility

    # Combine historical volatility and forecast volatility
    combined_risk = (0.7 * weighted_volatility) + (0.3 * weighted_forecast_volatility)
    
    return combined_risk


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
    excess_concentration = max(0, max_sector_exposure - settings.ga.max_sector_concentration)
    
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
    toolbox.register("mate", tools.cxUniform, indpb=settings.ga.mate_indpb)
    toolbox.register("mutate", mutFlipBit, indpb=settings.ga.mutation_indpb)
    toolbox.register("select", tools.selTournament, tournsize=settings.ga.tournament_size)
    return toolbox


def __optimize_internal(toolbox, gen_individual_func):
    pop = [creator.Individual(gen_individual_func()) for i in range(settings.ga.population)]
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses, strict=False):
        ind.fitness.values = fit

    return algorithms.eaSimple(pop, toolbox, cxpb=settings.ga.crossover_rate, mutpb=settings.ga.mutation_rate, ngen=settings.ga.generations)


def _prepare_stock_data(stocks: list[StockData]):
    """Extract and prepare stock data for optimization."""
    tickers = [stock.ticker_symbol for stock in stocks]
    current_prices = [stock.current_price for stock in stocks]
    predicted_prices = [stock.predict_price for stock in stocks]
    dividend_yields = [stock.dividend_yield for stock in stocks]
    expense_ratios = [stock.expense_ratio for stock in stocks]
    
    return tickers, current_prices, predicted_prices, dividend_yields, expense_ratios


def _calculate_max_shares(current_prices: list[float], max_per_etf_budget: float = 50.0):
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


def _create_ownership_weights(tickers: list[str], etf_map: dict[str, int]):
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
    current_prices: list[float],
    predicted_prices: list[float],
    dividend_yields: list[float],
    expense_ratios: list[float],
    ownership_weights: list[float],
    stocks: list[StockData],
    budget: float,
    include_risk: bool = True
) -> Callable:
    """Create an evaluator function for the genetic algorithm."""
    
    def evaluate_func(individual):
        # Calculate cost and predicted profit
        total_cost = sum(x * y for x, y in zip(current_prices, individual, strict=False))
        
        # Calculate net profit per stock: (capital_gain + dividend_income) - (cost * expense_ratio)
        total_net_profit = 0.0
        for i, shares in enumerate(individual):
            if shares == 0:
                continue

            current_price = current_prices[i]
            predicted_price = predicted_prices[i]
            dividend_yield = dividend_yields[i]
            expense_ratio = expense_ratios[i]
            ownership_weight = ownership_weights[i]

            expected_gain = predicted_price - current_price
            capital_gain = expected_gain * shares
            
            # Dividend income (assuming annual yield)
            dividend_income = current_price * dividend_yield * shares
            
            # Total cost of investment
            cost = current_price * shares
            
            # Total gross profit
            gross_profit = capital_gain + dividend_income
            
            # Apply expense ratio penalty (annual fee on assets under management)
            # Expense reduces profit: fee = cost * expense_ratio
            expense_fee = cost * expense_ratio
            net_profit = gross_profit - expense_fee
            
            # Apply ownership weight for diversification
            weighted_profit = net_profit * ownership_weight
            total_net_profit += weighted_profit
        
        # Penalize if cost exceeds budget
        if total_cost > budget:
            # Heavy penalty for exceeding budget
            return 100000000000, -10000000000
        
        budget_deviation = abs(budget - total_cost)
        if include_risk:
            # Calculate risk components
            volatility_risk = __calculate_volatility_risk(individual, stocks, current_prices)
            sector_risk = __calculate_sector_concentration_risk(individual, stocks, current_prices)
            overlap_risk = __calculate_company_overlap_risk(individual, stocks, current_prices)
            
            # Combined risk score with weights: 40% volatility, 35% sector, 25% overlap
            # Scale risk to be comparable to profit values (profit is in euros, risk is 0-1 scale)
            # Multiply by a scaling factor to make risk matter more in the optimization
            risk_scaling_factor = 20.0  # Reduced from 100.0 to better balance profit vs risk
            total_risk = risk_scaling_factor * (
                0.25 * volatility_risk +
                0.4 * sector_risk +
                0.35 * overlap_risk
            )
            
            # For risk-aware: maximize (profit - risk_penalty)
            adjusted_profit = total_net_profit - total_risk
            return budget_deviation, adjusted_profit
        else:
            # Profit-only optimization: maximize profit
            return budget_deviation, total_net_profit
    
    return evaluate_func


def _run_genetic_algorithm(
    stocks: list[StockData],
    budget: float,
    max_per_etf_budget: float,
    include_risk: bool = True
) -> tuple[list[int], list[StockData], list[float], list[float], list[float], list[float], list[str]]:
    """Run genetic algorithm optimization with specified risk inclusion."""
    # Get current ETF ownership
    etf_map = __get_etf_map()
    return _run_genetic_algorithm_with_map(stocks, budget, max_per_etf_budget, etf_map, include_risk)


def _run_genetic_algorithm_with_map(
    stocks: list[StockData],
    budget: float,
    max_per_etf_budget: float,
    etf_map: dict[str, int],
    include_risk: bool = True
) -> tuple[list[int], list[StockData], list[float], list[float], list[float], list[float], list[str]]:
    """Run genetic algorithm optimization with pre-fetched ETF ownership map."""
    # Prepare data
    tickers, current_prices, predicted_prices, dividend_yields, expense_ratios = _prepare_stock_data(stocks)
    
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
                # Mutate to a random value within bounds (0 to max_shares_per_stock[i])
                # This is better than simple flip as it explores more of the search space
                individual[i] = random.randint(0, max_shares_per_stock[i])
        return individual,
    
    # Create individual generator
    def gen_one_individual_wrapper():
        return __gen_one_individual(max_shares_per_stock, current_prices, budget)
    
    # Select weights based on risk inclusion
    weights = FUN_WEIGHTS_RISK_AWARE if include_risk else FUN_WEIGHTS_PROFIT_ONLY
    
    # Run optimization
    toolbox = __create_toolbox(evaluator, weights, mutFlipBit)
    best_solution = __optimize_internal(toolbox, gen_one_individual_wrapper)
    best_individual = tools.selBest(best_solution[0], 1)[0]
    
    return best_individual, stocks, current_prices, predicted_prices, dividend_yields, expense_ratios, tickers


def _format_portfolio_results(
    best_individual: list[int],
    stocks: list[StockData],
    current_prices: list[float],
    predicted_prices: list[float],
    dividend_yields: list[float],
    expense_ratios: list[float],
    tickers: list[str],
    include_risk: bool = True
) -> str:
    """Format portfolio optimization results into a string."""
    # Format results: only include ETFs with positive share count
    results = []
    for i, (ticker, shares) in enumerate(zip(tickers, best_individual, strict=False)):
        if shares > 0:
            cost = shares * current_prices[i]

            expected_gain = predicted_prices[i] - current_prices[i]
            capital_gain = expected_gain * shares

            dividend_income = current_prices[i] * dividend_yields[i] * shares
            gross_profit = capital_gain + dividend_income
            # Expense reduces profit: fee = cost * expense_ratio
            expense_fee = cost * expense_ratios[i]
            net_profit = gross_profit - expense_fee
            stock_name = stocks[i].stock_name

            results.append((ticker, stock_name, shares, cost, net_profit, capital_gain,
                           dividend_income, expense_ratios[i], expense_fee,
                           float(getattr(stocks[i], "forecast_volatility", 0.0))))
    
    # Sort by net profit descending
    results.sort(key=lambda x: x[4], reverse=True)
    
    # Create formatted string
    if not results:
        return "No ETFs to buy with current budget and constraints."
    
    total_cost = sum(r[3] for r in results)
    total_net_profit = sum(r[4] for r in results)
    total_capital_gain = sum(r[5] for r in results)
    total_dividend_income = sum(r[6] for r in results)
    total_expense_fee = sum(r[8] for r in results)

    message_lines = []

    for ticker, stock_name, shares, cost, net_profit, capital_gain, dividend_income, expense_ratio, expense_fee, forecast_volatility in results:
        net_profit_percentage = (net_profit / cost * 100) if cost > 0 else 0

        message_lines.append(f"• *{ticker}*: {shares} shares")
        message_lines.append(f"  {stock_name}")
        message_lines.append(f"  Cost: €{cost:.2f}")
        message_lines.append(f"  Net Profit: €{net_profit:.2f} ({net_profit_percentage:.1f}%)")
        message_lines.append(f"    - Capital Gain: €{capital_gain:.2f}")
        message_lines.append(f"    - Dividend Income: €{dividend_income:.2f}")
        message_lines.append(f"    - Expenses ({(expense_ratio*100):.2f}%): €{expense_fee:.2f}")
        if forecast_volatility > 0:
            message_lines.append(f"    - Forecast Volatility: {(forecast_volatility*100):.2f}%")
        message_lines.append("")
    
    message_lines.append(f"💰 *Total Investment:* €{total_cost:.2f}")
    message_lines.append(f"📈 *Total Net Profit:* €{total_net_profit:.2f}")
    message_lines.append(f"   - Capital Gains: €{total_capital_gain:.2f}")
    message_lines.append(f"   - Dividend Income: €{total_dividend_income:.2f}")
    message_lines.append(f"   - Total Expenses: €{total_expense_fee:.2f}")
    message_lines.append(f"   - Total Gross Profit: €{(total_capital_gain + total_dividend_income):.2f}")
    
    # Always calculate and display risk metrics, regardless of whether they were used in optimization
    # Calculate risk metrics for the final portfolio
    final_volatility = __calculate_volatility_risk(best_individual, stocks, current_prices)
    final_sector_risk = __calculate_sector_concentration_risk(best_individual, stocks, current_prices)
    final_overlap_risk = __calculate_company_overlap_risk(best_individual, stocks, current_prices)
    
    message_lines.append("")
    message_lines.append("⚠️ *Risk Metrics:*")
    message_lines.append(f"   - Volatility: {(final_volatility*100):.1f}%")
    message_lines.append(f"   - Sector Concentration: {final_sector_risk:.3f}")
    message_lines.append(f"   - Company Overlap: {final_overlap_risk:.3f}")
    
    return "\n".join(message_lines)


def optimize(stocks: list[StockData], budget: float = 50.0, max_per_etf_budget: float = 50.0) -> str:
    """
    Optimize portfolio to suggest what ETFs to buy next.
    
    Args:
        stocks: List of StockData objects containing current and predicted prices
        budget: Ideal budget to spend (default 50 EUR)
        max_per_etf_budget: Maximum to spend on a single ETF if expensive. 
                           If None, defaults to min(150, budget / 2) to ensure total doesn't exceed budget.
        
    Returns:
        Formatted string for Telegram with optimization results
    """
    # Set reasonable default for max_per_etf_budget if not provided
    if max_per_etf_budget is None:
        # Ensure max_per_etf_budget is at most half the budget to allow diversification
        max_per_etf_budget = min(50.0, budget / 2)
    elif max_per_etf_budget > budget:
        # Cap max_per_etf_budget at budget to prevent impossible constraints
        max_per_etf_budget = budget
    
    # Get current ETF ownership ONCE and reuse for both optimizations
    # This prevents the counter from being incremented between optimizations
    etf_map = __get_etf_map()
    
    # Run risk-aware optimization
    risk_aware_individual, risk_stocks, risk_current_prices, risk_predicted_prices, risk_dividend_yields, risk_expense_ratios, risk_tickers = _run_genetic_algorithm_with_map(
        stocks, budget, max_per_etf_budget, etf_map, include_risk=True
    )
    
    # Run profit-only optimization
    profit_only_individual, profit_stocks, profit_current_prices, profit_predicted_prices, profit_dividend_yields, profit_expense_ratios, profit_tickers = _run_genetic_algorithm_with_map(
        stocks, budget, max_per_etf_budget, etf_map, include_risk=False
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
    etf_to_count = requests.get(settings.GET_AND_INCREMENT_COUNTER_URL, params={"etf": "true"})    
    return etf_to_count.json()
  
