"""Genetic-algorithm portfolio optimizer using DEAP; reads tuning parameters from settings.ga; outputs OptimizationResult."""
import random
from collections.abc import Callable

import requests
from deap import algorithms, base, creator, tools
from deap.base import Toolbox

from config.configuration import settings
from src.logic.data.data import Allocation, AnalysisReport, OptimizationResult, Portfolio, RiskMetrics

StockData = AnalysisReport  # local alias kept for type annotations in this module

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
        std_dev = stocks[i].market.standard_deviation

        # If standard deviation is 0 or not available, use beta as fallback
        if std_dev <= 0:
            # Use beta as proxy for volatility (beta * market_volatility)
            # Assume market volatility of 0.15 (15% annualized)
            market_volatility = 0.15
            std_dev = stocks[i].market.beta * market_volatility

        # Standard deviation is already annualized (from yfinance)
        weighted_volatility += weight * std_dev

        forecast_volatility = stocks[i].forecast.forecast_volatility
        if forecast_volatility <= 0:
            # Prophet path: convert absolute forecast-band width to relative uncertainty.
            pred_unc = stocks[i].forecast.prediction_uncertainty
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
    tickers = [stock.asset.ticker_symbol for stock in stocks]
    current_prices = [stock.market.current_price for stock in stocks]
    predicted_prices = [stock.forecast.predict_price for stock in stocks]
    dividend_yields = [stock.market.dividend_yield for stock in stocks]
    expense_ratios = [stock.asset.expense_ratio for stock in stocks]
    
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


def _build_portfolio(
    best_individual: list[int],
    stocks: list[StockData],
    current_prices: list[float],
    predicted_prices: list[float],
    dividend_yields: list[float],
    expense_ratios: list[float],
    tickers: list[str],
) -> Portfolio:
    """Build a Portfolio value object from raw optimization output."""
    allocations: list[Allocation] = []
    for i, (_ticker, shares) in enumerate(zip(tickers, best_individual, strict=False)):
        if shares > 0:
            cost = shares * current_prices[i]
            capital_gain = (predicted_prices[i] - current_prices[i]) * shares
            dividend_income = current_prices[i] * dividend_yields[i] * shares
            expense_fee = cost * expense_ratios[i]
            net_profit = capital_gain + dividend_income - expense_fee
            allocations.append(Allocation(
                asset=stocks[i].asset,
                quantity=shares,
                total_cost=cost,
                net_profit=net_profit,
                capital_gain=capital_gain,
                dividend_income=dividend_income,
                expense_fee=expense_fee,
                expense_ratio=expense_ratios[i],
                forecast_volatility=stocks[i].forecast.forecast_volatility,
            ))

    allocations.sort(key=lambda a: a.net_profit, reverse=True)

    risk_metrics = RiskMetrics(
        volatility=__calculate_volatility_risk(best_individual, stocks, current_prices),
        sector_concentration=__calculate_sector_concentration_risk(best_individual, stocks, current_prices),
        company_overlap=__calculate_company_overlap_risk(best_individual, stocks, current_prices),
    )
    return Portfolio(allocations=allocations, risk_metrics=risk_metrics)


def optimize(stocks: list[StockData], budget: float = 50.0, max_per_etf_budget: float = 50.0) -> OptimizationResult:
    if max_per_etf_budget is None:
        max_per_etf_budget = min(50.0, budget / 2)
    elif max_per_etf_budget > budget:
        max_per_etf_budget = budget

    etf_map = __get_etf_map()

    risk_individual, r_stocks, r_prices, r_predicted, r_div, r_exp, r_tickers = _run_genetic_algorithm_with_map(
        stocks, budget, max_per_etf_budget, etf_map, include_risk=True
    )
    profit_individual, p_stocks, p_prices, p_predicted, p_div, p_exp, p_tickers = _run_genetic_algorithm_with_map(
        stocks, budget, max_per_etf_budget, etf_map, include_risk=False
    )

    return OptimizationResult(
        risk_aware=_build_portfolio(risk_individual, r_stocks, r_prices, r_predicted, r_div, r_exp, r_tickers),
        profit_only=_build_portfolio(profit_individual, p_stocks, p_prices, p_predicted, p_div, p_exp, p_tickers),
    )

def __get_etf_map():
    etf_to_count = requests.get(settings.GET_AND_INCREMENT_COUNTER_URL, params={"etf": "true"})    
    return etf_to_count.json()
  
