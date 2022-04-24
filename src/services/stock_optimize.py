from deap import base
from deap import creator
from deap import tools
from deap.base import Toolbox
from tqdm import tqdm

import random

# CXPB  is the probability with which two individuals are crossed
# MUTPB is the probability for mutating an individual
from data import StockOptimizationJob

CXPB, MUTPB, NUMBER_OF_ITERATIONS = 0.5, 0.2, 100
FUN_WEIGHTS = {"profit_func": 1.0, "cost_func": -1.0}


def gen_one_individual(max_count_data):
    return [random.randint(0, max_count) for max_count in max_count_data]


def evaluate(individual, value_data, price_data, budget):
    individual = individual[0]
    profit = sum(x * y for x, y in zip(value_data, individual))
    sum_price = sum(x * y for x, y in zip(price_data, individual))

    if sum_price > budget:
        return 0, 100000000000000000000
    return profit, abs(budget - sum_price)


def create_toolbox(eval_func, gen_individual_func, weights: tuple) -> Toolbox:
    creator.create("FitnessFunc", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessFunc)

    toolbox = Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, gen_individual_func, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def optimize_internal(toolbox: Toolbox):
    pop = toolbox.population(n=300)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Begin the evolution
    for g in tqdm(range(NUMBER_OF_ITERATIONS)):

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1[0], child2[0])

                # fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant[0])
                del mutant.fitness.values

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    return best_ind, best_ind.fitness.values


def optimize(job: StockOptimizationJob):
    def evaluate_func_wrapper(i):
        return evaluate(i, job.predicted_prices, job.current_prices, job.budget)

    def gen_one_individual_wrapper():
        return gen_one_individual(job.max_stock_count_list)

    toolbox = create_toolbox(evaluate_func_wrapper, gen_one_individual_wrapper, tuple(FUN_WEIGHTS.values()))
    best_solution = optimize_internal(toolbox)
    return best_solution[0][0]
