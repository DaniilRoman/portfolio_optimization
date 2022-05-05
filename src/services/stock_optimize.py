from deap import base, creator, tools, algorithms
from deap.base import Toolbox
from tqdm import tqdm

import random

# CXPB  is the probability with which two individuals are crossed
# MUTPB is the probability for mutating an individual
from data import StockOptimizationJob

CXPB, MUTPB, NUMBER_OF_ITERATIONS, NUMBER_OF_POPULATION = 0.3, 0.7, 300, 250
FUN_WEIGHTS = {"cost_func": -1.0, "profit_func": 1.0}


def gen_one_individual(max_count_data):
    return [random.randint(0, max_count) for max_count in max_count_data]

def evaluate(individual, predicted_prices, prices, budget):
    # individual = individual[0]
    predicted_cost = sum(x * y for x, y in zip(predicted_prices, individual))
    cost = sum(x * y for x, y in zip(prices, individual))

    # print(f"COST: {cost}")
    if cost > budget:
        return 100000000000000000000, -100000000000000000000
    return abs(budget - cost), predicted_cost - cost

def rand_int():
    return random.randint(0, 10)

def create_toolbox(eval_func, weights: tuple) -> Toolbox:
    creator.create("FitnessFunc", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessFunc)

    toolbox = Toolbox()
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxUniform, indpb=0.05)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.6)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

def optimize_internal_v2(toolbox, gen_individual_func):
    pop = [creator.Individual(gen_individual_func()) for i in range(NUMBER_OF_POPULATION)]
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    return algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NUMBER_OF_ITERATIONS)

def optimize_internal(toolbox: Toolbox):
    pop = toolbox.population(n=NUMBER_OF_POPULATION)

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

    toolbox = create_toolbox(evaluate_func_wrapper, tuple(FUN_WEIGHTS.values()))
    best_solution = optimize_internal_v2(toolbox, gen_one_individual_wrapper)
    return tools.selBest(best_solution[0], 1)[0]
