from deap import base, creator, tools, algorithms
from deap.base import Toolbox

import random

# CXPB  is the probability with which two individuals are crossed
# MUTPB is the probability for mutating an individual
from data import StockOptimizationJob

CXPB, MUTPB, NUMBER_OF_ITERATIONS, NUMBER_OF_POPULATION = 0.3, 0.7, 100, 70 # 200
FUN_WEIGHTS = {"cost_func": -1.0, "profit_func": 1.0}


def gen_one_individual(max_count_data):
    return [random.randint(0, max_count) for max_count in max_count_data]


def evaluate(individual, predicted_prices, prices, budget):
    predicted_cost = sum(x * y for x, y in zip(predicted_prices, individual))
    cost = sum(x * y for x, y in zip(prices, individual))

    if cost > budget:
        return 100000000000, -10000000000
    return abs(budget - cost), predicted_cost - cost

def create_toolbox(eval_func, weights: tuple, mutFlipBit) -> Toolbox:
    creator.create("FitnessFunc", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessFunc)

    toolbox = Toolbox()
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", mutFlipBit, indpb=0.4)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def optimize_internal(toolbox, gen_individual_func):
    pop = [creator.Individual(gen_individual_func()) for i in range(NUMBER_OF_POPULATION)]
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    return algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NUMBER_OF_ITERATIONS)


def optimize(job: StockOptimizationJob):
    def evaluate_func_wrapper(i):
        return evaluate(i, job.predicted_prices, job.current_prices, job.budget)

    def gen_one_individual_wrapper():
        return gen_one_individual(job.max_stock_count_list)

    def mutFlipBit(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = job.max_stock_count_list[i] - individual[i]

        return individual,

    toolbox = create_toolbox(evaluate_func_wrapper, tuple(FUN_WEIGHTS.values()), tools.mutFlipBit)
    best_solution = optimize_internal(toolbox, gen_one_individual_wrapper)
    return tools.selBest(best_solution[0], 1)[0]
