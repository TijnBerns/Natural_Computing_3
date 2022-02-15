from deap import base, creator, gp, tools, algorithms
import operator
import math
import numpy as np
import matplotlib.pyplot as plt


def protectedDiv(left, right):
    if right != 0:
        return left / right
    else:
        return 10 ** 9


def protectedLog(left):
    try:
        return math.log(left)
    except ValueError:
        return - 10 ** 9


def protectedExp(left):
    try:
        return math.exp(left)
    except OverflowError:
        return 10 ** 9


def protectedSin(left):
    try:
        return math.sin(left)
    except ValueError:
        return 10 ** 9


def protectedCos(left):
    try:
        return math.cos(left)
    except ValueError:
        return 10 ** 9


def evalSymbReg(individual, toolbox, input, y):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    y_hat = np.array([func(x) for x in input])
    return - np.sum(np.abs(y - y_hat)),


def exercise_8():
    print("EXECUTING CODE OF EXERCISE 8...")

    # Specify input and output
    input = list(np.linspace(-1, 1, 21))
    y = np.array([0, -0.1629, -0.2624, -0.3129, -0.3264, -0.3125, -0.2784, -0.2289, -0.1664, -0.0909, 0.,
                  0.1111, 0.2496, 0.4251, 0.6496, 0.9375, 1.3056, 1.7731, 2.3616, 3.0951, 4])

    # Specify primitive set
    pset = gp.PrimitiveSet("MAIN", arity=1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedLog, 1)
    pset.addPrimitive(protectedExp, 1)
    pset.addPrimitive(protectedSin, 1)
    pset.addPrimitive(protectedCos, 1)
    pset.addPrimitive(protectedDiv, 2)
    pset.renameArguments(ARG0='x')

    # Fitness
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # Creating toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
    toolbox.register("individual", tools.initIterate,
                     creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evalSymbReg,
                     toolbox=toolbox, input=input, y=y)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Crossover and mutation
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Add max depth to prevent bloat
    # toolbox.decorate("mate", gp.staticLimit(
    #     key=operator.attrgetter("height"), max_value=17))
    # toolbox.decorate("mutate", gp.staticLimit(
    #     key=operator.attrgetter("height"), max_value=17))

    # Statistics
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # Create the population
    pop = toolbox.population(n=100)

    # Run GP
    n_generations = 50
    p_crossover = 0.7
    p_mutation = 0
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, p_crossover, p_mutation, n_generations, stats=mstats,
                                   halloffame=hof, verbose=True)
    gen = log.select("gen")
    fit_max = log.chapters["fitness"].select("max")

    # Plot best fitness against number of iterations
    plt.figure(figsize=(7, 7))
    plt.plot(gen, fit_max)
    plt.title("Best fitness against number of iterations of GP algorithm")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.show()

    # Plot best fitness against number of iterations
    size_max = log.chapters["size"].select("max")
    size_avg = log.chapters["size"].select("avg")
    size_min = log.chapters["size"].select("min")
    plt.figure(figsize=(7, 7))
    plt.plot(gen, size_max, label="max")
    plt.plot(gen, size_avg, label="avg")
    plt.plot(gen, size_min, label="min")
    plt.fill_between(gen, size_min, size_max, alpha=0.3)
    plt.title(
        "Average, min and max tree sizes against number of iterations of GP algorithm")
    plt.xlabel("Iteration")
    plt.ylabel("Size")
    plt.ylim(bottom=0)
    plt.show()


if __name__ == "__main__":
    exercise_8()
