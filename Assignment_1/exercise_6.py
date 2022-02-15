import random
import numpy as np
import matplotlib.pyplot as plt


def fitness(cities: np.array, path: np.array) -> float:
    """Computes the fitness score of a given path

    Args:
        cities (np.array): N x 2 array of city coordinates
        path (np.array): 1 x N array representing a order of cities

    Returns:
        float: The fitness (1-(sum of distances)) of the given path
    """
    return 1 / sum([np.linalg.norm(cities[path[i]] - cities[path[i+1]])
                   for i in range(-1, len(path)-1)])


def compute_population_scores(cities: np.array, population: np.array) -> np.array:
    """Computes fitness scores for an array of paths

    Args:
        cities (np.array): N x 2 array of city coordinates
        population (np.array): M x N array of paths

    Returns:
        np.array: fitness scores for each path in population
    """
    return np.array([fitness(cities, candidate) for candidate in population])


def tournament(cities: np.array, population: np.array, k: int) -> np.array:
    """Performs a tournament between k candidates

    Args:
        cities (np.array): N x 2 array of coordinates
        population (np.array): M x N array of paths
        k (int, optional): Number of candidates in tournement. Defaults to 2.

    Returns:
        [type]: Winner of tournement
    """
    tour = np.random.choice(np.arange(len(population)), k, replace=False)
    tour_scores = compute_population_scores(cities, population[tour])
    tour_best = np.argmax(tour_scores)
    return population[tour[tour_best]]


def crossover(parent1: np.array, parent2: np.array):
    """Does a crossover overation between two paths

    Args:
        p1 (np.array): Parent 1
        p2 (np.array): Parent 2

    Returns:
        (np.array, np.array): The resulting offsprings
    """
    assert (len(parent1) == len(parent2))
    # Randomly pick 2 cuts
    cut1 = random.randint(0, len(parent1) - 1)
    cut2 = random.randint(cut1 + 1, len(parent1))

    # Sublists of parents to be kept intact
    sub1 = parent1[cut1:cut2]
    sub2 = parent2[cut1:cut2]

    # Lists of elements which are missing in offsprings
    miss1 = [x for x in parent2 if x not in sub1]
    miss2 = [x for x in parent1 if x not in sub2]

    # The offsprings
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    # Replace elements before first cut
    for i in range(cut1):
        offspring1[i] = miss1[i]
        offspring2[i] = miss2[i]

    # Replace relements after second cut
    j = cut1
    for i in range(cut2, len(offspring1)):
        offspring1[i] = miss1[j]
        offspring2[i] = miss2[j]
        j += 1

    return offspring1, offspring2


def mutate(parent: np.array) -> np.array:
    """Mutates a given path by performing a random swap of cities

    Args:
        parent (np.array): The path to be mutated

    Returns:
        np.array: The resulting offspring
    """
    pos1, pos2 = random.sample(list(np.arange(len(parent))), 2)
    x = parent[pos1]
    parent[pos1] = parent[pos2]
    parent[pos2] = x

    return parent


def read_TSP(fname: str) -> np.array:
    """Reads a .txt file to initialize the TSP problem

    Args:
        fname (str): The filename

    Returns:
        [type]: List of city coordinates
    """
    # Read contents of file
    with open(fname) as f:
        lines = f.readlines()

    # Give each set of coordinates a number
    cities = []
    for line in lines:
        city_coordinates = np.array(list(map(float, line.split())))
        cities.append(city_coordinates)

    cities = np.array(cities)
    print(cities.shape)
    return cities[:, 1:] if cities.shape == (len(cities), 3) else cities


def local_search(cities: np.array, path: np.array, max_iter=1):
    """Performs local search based on 2-opt algorithm

    Args:
        cities (np.array): N x 2 array of city coordinates
        path (np.array): 1 x N array of cities denoting the path
        max_iter (int, optional): Maximum number of local search iterations. Defaults to 1.

    Returns:
        np.array: Best found path during local search
    """
    best_path = path.copy()
    best_fitness = fitness(cities, best_path)
    found = True
    iter = 0

    while found and iter < max_iter:
        found = False
        iter += 1

        for i in range(1, len(path) - 2):

            for k in range(i + 1, len(path)):
                new_path = two_opt_swap(path, i, k)
                new_fitness = fitness(cities, new_path)

                if (new_fitness > best_fitness):
                    best_path = new_path
                    best_fitness = new_fitness
                    found = True

        path = best_path

    return best_path


def two_opt_swap(path: np.array, i: int, k: int):
    """

    Args:
        path (np.array): The path to be altered
        i (int): index 1 
        k (int): index 2

    Returns:
        np.array: altered path based on 2-opt
    """
    new_path = path.copy()
    new_path[i:k] = new_path[k-1:i-1:-1]
    return new_path


def EA(cities: np.array, pop_size: int = 10, iterations: int = 1500, k: int = 2, ma: bool = False):
    """Executes the EA algorithm

    Args:
        cities (np.array): N x 2 array of city coordinates
        pop_size (int, optional): Size of the entire population. Defaults to 10.
        iterations (int, optional): Number of iterations. Defaults to 1500.
        k (int, optional): Number of contenders during parent selection. Defaults to 2.
        ma (bool, optional): Flag, whether to include local search. Defaults to False.

    Returns:
        np.array, np.array: Best and average scores per iteration; and best candidate in final population 
    """
    population = np.array([np.random.permutation(len(cities))
                           for _ in range(pop_size)])

    if ma:
        for i in range(len(population)):
            population[i] = local_search(cities, population[i])

    pop_scores = compute_population_scores(cities, population)
    results = []

    for i in range(iterations):
        # Binary tournament
        parent1 = tournament(cities, population, k)
        parent2 = tournament(cities, population, k)

        # Recombine selected parents
        offsprings = np.array(crossover(parent1, parent2))

        # Mutate resulting offsprings
        for i in range(len(offsprings)):
            offsprings[i] = mutate(offsprings[i])
            if ma:
                offsprings[i] = local_search(cities, offsprings[i])

        # Population update (replace worst candidate in population with best offspring)
        off_scores = compute_population_scores(cities, offsprings)
        best_offspring = np.argmax(off_scores)
        worst_candidate = np.argmin(pop_scores)
        population[worst_candidate] = offsprings[best_offspring]
        pop_scores[worst_candidate] = off_scores[best_offspring]

        # Compute average and best fitness
        results.append([np.average(pop_scores), np.max(pop_scores)])

    return np.array(results), population[np.argmax(pop_scores)]


def plot_fitness(results: np.array, title: str = ""):
    """Plots the average and best fitness against the number of iterations

    Args:
        results (np.array): Array containin average and best results
        title (str): The title of the plot
    """
    plt.figure(figsize=(7, 7))
    plt.plot(results[:, 1], label="Best fitness")
    plt.plot(results[:, 0], label="Average fitness")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.show()


def plot_route(cities: np.array, path: np.array):
    """Plot the cities and the given path

    Args:
        cities (np.array): Array of coordinates representing the cities
        path (np.array): The order in which cities are traversed
    """
    plt.figure(figsize=(7, 7))
    plt.scatter(cities[:, 0], cities[:, 1])

    for i in range(-1, len(path)-1):
        plt.annotate(str(path[i]), (cities[path[i]][0], cities[path[i]][1]))
        plt.plot([cities[path[i]][0], cities[path[i+1]][0]],
                 [cities[path[i]][1], cities[path[i+1]][1]], color="tab:blue")

    plt.show()


def exercise_6():
    print("EXECUTING CODE OF EXERCISE 4...")
    files = ["data/airports.txt", "data/file-tsp.txt"]

    for file in files:
        print(f"Running experiments for {file}...")

        # Load data
        cities = read_TSP(file)

        # Run simple EA and plot results
        _, axis = plt.subplots(1, 2, figsize=(14, 7))
        for i in range(10):
            print(f"Simple EA {i}/10")
            results, path = EA(cities)
            axis[0].plot(results[:, 1])
            axis[1].plot(results[:, 0])
        for ax in axis:
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fitness")
        axis[0].set_title(
            "Best fitness against number of iterations of simple EA algorithm")
        axis[1].set_title(
            "Average fitness against number of iterations of simple EA algorithm")
        plt.show()
        plot_route(cities, path)

        # Run MA and plot results
        _, axis = plt.subplots(1, 2, figsize=(14, 7))
        for i in range(10):
            print(f"Memetic EA {i}/10")
            results, _ = EA(cities, ma=True)
            np.savetxt(f"{file[5:10]}_MA_{i}.txt", results)
            axis[0].plot(results[:, 1])
            axis[1].plot(results[:, 0])
        for ax in axis:
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fitness")
        axis[0].set_title(
            "Best fitness against number of iterations of memetic EA algorithm")
        axis[1].set_title(
            "Average fitness against number of iterations of memetic EA algorithm")
        plt.show()


if __name__ == "__main__":
    exercise_6()
