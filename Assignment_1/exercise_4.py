import random
import matplotlib.pyplot as plt


def fitness(bitstring: str) -> int:
    """Computes the fitness by counting the number of 1's in a given bitstring.
    Args:
        bitstring (str): The bitstring of which the fitness is computed.
    Returns:
        int: The fitness of the bitstring.
    """
    return sum(map(int, bitstring))


def bitflip(bitstring: str, p: float) -> str:
    """Randomly flips bits in the given bitstring with a specified probability.
    Args:
        bitstring (str): Bitstring of which bits are flipped.
        p (float): Probability of which a single bit is flipped.
    Returns:
        str: Bitstring with flipped bits according to provided probability.
    """
    result = []
    for b in bitstring:
        if (random.uniform(0, 1) < p):
            result.append('0' if b == '1' else '1')  # Apply bitflip
        else:
            result.append(b)
    return ''.join(result)


def GA(l: int, p: float, n_iter: int = 1500, replace: bool = False, show: int = -1) -> list:
    """Executes a standar GA algorithm for counting ones problem.
    Args:
        l (int): Length of the bitstring.
        p (float): Probability of bitflip in mutation step.
        n_iter (int, optional): Number of iterations. Defaults to 1500.
        replace (bool, optional): If True, always replace x with x_m (exercise 4c). Defaults to False.
        show (int, optional): Print string status each 'show' number of iterations. Defaults to -1.
    Returns:
        list(int): List of fitness scores at each iteration of the algorithm.
    """
    # Set initial values
    x = '{0:b}'.format(random.getrandbits(l))
    goal = ''.join(['1' for _ in range(l)])
    results = []
    fitness_scores = []

    # Run GA algorithm
    for i in range(n_iter):
        x_m = bitflip(x, p)

        # Print info
        if i % show == 1:
            print(f"Iteration: {i+1}\n" +
                  f"x_m: {x_m}\n" +
                  f"fitness: {fitness(x_m)}\n")

        # Replace x with x_m if condition is satisfied
        if replace or fitness(x_m) > fitness(x):
            x = x_m

        # Append scores
        fitness_scores.append(fitness(x))
        results.append(max(fitness_scores))

        # # If optimum is reached, terminate (to obtain nicer plots, this code is disabled)
        # if x == goal:
        #     return results

    return results


def exercise_4():
    l = 100     # Length of bitstring
    p = 1/l     # Probability of flipping bit
    n = 1500    # Number of iterations

    # Exercise 4a: Run algorithm for 1500 iterations and plot results
    print(f"\nEXECUTING CODE OF EXERCISE 4A...\n")
    results = GA(l, p, n)
    plt.figure(figsize=(7, 7))
    plt.plot(results)
    plt.title("Best fitness against number of iterations of (1+1)-GA")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.ylim(top=101)
    plt.show()

    # Exercise 4b: Perform 10 runs of GA plot results in single figure
    print(f"\nEXECUTING CODE OF EXERCISE 4b...\n")
    count_optimum = 0
    plt.figure(figsize=(7, 7))
    for _ in range(10):
        results = GA(l, p, n)
        plt.plot(results)

        if results[len(results)-1] == l:
            count_optimum += 1
    plt.title("Best fitness against number of iterations of 10 runs of (1+1)-GA")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.ylim(top=101)
    plt.show()
    print(f"Number of times global optimum found: {count_optimum}")

    # Exercise 4c: Modify step c and perform 10 runs of modified GA algorithm
    print(f"\nEXECUTING CODE OF EXERCISE 4c...\n")
    count_optimum = 0
    plt.figure(figsize=(7, 7))
    for _ in range(10):
        results = GA(l, p, n, replace=True)
        plt.plot(results)

        if results[len(results)-1] == l:
            count_optimum += 1
    plt.title(
        "Best fitness against number of iterations of 10 runs of (1+1)-GA with modification")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.ylim(top=101)
    plt.show()
    print(f"Number of times global optimum found: {count_optimum}")


if __name__ == "__main__":
    exercise_4()
