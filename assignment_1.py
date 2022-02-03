import random
import matplotlib.pyplot as plt
import argparse

"""Methods for exercise 4"""


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


def GA(l: int, p: float, n_iter: int = 1500) -> list(int):
    """Executes a standar GA algorithm for counting ones problem.
    Args:
        l (int): Length of the bitstring.
        p (float): Probability of bitflip in mutation step.
        n_iter (int, optional): Number of iterations. Defaults to 1500.
    Returns:
        list(int): List of fitness scores at each iteration of the algorithm.
    """
    x = '{0:b}'.format(random.getrandbits(l))
    goal = ''.join(['1' for _ in range(l)])
    results = []
    print(f"Initial bitstring: {x}\n")

    for i in range(n_iter):
        x_m = bitflip(x, p)
        results.append(fitness(x_m))

        if (i + 1) % 100 == 0:
            print(f"Iteration: {i+1}\n" +
                  f"x_m: {x_m}\n" +
                  f"fitness: {fitness(x_m)}\n")

        if x_m == goal:
            print(f"Goal reached in {i} iterations\n")
            return results

        if fitness(x_m) > fitness(x):
            x = x_m

    print(f"Goal NOT reached\n")
    return results


def exercise_4():
    l = 100     # Length of bitstring
    p = 1/l     # Probability of flipping bit

    # Run algorithm for 1500 iterations and plot results
    results = GA(l, p, 1500)
    print(f"Fitness score at each iteration: \n{results}\n")
    plt.plot(results)
    plt.show()


"""Methods for exercise 6"""


def exercise_6():
    pass


"""Methods for exercise 6"""


def exercise_8():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exercise", choices=[4, 6, 8], default=4, type=int,
                        help="The number of the exercise of which the code will be executed")
    args = parser.parse_args()

    if args.exercise == 4:
        exercise_4()
    elif args.exercise == 6:
        exercise_6()
    elif args.exercise == 8:
        exercise_8()
