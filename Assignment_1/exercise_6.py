import random
import numpy as np


def fitness():
    pass


def crossover(p1: np.array, p2: np.array):
    assert (len(p1) == len(p2))
    # Randomly pick 2 cuts
    cut1 = random.randint(0, len(p1) - 1)
    cut2 = random.randint(cut1 + 1, len(p1))

    # Sublists of parents to be kept intact
    sub1 = p1[cut1:cut2]
    sub2 = p2[cut1:cut2]
    print(sub2)

    # Lists of elements which are missing in offsprings
    miss1 = [x for x in p2 if x not in sub1]
    miss2 = [x for x in p1 if x not in sub2]
    print(miss2)

    # The offsprings
    off1 = p1.copy()
    off2 = p2.copy()

    # Replace elements before first cut
    for i in range(cut1):
        off1[i] = miss1[i]
        off2[i] = miss2[i]

    # Replace relements after second cut
    j = cut1
    for i in range(cut2, len(off1)):
        off1[i] = miss1[j]
        off2[i] = miss2[j]
        j += 1

    return off1, off2


def mutate(parent: np.array) -> np.array:
    pos1, pos2 = random.sample(list(np.arange(len(parent))), 2)
    x = parent[pos1]
    parent[pos1] = parent[pos2]
    parent[pos2] = x

    return parent


def simple_EA():
    pass


def exercise_6():
    pass


if __name__ == "__main__":
    for i in range(1):
        crossover(np.array([3, 5, 7, 2, 1, 6, 4, 8]),
                  np.array([2, 5, 7, 6, 8, 1, 3, 4]))
