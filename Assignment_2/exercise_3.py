import numpy as np


def init_particles(N_o, N_c, N_d):
    random = np.random.default_rng()

    x = []
    for i in range(N_o):
        x.append(random.uniform(size=N_c * N_d).reshape((-1, N_d)))

    return x


def update_particle(x_i, data_points):
    for z in data_points:
        return 0


def pso_clustering(max_iter=10):
    x = init_particles(10, 5, 2)

    for i in range(max_iter):
        for x_i in x:
            update_particle(x_i, [])
        # for each particle x_i
        # update global best
        # update each particle x_i using PSO update rules (formula)

    return x


def k_means_clustering():
    N_d = 2
    N_o = 10
    N_c = 5
    z_p = 2
    m = []
    n = []
    C = []

    return 0


def exercise_3():
    x = pso_clustering()
    print(x)


exercise_3()
