import math
import sys
from curses import KEY_MARK
import numpy as np
import matplotlib.pyplot as plt

random = np.random.default_rng()


def k_means(k: int, data: list):
    """_summary_

    Args:
        k (int): _description_
        data (list): _description_

    Returns:
        _type_: _description_
    """
    centroids = data[np.random.choice(      # initial centroids
        data.shape[0], k, replace=False)]
    clusters = np.zeros(len(data))          # initial cluster assignment
    update = True                           # flag to test stop condition
    iter = 0                                # iteration nr

    while update:
        iter += 1
        update = False

        # Assign data points to centroids
        clusters = np.array([np.argmin([np.linalg.norm(point - centroid)
                                        for centroid in centroids]) for point in data])

        # Update cluster centroids
        for i in range(len(centroids)):
            prev_centroid = np.copy(centroids[i])
            centroids[i] = np.average(data[np.where(clusters == i)[0]], axis=0)

            if np.any(prev_centroid != centroids[i]):
                update = True

    print(f"Found stable clustering of {k} clusters in {iter} iterations.")
    return centroids, clusters


def distance(z, centroid):
    d_squared = 0
    for k in range(len(z)):
        d_squared += (z[k] - centroid[k]) ** 2

    return math.sqrt(d_squared)


def particle_fitness(clustering: list, k: int):
    sum_over_k = 0
    for j in range(k):
        sum_over_k += sum(clustering[j]) / len(clustering[j])

    return sum_over_k / k


def update_particle_velocity(v: float, x_i, local_best, global_best):
    r_1 = random.uniform()
    r_2 = random.uniform()

    v_next = 0.72 * v + 1.49 * r_1 * (local_best - x_i) + 1.49 * r_2 * (global_best - x_i)

    return v_next


def init_particles(N_o: int, N_c, N_d):
    x = []
    for i in range(N_o):
        x.append(random.uniform(size=N_c * N_d).reshape((-1, N_d)))

    return x


def pso_clustering(k: int, data: list, max_iter: int, N_p: int):
    dim = len(data[0])
    x = init_particles(N_p, k, dim)
    v = np.zeros((N_p, k, dim))

    local_fits = [2 ^ 30 for p in range(N_p)]
    global_fit = 2 ^ 30
    local_bests = [[] for p in range(N_p)]
    global_best = []
    for iteration in range(max_iter):
        for i in range(len(x)):
            clustering = [[] for cluster in range(k)]
            for z in data:
                # calculate distance of z to each centroid
                distances = list(map(lambda centroid: distance(z, centroid), x[i]))

                # assign z to centroid with minimal distance
                clustering[distances.index(min(distances))].append(min(distances))

            # compute fitness of x_i
            fitness = particle_fitness(clustering, k)

            # update local best
            if fitness <= local_fits[i]:
                local_fits[i] = fitness
                local_bests[i] = x[i]

            # update global best
            if fitness <= global_fit:
                global_fit = fitness
                global_best = x[i]
                print(fitness)

        # update each particle x_i using PSO update rules (formula)
        for i in range(len(x)):
            v[i] = update_particle_velocity(v[i], x[i], local_bests[i], global_best)
            x[i] = x[i] + v[i]

    return x


def generate_data(means: np.array, cov: np.array) -> np.array:
    """Generates artificial data c

    Args:
        means (np.array): _description_
        cov (np.array): _description_

    Returns:
        np.array: _description_
    """
    data = []
    for mu in means:
        data.append(np.random.multivariate_normal(mu, cov, 150))
    return np.concatenate(np.array(data))


def plot_clusters(data, clusters):
    """_summary_

    Args:
        data (_type_): _description_
        clusters (_type_): _description_
    """
    plt.figure(figsize=(7, 7))
    for i in np.unique(clusters):
        cluster = data[np.where(clusters == i)[0]]
        plt.scatter(cluster[:, 0], cluster[:, 1], s=10)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def exercise_3() -> None:
    # Generate artificial datasets
    means = [[-3, 0], [0, 0], [3, 0]]
    cov = [[0.50, 0.05], [0.05, 0.50]]
    art_data_1 = np.random.uniform(-1, 1, (400, 2))
    art_data_2 = generate_data(means, cov)      
    
    # Perform clustering on both datasets and plot results
    _, clusters_1 = k_means(2, art_data_1)
    _, clusters_2 = k_means(3, art_data_2)
    #plot_clusters(art_data_1, clusters_1)
    #plot_clusters(art_data_2, clusters_2)

    # Perform PSO-clustering
    pso_clustering(3, art_data_2, 10, 10)

if __name__ == "__main__":
    exercise_3()
