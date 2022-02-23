from curses import KEY_MARK
import numpy as np
import matplotlib.pyplot as plt


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
    means = [[-3, 0], [0, 0], [3, 0]]
    cov = [[0.50, 0.05], [0.05, 0.50]]
    art_data_1 = np.random.uniform(-1, 1, (400, 2))
    art_data_2 = generate_data(means, cov)

    _, clusters = k_means(3, art_data_2)
    plot_clusters(art_data_2, clusters)


if __name__ == "__main__":
    exercise_3()

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


