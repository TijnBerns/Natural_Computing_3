#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#================= K-means Clustering =================#

def k_means(k: int, data: list, max_iter: int):
    """Exercutes the k-means clustering algorithm

    Args:
        k (int): The number of clusters
        data (list): The data to cluster

    Returns:
        (np.array, np.array): Array of centroids, and array of cluster assignment
    """
    centroids = get_random_points(data, k)  # random initial centroids
    clusters = np.zeros(len(data))          # initial cluster assignment
    update = True                           # flag to test stop condition
    iter = 0                                # iteration nr
 
    while update and iter < max_iter:
        iter += 1
        update = False

        # Assign data points to centroids
        clusters = assign_clusters(data, centroids)

        # Update cluster centroids
        for i in range(len(centroids)):
            prev_centroid = np.copy(centroids[i])
            centroids[i] = np.average(data[np.where(clusters == i)[0]], axis=0)

            if np.any(prev_centroid != centroids[i]):
                update = True

    # print(f"Found stable clustering of {k} clusters in {iter} iterations.")
    return centroids, clusters


#================= PSO Clustering =================#

def pso(k: int, data: np.array, n_particles: int, w: float, c1: float, c2: float, max_iter: int):
    particles = np.array([get_random_points(data, k)
                         for _ in range(n_particles)])      # List of particles
    velocities = np.zeros_like(particles)                   # Stores velocity of each particle
    local_best = np.zeros((n_particles, k, *data[0].shape)) # Stores best position of each particle
    local_best_fitness = np.full(n_particles, 1_000_000)    # Stores best fitness of each particle
    global_best = np.zeros(k)                               # The global best position
    global_best_fitness = 1_000_000                         # The global best fitness

    for _ in range(max_iter):
        for i, particle in enumerate(particles):
            clusters = assign_clusters(data, particle)
            fitness = compute_fitness(data, clusters, particle, k)
            
            # Update local best
            if fitness < local_best_fitness[i]:
                local_best[i] = np.copy(particle)
                local_best_fitness[i] = fitness
            
            # Update global best
            if fitness < global_best_fitness:
                global_best = np.copy(particle) 
                global_best_fitness = fitness
        
        # Update centroids      
        for i, particle in enumerate(particles):
            r1 = np.random.uniform()
            r2 = np.random.uniform()
            velocities[i] = w * velocities[i] + c1 * r1 * (local_best[i] - particle) + \
                c2 * r2 * (global_best - particle)
            particles[i] = particle + velocities[i]
            
    clusters = assign_clusters(data, global_best)
    return global_best, clusters
        

#================= UTILITY METHODS =================#

def get_random_points(data: np.array, n: int):
    """Draws two random points from a 2D np.array

    Args:
        data (np.array): Data from which samples are drawn
        n (int): The number of samples to draw

    Returns:
        np.array: The random samples
    """
    return data[np.random.choice(data.shape[0], n, replace=False)]


def assign_clusters(data: np.array, centroids: np.array):
    """Assigns datapoints to cluster centroids

    Args:
        data (np.array): The data to cluster
        centroids (np.array): The cluster centroids

    Returns:
        np.array: Array containing cluster assignments for all datapoints
    """
    return np.array([np.argmin([np.linalg.norm(point - centroid) for centroid in centroids]) for point in data])


def compute_fitness(data, clusters, centroids, k):
    """Computes the fitness of a clustering 
    based on Eq. 8 of Data Clustering Using particle swarm optimization

    Args:
        data (_type_): The data
        clusters (_type_): The cluster assignment of the data
        centroids (_type_): The centroids of the clusters
        k (_type_): The number of centroids (included for better performance)

    Returns:
        float: The quantization error of the clustering
    """
    return np.sum([np.sum([np.linalg.norm(z - centroids[j]) / len(data[np.where(clusters == j)])
                           for z in data[np.where(clusters == j)]])
                   for j in range(k)]) / k


def generate_data(means: np.array, cov: np.array) -> np.array:
    """Generates artificial data

    Args:
        means (np.array): Array of means
        cov (np.array): Covariance matrix

    Returns:
        np.array: Data drawn from bivariate Gaussian
    """
    data = []
    clusters = []
    for i, mu in enumerate(means):
        data.append(np.random.multivariate_normal(mu, cov, 150))
        clusters.append(np.full(150, i))
    return np.concatenate(np.array(data)), np.concatenate(clusters)

def get_iris_data(f_name: str):
    """Loads the iris dataset

    Args:
        f_name (str): Path to iris dataset

    Returns:
        (np.array, np.array): The dataset and the true clusters of the dataset
    """
    df = pd.read_csv('data/iris.data', header=None)
    data = df.values[:,:-1].astype(np.float32)
    clusters = np.unique(df.values[:,-1], return_inverse=True)[1].astype(int)
    return data, clusters
    


def plot_clusters(data, true, pred):
    """Plots a given clustering

    Args:
        data (_type_): The data
        clusters (_type_): Cluster assignments of the datapoints
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    for i in np.unique(true):
        true_cluster = data[np.where(true == i)[0]]
        pred_cluster = data[np.where(pred == i)[0]]
        ax1.scatter(true_cluster[:, 0], true_cluster[:, 1], s=15)
        ax2.scatter(pred_cluster[:, 0], pred_cluster[:, 1], s=15)

    ax1.set_xlabel("x")
    ax2.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("True clustering")
    ax2.set_title("Predicted clustering")

    plt.show()


def exercise_3() -> None:
    """Exercutes code for exercise 3
    """
    datasets = {}
    
    # Genrate artificial datasets
    means = [[-3, 0], [0, 0], [3, 0]]
    cov = [[0.50, 0.05], [0.05, 0.50]]
    data_1 = np.random.uniform(-1, 1, (400, 2))
    true_1 = np.array([1 if z1 >= 0.7 or (z1 <= 0.3 and z2 >= -0.2 - z1) else 0
                       for (z1, z2) in data_1])
    data_2, true_2 = generate_data(means, cov)
        
    # Get Iris dataset with true clustering
    iris, true_iris = get_iris_data('data/iris.data')
    
    
    # Perform k_means clustering on all datasets and plot results
    _, pred_1 = k_means(2, data_1, 100)
    _, pred_2 = k_means(3, data_2, 100)
    _, pred_iris = k_means(len(np.unique(true_iris)), iris, 100)
    plot_clusters(data_1, true_1, pred_1)
    plot_clusters(data_2, true_2, pred_2)
    plot_clusters(iris, true_iris, pred_iris)
    
    # Perform PSO clustering on both datasets and plot results
    _, pred_1 = pso(2, data_1, 10, 0.72, 1.49, 1.49, 100)
    _, pred_2 = pso(3, data_2, 10, 0.72, 1.49, 1.49, 100)
    _, pred_iris = pso(len(np.unique(true_iris)), iris, 10, 0.72, 1.49, 1.49, 100)
    plot_clusters(data_1, true_1, pred_1)
    plot_clusters(data_2, true_2, pred_2)
    plot_clusters(iris, true_iris, pred_iris)
    
    # For each dataset run the algorithms for 30 trials
    for name, (data, true) in zip(["artifical dataset 1", "artifical dataset 2", "iris dataset"],
                          [[data_1, true_1], [data_2, true_2], [iris, true_iris]]):
        kmeans_scores = []
        pso_scores = []
        
        k = len(np.unique(true))
        for i in range(30):
            print(f"Running trials for {name} {i}/30", flush=True, end='\r')
            centroids_pso, pred_pso = k_means(k, data, 100)
            centroids_k, pred_k = pso(k, data, 10, 0.72, 1.49, 1.49, 100)
            kmeans_scores.append(compute_fitness(data, pred_k, centroids_k, k))
            pso_scores.append(compute_fitness(data, pred_pso, centroids_pso, k))
        
        print(f"\nAverage fitness of kmeans on {name}: {np.average(np.array(kmeans_scores))}")
        print(f"Best fitness of kmeans on {name}: {np.min(np.array(kmeans_scores))}")
        print(f"Average fitness of PSO on {name}: {np.average(np.array(pso_scores))}")
        print(f"Best fitness of PSO on {name}: {np.min(np.array(pso_scores))}\n")

if __name__ == "__main__":
    exercise_3()

# %%
