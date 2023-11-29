
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.spatial.distance import cdist

def initialize_random_centroids(X, k):
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def assign_to_clusters(X, centroids):
    distances = cdist(X, centroids)
    return np.argmin(distances, axis=1)

def update_centroids(X, clusters, k):
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        new_centroids[i] = np.mean(X[clusters == i], axis=0)
    return new_centroids

def kmeans_algorithm(X, k, max_iterations=100):
    centroids = initialize_random_centroids(X, k)
    for _ in range(max_iterations):
        clusters = assign_to_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters

def plot_clusters(X, clusters, centroids):
    colors = ['r', 'g', 'b', 'cyan']
    plt.figure()
    for i in range(len(clusters)):
        plt.scatter(X[i, 0], X[i, 1], c=colors[clusters[i]])
    plt.scatter(centroids[:, 0], centroids[:, 1], c='y', marker='X')
    plt.show()

# Generate sample data
number_of_samples = 200
number_of_clusters = 3
number_of_iterations = 10
number_of_features = 2
X, _ = make_blobs(n_samples=number_of_samples, centers=number_of_clusters, cluster_std=0.60, random_state=0)

# Convert data to TensorFlow tensors
X_tensor = tf.constant(X, dtype=tf.float32)

# K-means algorithm using TensorFlow and Keras
kmeans_model = keras.Sequential([
    keras.layers.Input(shape=(number_of_features,)),
    keras.layers.Dense(number_of_clusters, activation='softmax')
])

kmeans_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
kmeans_model.fit(X_tensor, np.zeros(number_of_samples), epochs=number_of_iterations, verbose=0)

centroids_tensor = kmeans_model.get_layer(index=0).get_weights()[0].T
clusters_tensor = np.argmax(kmeans_model.predict(X_tensor), axis=1)

# Plot the clusters
plot_clusters(X, clusters_tensor, centroids_tensor)
