import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# Load dataset from .npy file
def load_data(points_file):
    return np.load(points_file)

# Replace with your dataset file path
points_file = 'processed-point.npy'
X = load_data(points_file)

# Standardize the data (optional but recommended)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Grid Search for Clusters from 2 to 500
k_range = range(2, 501)
kmeans_silhouette = []
agglo_silhouette = []

for k in k_range:
    print(f"Processing {k} clusters...")
    
    # KMeans Clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X)
    kmeans_silhouette.append(silhouette_score(X, labels_kmeans))
    
    # Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters=k)
    labels_agglo = agglo.fit_predict(X)
    agglo_silhouette.append(silhouette_score(X, labels_agglo))

# Plot Silhouette Scores
plt.figure(figsize=(10, 5))
plt.plot(k_range, kmeans_silhouette, label='KMeans', linestyle='--', marker='o')
plt.plot(k_range, agglo_silhouette, label='Agglomerative', linestyle='--', marker='s', color='r')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Grid Search (KMeans vs Agglomerative)')
plt.legend()
plt.show()
