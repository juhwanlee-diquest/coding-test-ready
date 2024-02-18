from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

# Adjusted example dataset creation function for clustering
def create_example_dataset_for_clustering(num_samples=1000):
    np.random.seed(0)  # Ensure reproducibility
    data = {
        'feature1': np.random.rand(num_samples),
        'feature2': np.random.rand(num_samples),
        'feature3': np.random.rand(num_samples)
    }
    df = pd.DataFrame(data)
    return df

# Create the dataset for clustering
df = create_example_dataset_for_clustering()

# Define the features
X = df[['feature1', 'feature2', 'feature3']]  # Features

# Initialize the KMeans clustering with k=3 clusters (you can change k as needed)
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit KMeans to the data
kmeans.fit(X)

# Get the cluster centroids
centroids = kmeans.cluster_centers_

# Get the cluster labels for each data point
cluster_labels = kmeans.labels_

# Add cluster labels to the DataFrame
df['cluster_label'] = cluster_labels

print("Cluster Centroids:")
print(centroids)
print("\nCluster Labels:")
print(df['cluster_label'].value_counts())
