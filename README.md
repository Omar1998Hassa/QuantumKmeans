#Quantum K-means Clustering Algorithm

## Overview
This repository contains Python code implementing the K-means clustering algorithm along with quantum-inspired data encoding for clustering. K-means is an unsupervised machine learning algorithm used for partitioning a dataset into K distinct, non-overlapping clusters. The algorithm aims to minimize the within-cluster variance, assigning each data point to the nearest centroid.

## Contents
- `k_means.py`: Python script containing functions for performing K-means clustering using quantum encoding and fidelity between evolved states of data points and the centroids.
- `README.md`: Documentation providing an overview of the repository and instructions for usage.

## Dependencies
- `numpy`: Library for numerical computing in Python.
- `matplotlib`: Library for creating visualizations in Python.
- `qiskit`: Quantum computing framework for Python.
- `scikit-learn`: Machine learning library for Python.

## Usage
1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/your_username/k-means-clustering.git
2. Install the required dependencies using pip.
   ```bash
   pip install -r requirements.txt
3. Run the k_means.py script to perform K-means clustering on your dataset.
   ```bash
    python k_means.py
4. Modify the dataset and parameters in the script as needed for your specific application.   

## Example Usage

```python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from k_means import k_means

# Generate sample data
data = np.random.randn(100, 2)

# Perform K-means clustering with K=3
centroids, cluster_idx, sil_score = k_means(data, 3)

# Plot the clustered data
plt.scatter(data[:, 0], data[:, 1], c=cluster_idx)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

print(f"Silhouette Score: {sil_score}")

   
