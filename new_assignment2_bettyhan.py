#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:32:18 2024

@author: bettyhan
"""

#1. Retrieve and load the Olivetti faces dataset
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,KFold
import matplotlib.pyplot as plt
olivetti_faces = fetch_olivetti_faces()
olivetti_faces.data.shape

X = olivetti_faces.data  
y = olivetti_faces.target  

X.shape
num_classes = len(np.unique(y))

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for ax, image, label in zip(axes.ravel(), X, y):
    ax.imshow(image.reshape(64, 64), cmap='gray')
    ax.set_title(f'Person {label}')
    ax.axis('off')
plt.show()


#2. Split the training set, a validation set, and a test set using stratified sampling
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
X_train.shape
#3. Using k-fold cross validation, train a classifier to predict which person is represented in each picture, and evaluate it on the validation set

model = LogisticRegression(max_iter=1000)

# # K-Fold Cross Validation

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=kf,scoring='accuracy')
print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {np.mean(scores)}")

model.fit(X_train, y_train)
val_predictions = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)

# # Output validation set accuracy
print(f"Validation Set Accuracy: {val_accuracy}")

## plt the prediction on Val dataset
n_images = 10  # Number of images to display
indices = np.random.choice(len(X_val), n_images, replace=False)

plt.figure(figsize=(15, 6))
for i, idx in enumerate(indices):
    ax = plt.subplot(2, 5, i + 1)
    ax.imshow(X_val[idx].reshape(64, 64), cmap='gray')
    ax.set_title(f'True: {y_val[idx]}\nPred: {val_predictions[idx]}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# 4.Use K-Means to reduce the dimensionality of the set. 
# Find optimal number of clusters using silhouette score
silhouette_scores = []
cluster_range = range(2, 99)

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_train)
    silhouette_avg = silhouette_score(X_train, labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title('Silhouette Scores for K-Means Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(cluster_range)
plt.grid()
plt.show()

# Optimal number of clusters
optimal_n_clusters = 97
print(f"Optimal number of clusters: {optimal_n_clusters}")
kmeans82 = KMeans(n_clusters=optimal_n_clusters, random_state=42)

X_New_train=kmeans82.fit_transform(X_train)

X_val_k = kmeans82.transform(X_val)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_New_train, y_train)

# Make predictions
y_pred = classifier.predict(X_val_k)

# Check the score
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)


#5. from sklearn.cluster import DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
dbscan = DBSCAN(eps=0.5, min_samples=2)
X_scaled = StandardScaler().fit_transform(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_pca.shape
dbscan.fit(X_pca)

clusters = dbscan.fit_predict(X_pca)
dbscan.labels_


#Plot the results 
def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)
    # Obtain core instances from dbscan.components_
    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]
    
    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20,
                c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1],
                c=dbscan.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("$x_1$")
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title(f"eps={dbscan.eps:.2f}, min_samples={dbscan.min_samples}")
    plt.grid()
    plt.gca().set_axisbelow(True)
 
dbscan2 = DBSCAN(eps=0.05,min_samples=2)
dbscan2.fit(X_pca)
clusters2 = dbscan2.fit_predict(X_pca)
fig = plt.figure(figsize=(9, 3.2))
plt.subplot(121)
plot_dbscan(dbscan, X_scaled, size=100)
plt.subplot(122)
plot_dbscan(dbscan2, X_scaled, size=600,show_ylabels=False)
plt.show()
unique_clusters = np.unique(clusters)

len(unique_clusters)
