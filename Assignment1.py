#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:07:52 2024

@author: bettyhan
"""
#####  Question 1  ####
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
import numpy as np
from sklearn.decomposition import IncrementalPCA
# 1. Retrieve and load the mnist_784 dataset of 70,000 instances

path = "/Users/bettyhan/School/Term6/UnsupervisedLearning"

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X=mnist["data"]
y=mnist["target"]

print(mnist.data.shape)
# 2. Display each digit. 
# Display first 25 digits  

plt.figure(figsize=(10, 8))
for i in range(25):
  # Display original
  some_digit = X[i]
  some_digit_image = some_digit.reshape(28, 28)
  ax = plt.subplot(5, 5, i + 1)
  plt.imshow(some_digit_image,cmap='gray', interpolation='nearest')

#3.Use PCA to retrieve the principal component and output their explained variance ratio. [5 points]

pca=PCA()
X_reduced =pca.fit_transform(X)
pc1=pca.components_.T[:,0]
pc2=pca.components_.T[:,1]
print("1st", pc1)
print("2nd", pc2)
print("explained variance ratio", pca.explained_variance_ratio_[0:2])
#4. Plot the projections of the principal component onto a 1D hyperplane. [5 points]
# Create scatter plot
plt.scatter(pc1, np.zeros_like(pc1))
plt.title("1st plot")
plt.xlabel("1st component")
plt.show()
plt.scatter(pc2, np.zeros_like(pc2))
plt.title("2nd plot")
plt.xlabel("2nd component")
plt.show()

#5. Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions. [10 points]
inc_pca = IncrementalPCA(n_components=154)
X_reduced_new= inc_pca.fit_transform(X)
X_reduced_new.shape

#6. Display the original and compressed digits from (5). [5 points]
n = 10 # number of images to display
for i in range(10):
    some_digit = X[i]
    some_digit_image = some_digit.reshape(28, 28)
    reconstructed = inc_pca.inverse_transform(X_reduced_new[i])
    
    # Display original    
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(some_digit_image,cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Display compressed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed.reshape(28, 28),cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
###### Question 2: #######
#1. Generate Swiss roll dataset.
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
X, t = make_swiss_roll(n_samples=3000,noise=0.1, random_state=42)
X.shape
# 2. Plot the resulting generated Swiss roll dataset.
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap='Spectral', s=5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Swiss Roll Dataset')
plt.show()
#3. Use Kernel PCA (kPCA) with linear kernel (2 points), a RBF kernel (2 points), and a sigmoid kernel (2 points).
linear_pca= KernelPCA(n_components=2, kernel="linear")
X_reduced_linearKernel = linear_pca.fit_transform(X)

rbf_pca= KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
X_reduced_rbfKernel = rbf_pca.fit_transform(X)

sigmoid_pca= KernelPCA(n_components=2, kernel="sigmoid", gamma=0.001)
X_reduced_sigmoidKernel = sigmoid_pca.fit_transform(X)
#4.Plot the kPCA results
plt.scatter(X_reduced_linearKernel[:,0],X_reduced_linearKernel[:,1],c=t,cmap='Spectral', s=5)
plt.show()
plt.scatter(X_reduced_rbfKernel[:,0],X_reduced_rbfKernel[:,1],c=t,cmap='Spectral', s=5)
plt.show()
plt.scatter(X_reduced_sigmoidKernel[:,0],X_reduced_sigmoidKernel[:,1],c=t,cmap='Spectral', s=5)
plt.show()
#5. Using kPCA and a kernel of your choice, apply Logistic Regression for classification. Use GridSearchCV to find the best kernel and gamma value for kPCA in order to get the best classification accuracy at the end of the pipeline. Print out best parameters found by GridSearchCV

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression())
    ])
param_grid = [{
    "kpca__gamma":np.linspace(0.03, 0.1,10),
    "kpca__kernel": ["rbf", "sigmoid"]    
    }]

# Example labels (numeric)
labels = t.reshape(-1, 1)

# Initialize the MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Fit and transform the labels
labels_normalized = scaler.fit_transform(labels)
labels_normalized.flatten()
bins = np.linspace(0, 1, num=6)  # Create 5 bins
categorical_labels = np.digitize(labels_normalized, bins) - 1  # Convert to integer labels
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3)
grid_search.fit(X,categorical_labels)

print(grid_search.best_params_)
# save best pipeline and model
best_pipeline = grid_search.best_estimator_
best_kpca= best_pipeline.named_steps['kpca']
X_reduced_best = best_kpca.transform(X)
#Plot the results from using GridSearchCV 
plt.scatter(X_reduced_best[:,0],X_reduced_best[:,1],c=categorical_labels,cmap='Spectral', s=5)
plt.show()