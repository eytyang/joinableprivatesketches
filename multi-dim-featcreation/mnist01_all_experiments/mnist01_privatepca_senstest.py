from sklearnex import patch_sklearn 
patch_sklearn()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import median
from math import log

import sys
sys.path.append('../')
from dp_sketch import DP_Join

import tensorflow as tf
from sklearn import metrics 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

method_to_obj = {'NaiveBayes': GaussianNB(),
				'DecisionTree': DecisionTreeClassifier(),
				'LogisticRegression': LogisticRegression(),
				'SVM': SVC(),
				'AdaBoost': AdaBoostClassifier(), 
				'RandomForest': RandomForestClassifier(n_jobs = 2),
				'MultiLayerPerceptron': MLPClassifier(),
				'KNN': KNeighborsClassifier(n_jobs = 2)}

def get_random_orthonormal(vec_dim, num_vecs):
	rand_mat = np.random.normal(size = (vec_dim, num_vecs))
	Q, R = np.linalg.qr(rand_mat)
	return Q

def priv_power_method(mat, num_iters, dim, eps = None, delta = 0.0001): 
	cov = np.matmul(mat.T, mat)
	X = get_random_orthonormal(cov.shape[1], dim)
	if eps is not None:
		sigma = (1.0 / eps) * ((4 * dim * num_iters * log(1.0 / delta)) ** 0.5)

	for i in range(num_iters):
		sens = np.absolute(X).max()
		if eps is None:
			Y = np.matmul(cov, X)
		else:
			Y = np.matmul(cov, X) + np.random.normal(scale = (sens * sigma) ** 2, size = (cov.shape[0], dim))
		X, R = np.linalg.qr(Y)
	return X

def get_sens_list(f_train):
	f_train_abs = np.absolute(f_train)
	return [f_train_abs[:, i].max() for i in range(f_train.shape[1])] 

if __name__ == "__main__":
	num_trials = 25

	# Load MNIST dataset
	mnist = tf.keras.datasets.mnist
	(f_train, l_train), (f_test, l_test) = mnist.load_data()

	# Filter only 4s and 9s
	train_filter = np.logical_or(l_train == 0, l_train == 1)
	test_filter = np.logical_or(l_test == 0, l_test == 1)

	f_train = f_train[train_filter]
	l_train = l_train[train_filter]
	f_test = f_test[test_filter]
	l_test = l_test[test_filter]

	# Flatten the image data
	f_train = f_train.reshape((-1, 28 * 28))
	f_test = f_test.reshape((-1, 28 * 28))

	# Convert pixel values to float32 and scale them between 0 and 1
	f_train = f_train.astype(np.float32) / 255.0
	f_test = f_test.astype(np.float32) / 255.0

	# Create pandas DataFrames
	f_train = pd.DataFrame(f_train)
	l_train = pd.DataFrame(l_train, index = f_train.index, columns = ['label'])
	f_test = pd.DataFrame(f_test)
	l_test = pd.DataFrame(l_test, index = f_test.index, columns = ['label'])
	index_train = f_train.index
	f_train = f_train.to_numpy()
	f_test = f_test.to_numpy()

	# Print the shape of the matrices
	print("f_train shape:", f_train.shape)
	print("f_test shape:", f_test.shape)
	print("l_train shape:", l_train.shape)
	print("l_test shape:", l_test.shape)

	sketch_dim = [10, 20, 30, 40, 50]
	num_iters = 50
	eps_pca = 1000 # 0.1
	total_eps_list = [1.0, 2.0, 3.0, 4.0, 5.0]
	algs = ['LogisticRegression', 'AdaBoost', 'RandomForest', 'MultiLayerPerceptron', 'KNN']

	pca = priv_power_method(f_train, num_iters, 30)
	f_train_pca = np.matmul(f_train, pca)
		
	sens_list = get_sens_list(f_train_pca)
	print('Uncentered:')
	print([f_train_pca[:, i].max() for i in range(f_train_pca.shape[1])])
	print(np.percentile(f_train_pca, 95, axis = 0))
	print(np.percentile(f_train_pca, 5, axis = 0))
	print([f_train_pca[:, i].min() for i in range(f_train_pca.shape[1])])
	print()

	f_train_pca = f_train_pca - np.mean(f_train_pca, axis = 0).reshape(-1, f_train_pca.shape[1])
	print('Centered:')
	print([f_train_pca[:, i].max() for i in range(f_train_pca.shape[1])])
	print(np.percentile(f_train_pca, 95, axis = 0))
	print(np.percentile(f_train_pca, 5, axis = 0))
	print([f_train_pca[:, i].min() for i in range(f_train_pca.shape[1])])
	print()


