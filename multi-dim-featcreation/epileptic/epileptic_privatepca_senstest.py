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

def prep_data(file, l_name, index_name = None, f_names = None, test_size = 0.2, center_data = False):
	# Load dataset 
	if index_name is not None:
		df = pd.read_csv(file).set_index(index_name)
	else:
		df = pd.read_csv(file)

	if f_names is None:
		f_names = list(df.columns)
		f_names.remove(l_name[0])
	if center_data:
		df = center(center(df, f_names), l_name)

	df_train, df_test = train_test_split(df, test_size = test_size)
	f_train, l_train = df_train[f_names], df_train[l_name]

	if center_data:
		f_test, l_test = center(df_test[f_names], f_names), center(df_test[l_name], l_name)
	else:
		f_test, l_test = df_test[f_names], df_test[l_name]
	return f_train, l_train, f_test, l_test

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

	file = '../../data/epileptic.csv'
	l_name = ['y']
	f_train, l_train, f_test, l_test = prep_data(file, l_name)
	f_names = list(f_train.columns)
	f_names.remove('Unnamed: 0')
	index_train = f_train.index

	print(l_train.value_counts())
	print(l_test.value_counts())
	
	f_train, l_train = f_train[f_names], l_train[l_name].loc[f_train.index]
	f_test, l_test = f_test[f_names], l_test[l_name].loc[f_test.index]
	l_train = l_train.replace(2, -1)
	l_train = l_train.replace(3, -1)
	l_train = l_train.replace(4, -1)
	l_train = l_train.replace(5, -1)
	l_test = l_test.replace(2, -1)
	l_test = l_test.replace(3, -1)
	l_test = l_test.replace(4, -1)
	l_test = l_test.replace(5, -1)

	f_train = f_train.to_numpy()
	f_test = f_test.to_numpy()

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
	print(np.percentile(f_train_pca, 99, axis = 0))
	print(np.percentile(f_train_pca, 1, axis = 0))
	print([f_train_pca[:, i].min() for i in range(f_train_pca.shape[1])])
	print()

	f_train_pca = f_train_pca - np.mean(f_train_pca, axis = 0).reshape(-1, f_train_pca.shape[1])
	print('Centered:')
	print([f_train_pca[:, i].max() for i in range(f_train_pca.shape[1])])
	print(np.percentile(f_train_pca, 99, axis = 0))
	print(np.percentile(f_train_pca, 1, axis = 0))
	print([f_train_pca[:, i].min() for i in range(f_train_pca.shape[1])])
	print()


