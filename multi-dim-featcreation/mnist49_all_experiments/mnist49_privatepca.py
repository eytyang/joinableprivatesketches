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

def priv_pca_laplace(mat, num_iters, dim, eps = None):
	if eps is None:
		U, S, VT = np.linalg.svd(np.matmul(mat.T, mat))
		return None, None, U[:, :dim]
	sens = np.array(get_sens_list(mat)) + 0.001
	sens_matrix = np.tile(sens, (mat.shape[0], 1))
	mat = (np.divide(mat, 2 * sens_matrix) + 1.0) / 2

	laplace = np.random.laplace(scale = (3 * mat.shape[1] ** 2) / (mat.shape[0] * eps), size = (mat.shape[1], mat.shape[1]))
	
	for i in range(mat.shape[1]):
		for j in range(i + 1, mat.shape[1]):
			laplace[i][j] = laplace[j][i]
	for i in range(mat.shape[1]):
		laplace[i][i] *= 2

	cov = (1.0 / mat.shape[0]) * np.matmul(mat.T, mat) + laplace
	U, S, VT = np.linalg.svd(cov)
	return sens, np.mean(mat, axis = 0).reshape(-1, mat.shape[1]), U[:, :dim]

def get_sens_list(f_train):
	f_train_centered = f_train - np.mean(f_train, axis = 0).reshape(-1, f_train.shape[1])
	f_train_abs = np.absolute(f_train_centered)

	return [f_train_abs[:, i].max() for i in range(f_train.shape[1])] 

def get_loss(f_train, l_train, f_test, l_test, alg = 'Logistic Regression'):
	classifier = method_to_obj[alg]
	classifier.fit(f_train, l_train.to_numpy().reshape(l_train.size))
	pred = classifier.predict(f_test)
	return metrics.accuracy_score(l_test.to_numpy().reshape(l_test.size), pred) 

if __name__ == "__main__":
	num_trials = 25

	# Load MNIST dataset
	mnist = tf.keras.datasets.mnist
	(f_train, l_train), (f_test, l_test) = mnist.load_data()

	# Filter only 4s and 9s
	train_filter = np.logical_or(l_train == 4, l_train == 9)
	test_filter = np.logical_or(l_test == 4, l_test == 9)

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
	f_mean = np.mean(f_train, axis = 0, keepdims = True).reshape((-1, f_train.shape[1]))
	f_train = f_train - f_mean
	f_test = f_test - f_mean

	# Print the shape of the matrices
	print("f_train shape:", f_train.shape)
	print("f_test shape:", f_test.shape)
	print("l_train shape:", l_train.shape)
	print("l_test shape:", l_test.shape)

	sketch_dim = [1, 5, 10, 15, 20, 25]
	num_iters = 50
	eps_pca = 0.1
	total_eps_list = [1.0, 3.0, 5.0]
	algs = ['LogisticRegression', 'MultiLayerPerceptron']

	trial_dict = {}
	loss_dict = {}
	loss_ctrl = {}
	for alg in algs:
		loss_dict[alg] = {}
		loss_dict[alg]['Dimension'] = []
		loss_dict[alg]['PCA'] = []
		for total_eps in total_eps_list:
			loss_dict[alg]['Eps = %s' % str(total_eps)] = []
			loss_dict[alg]['Eps = %s 25' % str(total_eps)] = []
			loss_dict[alg]['Eps = %s 75' % str(total_eps)] = []
		loss_ctrl[alg] = get_loss(f_train, l_train, f_test, l_test, alg)
		loss_dict[alg]['Original Features'] = []

	print(loss_ctrl)

	for dim in sketch_dim:
		print('Dimension %i' % dim)

		sens, means, pca = priv_pca_laplace(f_train, num_iters, dim)
		f_train_pca = np.matmul(f_train, pca)
		f_test_pca = np.matmul(f_test, pca)
		for alg in algs:
			loss_dict[alg]['Dimension'].append(dim)
			loss_dict[alg]['PCA'].append(get_loss(f_train_pca, l_train, f_test_pca, l_test, alg))
			loss_dict[alg]['Original Features'].append(loss_ctrl[alg])

		for total_eps in total_eps_list:
			print('Total Eps = %s' % str(total_eps))
			# eps = total_eps - eps_pca
			eps_memb = 1000 # eps / (dim + 1)
			eps_val = total_eps # - eps_memb
			
			for alg in algs:
				trial_dict[alg] = []
			
			for trial in range(num_trials):
				print('Trial %i' % (trial + 1))
				sens, means, priv_pca = priv_pca_laplace(f_train, num_iters, dim, eps_pca)
				f_train_priv = np.matmul(0.5 * (np.divide(f_train, np.tile(sens, (f_train.shape[0], 1))) + 1.0) - means, priv_pca)
				f_test_priv = np.matmul(0.5 * (np.divide(f_test, np.tile(sens, (f_test.shape[0], 1))) + 1.0) - means, priv_pca)

				# perc05 = np.percentile(f_train_priv, 5, axis = 0 , keepdims = True)
				# perc95 = np.percentile(f_train_priv, 95, axis = 0 , keepdims = True)
				# f_train_priv = np.clip(f_train_priv, a_min = perc05, a_max = perc95)
				sens_list = get_sens_list(f_train_priv)
				f_train_priv = pd.DataFrame(data = f_train_priv, index = index_train, columns = ["Comp %i" % (i + 1) for i in range(dim)])
				
				dp_join = DP_Join(eps_memb, eps_val, sens_list, 'Real')
				dp_join.join(l_train, f_train_priv)
				# dp_join.features = np.clip(dp_join.features, a_min = perc05, a_max = perc95)
				# f_test_priv = np.clip(f_test_priv, a_min = perc05, a_max = perc95)

				for alg in algs:
					trial_dict[alg].append(get_loss(dp_join.features, dp_join.labels, f_test_priv, l_test, alg))

			for alg in algs:
				loss_dict[alg]['Eps = %s' % str(total_eps)].append(median(trial_dict[alg]))
				loss_dict[alg]['Eps = %s 25' % str(total_eps)].append(median(trial_dict[alg]) - np.percentile(trial_dict[alg], 25))
				loss_dict[alg]['Eps = %s 75' % str(total_eps)].append(np.percentile(trial_dict[alg], 75) - median(trial_dict[alg]))
		print()

	for alg in algs:
		alg_df = pd.DataFrame(loss_dict[alg])
		alg_df = alg_df.set_index('Dimension')
		alg_df = alg_df
		print(alg_df)

		file = 'mnist49_pcalaplace_smalldim_%s_trials=%i' % (alg.lower(), num_trials)
		alg_df.to_csv('%s.csv' % file)
		shift = -0.25
		plt.ylim((0.0, 1.0))
		plt.errorbar(alg_df.index + shift, alg_df['Original Features'], \
			yerr = np.zeros(shape = (2, len(alg_df))), label = 'Original Features')
		plt.errorbar(alg_df.index + shift, alg_df['PCA'], \
			yerr = np.zeros(shape = (2, len(alg_df))), label = 'PCA')
		shift += 0.05
		for total_eps in total_eps_list:
			plt.errorbar(alg_df.index + shift, alg_df['Eps = %s' % str(total_eps)], \
				yerr = alg_df[['Eps = %s 25' % str(total_eps), 'Eps = %s 75' % str(total_eps)]].to_numpy().T, label = 'Eps = %s' % str(total_eps))
			shift += 0.05

		plt.xlabel("Dimension")
		plt.ylabel("Accuracy")
		plt.legend(loc = "lower right")
		plt.savefig('%s.jpg' % file)
		plt.close()