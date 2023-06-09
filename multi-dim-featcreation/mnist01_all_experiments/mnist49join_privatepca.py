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
	all_train = pd.DataFrame(f_train)
	all_train['label'] = l_train
	all_test = pd.DataFrame(f_test)
	all_test['label'] = l_test
	
	# Bias the training and test sets
	index4_train = all_train.index[all_train['label'] == 4] 
	subsample9_train = all_train[all_train['label'] == 9].sample(n = int(len(all_train) / 5))
	index9_train = subsample9_train.index
	index_train = index4_train.union(index9_train)
	f_train = all_train.loc[index_train].drop('label', axis = 1).to_numpy()
	l_train_ctrl = all_train['label'].loc[index_train]
	l_train = all_train['label'].to_frame()

	index4_test = all_test.index[all_test['label'] == 4] 
	subsample9_test = all_test[all_test['label'] == 9].sample(n = int(len(all_test) / 5))
	index9_test = subsample9_test.index
	index_test = index4_test.union(index9_test)
	f_test = all_test.loc[index_test].drop('label', axis = 1).to_numpy()
	l_test = all_test['label'].loc[index_test]

	l_train_ctrl = l_train_ctrl.replace(4, -1)
	l_train_ctrl = l_train_ctrl.replace(9, 1)
	l_train = l_train.replace(4, -1)
	l_train = l_train.replace(9, 1)
	l_test = l_test.replace(4, -1)
	l_test = l_test.replace(9, 1)

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
		loss_ctrl[alg] = get_loss(f_train, l_train_ctrl, f_test, l_test, alg)
		loss_dict[alg]['Original Features'] = []

	print(loss_ctrl)

	for dim in sketch_dim:
		print('Dimension %i' % dim)

		pca = priv_power_method(f_train, num_iters, dim)
		f_train_pca = np.matmul(f_train, pca)
		f_test_pca = np.matmul(f_test, pca)
		for alg in algs:
			loss_dict[alg]['Dimension'].append(dim)
			loss_dict[alg]['PCA'].append(get_loss(f_train_pca, l_train_ctrl, f_test_pca, l_test, alg))
			loss_dict[alg]['Original Features'].append(loss_ctrl[alg])

		for total_eps in total_eps_list:
			print('Total Eps = %s' % str(total_eps))
			# eps = total_eps - eps_pca
			eps_memb = total_eps / (dim + 1)
			eps_val = total_eps - eps_memb
			
			for alg in algs:
				trial_dict[alg] = []
			
			for trial in range(num_trials):
				print('Trial %i' % (trial + 1))
				priv_pca = priv_power_method(f_train, num_iters, dim, eps_pca)
				f_train_priv = np.matmul(f_train, priv_pca)
				f_test_priv = np.matmul(f_test, priv_pca)

				sens_list = get_sens_list(f_train_priv)
				f_train_priv = pd.DataFrame(data = f_train_priv, index = index_train, columns = ["Comp %i" % (i + 1) for i in range(dim)])
				dp_join = DP_Join(eps_memb, eps_val, sens_list, 'Real')
				dp_join.join(l_train, f_train_priv)

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

		file = 'mnist49join_pca_%s_trials=%i' % (alg.lower(), num_trials)
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