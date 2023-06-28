from sklearnex import patch_sklearn 
patch_sklearn()

import numpy as np
import scipy as sc
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

def round_threshold(f):
	return 0.5 * (1.0 - f / (2 ** 0.5))

def rand_round(mat):
	round_threshold_vec = np.vectorize(round_threshold)
	thres = round_threshold_vec(mat)
	rand_mat = np.random.uniform(size = (mat.shape[0], mat.shape[1]))
	binary_mat = rand_mat > thres 
	return (binary_mat * 2 * (2 ** (0.5))) - 2 ** (0.5)

def get_rffs(mat, dim, bandwidth):
	omega = (2 ** (0.5)) * np.random.normal(loc = 0, scale = 1.0 / bandwidth, size = (mat.shape[1], dim))
	beta = np.random.uniform(0, 2 * np.pi, dim).reshape(1, -1)
	return omega, beta, (2 ** (0.5)) * np.cos(np.matmul(mat, omega) + beta)

def get_loss(f_train, l_train, f_test, l_test, alg = 'LogisticRegression'):
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
	subsample9_train = all_train[all_train['label'] == 9].sample(n = int(len(all_train[all_train['label'] == 9]) / 5))
	index9_train = subsample9_train.index
	index_train = index4_train.union(index9_train)
	f_train = all_train.loc[index_train].drop('label', axis = 1).to_numpy()
	l_train_ctrl = all_train['label'].loc[index_train]
	l_train = all_train['label'].to_frame()
	
	index4_test = all_test.index[all_test['label'] == 4] 
	subsample9_test = all_test[all_test['label'] == 9].sample(n = int(len(all_test[all_test['label'] == 9]) / 5))
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

	# Compute bandwidth
	# pair_dists = sc.spatial.distance.pdist(f_train)
	# bandwidth = np.median(pair_dists)
	bandwidth = 10

	sketch_dim = [5, 10, 15, 20, 25]
	total_eps_list = [1.0, 2.0, 3.0, 4.0, 5.0]
	# algs = ['KNN', 'RandomForest']
	algs = ['AdaBoost', 'LogisticRegression', 'MultiLayerPerceptron']

	trial_dict = {}
	loss_dict = {}
	loss_ctrl = {}
	for alg in algs:
		loss_dict[alg] = {}
		loss_dict[alg]['Dimension'] = []
		loss_dict[alg]['RFF Real'] = []
		loss_dict[alg]['RFF Real 25'] = []
		loss_dict[alg]['RFF Real 75'] = []
		for total_eps in total_eps_list:
			loss_dict[alg]['Eps = %s' % str(total_eps)] = []
			loss_dict[alg]['Eps = %s 25' % str(total_eps)] = []
			loss_dict[alg]['Eps = %s 75' % str(total_eps)] = []
		loss_ctrl[alg] = get_loss(f_train, l_train_ctrl, f_test, l_test, alg)
		loss_dict[alg]['Original Features'] = []
	print(loss_ctrl)
	
	for dim in sketch_dim:
		print('Dimension %i' % dim)

		# TODO: Optimize this later. 
		for alg in algs:
			trial_dict[alg] = {}
			trial_dict[alg]['RFF Real'] = []
			for total_eps in total_eps_list:
				trial_dict[alg]['Eps = %s' % str(total_eps)] = []
			
		for trial in range(num_trials):
			print('Trial %i' % (trial + 1))
			omega, beta, f_train_rff = get_rffs(f_train, dim, bandwidth)
			f_test_rff = 2 ** (0.5) * np.cos(np.matmul(f_test, omega) + beta)

			for alg in algs:
				trial_dict[alg]['RFF Real'].append(get_loss(f_train_rff, l_train_ctrl, f_test_rff, l_test, alg))

			f_train_rff = pd.DataFrame(data = f_train_rff, index = index_train, columns = ["Feat %i" % (i + 1) for i in range(dim)])
			sens_list = [2 ** (0.5) for i in range(dim)]
			for total_eps in total_eps_list:
				print('Total Eps = %s' % str(total_eps))
				eps_memb = total_eps / (dim + 1)
				eps_val = total_eps - eps_memb

				dp_join = DP_Join(eps_memb, eps_val, sens_list) 
				dp_join.join(l_train, f_train_rff, 'Real Clip') 

				for alg in algs:
					trial_dict[alg]['Eps = %s' % total_eps].append(get_loss(dp_join.features, dp_join.labels, f_test_rff, l_test, alg))

		for alg in algs:
			loss_dict[alg]['Dimension'].append(dim)
			loss_dict[alg]['Original Features'].append(loss_ctrl[alg])
			loss_dict[alg]['RFF Real'].append(median(trial_dict[alg]['RFF Real']))
			loss_dict[alg]['RFF Real 25'].append(median(trial_dict[alg]['RFF Real']) - np.percentile(trial_dict[alg]['RFF Real'], 25))
			loss_dict[alg]['RFF Real 75'].append(np.percentile(trial_dict[alg]['RFF Real'], 75) - median(trial_dict[alg]['RFF Real']))
			for total_eps in total_eps_list:
				loss_dict[alg]['Eps = %s' % str(total_eps)].append(median(trial_dict[alg]['Eps = %s' % str(total_eps)]))
				loss_dict[alg]['Eps = %s 25' % str(total_eps)].append(median(trial_dict[alg]['Eps = %s' % str(total_eps)]) - np.percentile(trial_dict[alg]['Eps = %s' % str(total_eps)], 25))
				loss_dict[alg]['Eps = %s 75' % str(total_eps)].append(np.percentile(trial_dict[alg]['Eps = %s' % str(total_eps)], 75) - median(trial_dict[alg]['Eps = %s' % str(total_eps)]))
		print()

	for alg in algs:
		alg_df = pd.DataFrame(loss_dict[alg])
		alg_df = alg_df.set_index('Dimension')
		alg_df = alg_df 
		print(alg_df)

		file = 'mnist49join_rffrealclip_smalleps_%s_trials=%i' % (alg.lower(), num_trials)
		alg_df.to_csv('%s.csv' % file)
		shift = -0.25
		plt.ylim((0.0, 1.0))
		plt.errorbar(alg_df.index + shift, alg_df['Original Features'], \
			yerr = np.zeros(shape = (2, len(alg_df))), label = 'Original Features')
		plt.errorbar(alg_df.index + shift, alg_df['RFF Real'], \
			yerr = alg_df[['RFF Real 25', 'RFF Real 75']].to_numpy().T, label = 'RFF Real')
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
