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
	mat_mean = np.mean(mat, axis = 0, keepdims = True)
	mat = mat - mat_mean
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

	file = '../../data/epileptic.csv'
	l_name = ['y']
	f_train, l_train, f_test, l_test = prep_data(file, l_name)
	f_names = list(f_train.columns)
	f_names.remove('Unnamed: 0')
	index_train = f_train.index

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
	print(l_train.value_counts())
	print(l_test.value_counts())
	
	# Bias the training and test sets
	indexpos_train = l_train.index[l_train['y'] == 1] 
	subsampleneg_train = l_train[l_train['y'] == -1].sample(n = int(len(l_train[l_train['y'] == -1]) / 4))
	indexneg_train = subsampleneg_train.index
	index_train = indexpos_train.union(indexneg_train)
	f_train = f_train.loc[index_train].to_numpy()
	l_train_ctrl = l_train.loc[index_train]

	indexpos_test = l_test.index[l_test['y'] == 1] 
	subsampleneg_test = l_test[l_test['y'] == -1].sample(n = int(len(l_test[l_test['y'] == -1]) / 4))
	indexneg_test = subsampleneg_test.index
	index_test = indexpos_test.union(indexneg_test)
	f_test = f_test.loc[index_test].to_numpy()
	l_test = l_test.loc[index_test]

	sketch_dim = [5, 10, 15, 20, 25]
	num_iters = 50
	eps_pca = 1000 # 0.1
	total_eps_list = [1.0, 2.0, 3.0, 4.0, 5.0]
	algs = ['AdaBoost', 'RandomForest', 'KNN']

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
				mean_train = np.mean(f_train_priv, axis = 0, keepdims = True)
				f_train_priv = f_train_priv - mean_train

				sens_list = get_sens_list(f_train_priv)
				f_train_priv = pd.DataFrame(data = f_train_priv, index = index_train, columns = ["Comp %i" % (i + 1) for i in range(dim)])
				dp_join = DP_Join(eps_memb, eps_val, sens_list, 'Real')
				dp_join.join(l_train, f_train_priv)
				dp_join.features = dp_join.features + mean_train

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

		file = 'epilepticjoin_pca_%s_trials=%i' % (alg.lower(), num_trials)
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