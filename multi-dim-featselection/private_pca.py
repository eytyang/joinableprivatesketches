import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import median
from math import log

from sklearn import metrics 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

method_to_obj = {'Naive Bayes': GaussianNB(),
				'Decision Tree': DecisionTreeClassifier(),
				'Logistic Regression': LogisticRegression(),
				'SVM': SVC(),
				'AdaBoost': AdaBoostClassifier(), 
				'Random Forest': RandomForestClassifier()}

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

def priv_power_method(mat, num_iters, dim, eps, delta = 0.0001): 
	cov = np.matmul(mat.T, mat)
	print(cov)
	X = get_random_orthonormal(cov.shape[1], dim)
	sigma = (1.0 / eps) * ((4 * dim * num_iters * log(1.0 / delta)) ** 0.5)
	print(sigma)

	for i in range(num_iters):
		sens = np.absolute(X).max()
		print(sens * sigma)
		Y = np.matmul(cov, X) + np.random.normal(scale = (sens * sigma) ** 2, size = (cov.shape[0], dim))
		X, R = np.linalg.qr(Y)
	print(X)
	return X

def get_loss(f_train, l_train, f_test, l_test, alg = 'Logistic Regression'):
	classifier = method_to_obj[alg]
	classifier.fit(f_train, l_train)
	pred = classifier.predict(f_test)
	return metrics.accuracy_score(l_test, pred) 

if __name__ == "__main__":
	num_trials = 15
	bandwidth = 150

	file = 'data/covtype.csv'
	l_name = ['Cover_Type']
	f_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', \
		'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', \
		'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
	f_train, l_train, f_test, l_test = prep_data(file, l_name)

	f_train = f_train[f_names]
	l_train = l_train[(l_train['Cover_Type'] == 6) | (l_train['Cover_Type'] == 7)]
	f_train = f_train.loc[l_train.index]
	l_test = l_test[(l_test['Cover_Type'] == 6) | (l_test['Cover_Type'] == 7)]
	f_test = f_test.loc[l_test.index]
	print(l_train.value_counts())
	print(l_test.value_counts())
	
	f_test, l_test = f_test[f_names], l_test[l_name].loc[f_test.index]
	l_train = l_train.replace(6, 0)
	l_train = l_train.replace(7, 1)
	l_test = l_test.replace(6, 0)
	l_test = l_test.replace(7, 1)

	f_train = f_train.to_numpy()
	f_test = f_test.to_numpy()
	l_train = l_train.to_numpy().reshape(l_train.size)
	l_test = l_test.to_numpy().reshape(l_test.size)

	# sketch_dim = [2, 3, 4, 5]
	dim = 4
	eps = 1.0
	num_iters_list = [25, 50, 75, 100]
	algs = ['Naive Bayes', 'Logistic Regression', 'SVM', 'AdaBoost']

	trial_dict = {}
	loss_dict = {}
	loss_ctrl = {}
	for alg in algs:
		loss_dict[alg] = {}
		loss_dict[alg]['Num Iters'] = []
		loss_dict[alg]['PCA'] = []
		loss_dict[alg]['Private PCA'] = []
		loss_dict[alg]['Private PCA 25'] = []
		loss_dict[alg]['Private PCA 75'] = []
		loss_ctrl[alg] = get_loss(f_train, l_train, f_test, l_test, alg)

	print(loss_ctrl)

	for num_iters in num_iters_list:
		print('Num Iters %i' % num_iters)
		pca = PCA(n_components = dim)
		f_train_pca = pca.fit_transform(f_train)
		f_test_pca = pca.transform(f_test)

		for alg in algs:
			trial_dict[alg] = []
		
		for trial in range(num_trials):
			print('Trial %i' % (trial + 1))
			priv_pca = priv_power_method(f_train, num_iters, dim, eps)
			f_train_priv = np.matmul(f_train, priv_pca)
			f_test_priv = np.matmul(f_test, priv_pca)

			for alg in algs:
				trial_dict[alg].append(get_loss(f_train_priv, l_train, f_test_priv, l_test))

		for alg in algs:
			loss_dict[alg]['Num Iters'].append(num_iters)
			loss_dict[alg]['PCA'].append(get_loss(f_train_pca, l_train, f_test_pca, l_test))
			loss_dict[alg]['Private PCA'].append(median(trial_dict[alg]))
			loss_dict[alg]['Private PCA 25'].append(median(trial_dict[alg]) - np.percentile(trial_dict[alg], 25))
			loss_dict[alg]['Private PCA 75'].append(np.percentile(trial_dict[alg], 75) - median(trial_dict[alg]))
		print()

	for alg in algs:
		alg_df = pd.DataFrame(loss_dict[alg])
		alg_df = alg_df.set_index('Num Iters')
		alg_df = alg_df / loss_ctrl[alg]
		print(alg_df)

		file = 'covtype_pca_%s_trials=%i_eps=%s' % (alg.lower(), num_trials, str(eps))
		alg_df.to_csv('%s.csv' % file)
		shift = -0.02
		plt.errorbar(alg_df.index + shift, alg_df['PCA'], \
			yerr = np.zeros(shape = (2, len(alg_df))), label = 'PCA')
		shift += 0.01
		plt.errorbar(alg_df.index + shift, alg_df['Private PCA'], \
			yerr = alg_df[['Private PCA 25', 'Private PCA 75']].to_numpy().T, label = 'Private PCA')

		plt.xlabel("Dimension")
		plt.ylabel("(Loss With Dim Reduction) / (Actual Loss)")
		plt.legend(loc = "lower right")
		plt.savefig('%s.jpg' % file)
		plt.close()