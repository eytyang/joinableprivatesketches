import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import median
from math import log

from dp_sketch import DP_Join
from hrw import get_features

from sklearn import metrics 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

method_to_obj = {'Naive Bayes': GaussianNB(),
				'Decision Tree': DecisionTreeClassifier(),
				'Logistic Regression': LogisticRegression(),
				'SVM': SVC(),
				'AdaBoost': AdaBoostClassifier(), 
				'Random Forest': RandomForestClassifier(), 
				'Perceptron': Perceptron(), 
				'kNN': KNeighborsClassifier(), 
				'Gradient Boosting': GradientBoostingClassifier()}

def prep_data(file, l_name, index_name = None, f_names = None, test_size = 0.2, center_data = False):
	# Load dataset 
	if index_name is not None:
		df = pd.read_csv(file).set_index(index_name)
	else:
		df = pd.read_csv(file)
	df = df.set_index('id')

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
	return (binary_mat * 2 * 2 ** (0.5)) - 2 ** (0.5)

def get_loss(f_train, l_train, f_test, l_test, alg = 'Logistic Regression'):
	classifier = method_to_obj[alg]
	classifier.fit(f_train, l_train.to_numpy().reshape(l_train.size))
	pred = classifier.predict(f_test)
	return metrics.accuracy_score(l_test.to_numpy().reshape(l_test.size), pred)

if __name__ == "__main__":
	num_trials = 15
	bandwidth = 20

	file = '../data/breastcancer.csv'
	l_name = ['diagnosis']
	f_train, l_train, f_test, l_test = prep_data(file, l_name)

	f_names = list(f_train.columns)
	print(f_names)
	f_names.remove('Unnamed: 32')

	print(l_train.value_counts())
	print(l_test.value_counts())
	
	f_train, l_train = f_train[f_names], l_train[l_name].loc[f_train.index]
	f_test, l_test = f_test[f_names], l_test[l_name].loc[f_test.index]
	l_train = l_train.replace('B', -1)
	l_train = l_train.replace('M', 1)
	l_test = l_test.replace('B', -1)
	l_test = l_test.replace('M', 1)

	index_train = f_train.index
	f_train = f_train.to_numpy()
	f_test = f_test.to_numpy()

	sketch_dim = [5, 10, 15, 20, 25]
	total_eps_list = [2.0, 4.0, 6.0, 8.0, 10.0]
	algs = ['Random Forest', 'AdaBoost']

	trial_dict = {}
	loss_dict = {}
	loss_ctrl = {}
	for alg in algs:
		loss_dict[alg] = {}
		loss_dict[alg]['Dimension'] = []
		loss_dict[alg]['RFF HRW'] = []
		loss_dict[alg]['RFF HRW 25'] = []
		loss_dict[alg]['RFF HRW 75'] = []
		for total_eps in total_eps_list:
			loss_dict[alg]['Eps = %s' % str(total_eps)] = []
			loss_dict[alg]['Eps = %s 25' % str(total_eps)] = []
			loss_dict[alg]['Eps = %s 75' % str(total_eps)] = []
		loss_ctrl[alg] = get_loss(f_train, l_train, f_test, l_test, alg)
		loss_dict[alg]['Original Features'] = []
	print(loss_ctrl)
	
	for dim in sketch_dim:
		print('Dimension %i' % dim)

		# TODO: Optimize this later. 
		for alg in algs:
			trial_dict[alg] = {}
			trial_dict[alg]['RFF HRW'] = []
			for total_eps in total_eps_list:
				trial_dict[alg]['Eps = %s' % str(total_eps)] = []
			
		for trial in range(num_trials):
			print('Trial %i' % (trial + 1))
			f_train_rff = get_features(f_train, f_train, dim, bandwidth)
			print('Got Training RFFs')
			f_test_rff = get_features(f_test, f_test, dim, bandwidth)
			print('Got Test RFFs')

			for alg in algs:
				trial_dict[alg]['RFF HRW'].append(get_loss(f_train_rff, l_train, f_test_rff, l_test, alg))

			f_train_df = pd.DataFrame(data = f_train, index = index_train, columns = ["Feat %i" % (i + 1) for i in range(f_train.shape[1])])
			sens_list = [2 ** (0.5) for i in range(dim)]
			for total_eps in total_eps_list:
				print('Total Eps = %s' % str(total_eps))
				eps_memb = 10000 # total_eps / (dim + 1)
				eps_val = total_eps # - eps_memb

				dp_join = DP_Join(eps_memb, eps_val, sens_list) 
				dp_join.join(l_train, f_train_df, 'HRW', dim, bandwidth) 

				for alg in algs:
					trial_dict[alg]['Eps = %s' % total_eps].append(get_loss(dp_join.features, dp_join.labels, f_test_rff, l_test, alg))

		for alg in algs:
			loss_dict[alg]['Dimension'].append(dim)
			loss_dict[alg]['Original Features'].append(loss_ctrl[alg])
			loss_dict[alg]['RFF HRW'].append(median(trial_dict[alg]['RFF HRW']))
			loss_dict[alg]['RFF HRW 25'].append(median(trial_dict[alg]['RFF HRW']) - np.percentile(trial_dict[alg]['RFF HRW'], 25))
			loss_dict[alg]['RFF HRW 75'].append(np.percentile(trial_dict[alg]['RFF HRW'], 75) - median(trial_dict[alg]['RFF HRW']))
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

		file = 'breastcancer_rffhrw_%s_trials=%i' % (alg.lower(), num_trials)
		alg_df.to_csv('%s.csv' % file)
		shift = -0.3
		plt.ylim((0.0, 1.0))
		plt.errorbar(alg_df.index + shift, alg_df['Original Features'], \
			yerr = np.zeros(shape = (2, len(alg_df))), label = 'Original Features')
		plt.errorbar(alg_df.index + shift, alg_df['RFF HRW'], \
			yerr = alg_df[['RFF HRW 25', 'RFF HRW 75']].to_numpy().T, label = 'RFF HRW')
		shift += 0.1
		for total_eps in total_eps_list:
			plt.errorbar(alg_df.index + shift, alg_df['Eps = %s' % str(total_eps)], \
				yerr = alg_df[['Eps = %s 25' % str(total_eps), 'Eps = %s 75' % str(total_eps)]].to_numpy().T, label = 'Eps = %s' % str(total_eps))
			shift += 0.1

		plt.xlabel("Dimension")
		plt.ylabel("(Loss With Dim Reduction) / (Actual Loss)")
		plt.legend(loc = "lower right")
		plt.savefig('%s.jpg' % file)
		plt.close()