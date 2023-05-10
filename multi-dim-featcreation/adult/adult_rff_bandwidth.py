import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import median
from math import log

from dp_sketch import DP_Join

from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
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
	df = df.dropna()

	if f_names is None:
		f_names = list(df.columns)
		f_names.remove(l_name[0])
	categorical = df[f_names].select_dtypes(include=['object', 'bool']).columns
	df = pd.get_dummies(data = df, columns= categorical)

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

def get_rffs(mat, dim, bandwidth):
	omega = (2 ** (0.5)) * np.random.normal(0, 1, size = (mat.shape[1], dim)) / (bandwidth ** 2)
	beta = np.random.uniform(0, 2 * np.pi, dim).reshape(1, -1)
	return omega, beta, (2 ** (0.5)) * np.cos(np.matmul(mat, omega) + beta)

def get_loss(f_train, l_train, f_test, l_test, alg = 'Logistic Regression'):
	classifier = method_to_obj[alg]
	classifier.fit(f_train, l_train.replace(-1, 0).to_numpy().reshape(l_train.size))
	pred = classifier.predict(f_test)
	return metrics.accuracy_score(l_test.replace(-1, 0).to_numpy().reshape(l_test.size), pred)

if __name__ == "__main__":
	num_trials = 19

	file = '../data/adult.csv'
	l_name = ['income']
	f_train, l_train, f_test, l_test = prep_data(file, l_name)
	print(f_train.head(), l_train.head())

	f_names = f_train.columns
	print(f_names)
	
	f_test, l_test = f_test[f_names], l_test[l_name].loc[f_test.index]
	l_train = l_train.replace('<=50K', -1)
	l_train = l_train.replace('>50K', 1)
	l_test = l_test.replace('<=50K', -1)
	l_test = l_test.replace('>50K', 1)

	index_train = f_train.index
	f_train = f_train.to_numpy()
	f_test = f_test.to_numpy()

	sketch_dim = [150, 200, 250]
	bandwidth_list = [75, 100, 125]
	algs = ['Logistic Regression', 'AdaBoost']

	trial_dict = {}
	loss_dict = {}
	loss_ctrl = {}
	for alg in algs:
		loss_dict[alg] = {}
		loss_dict[alg]['Dimension'] = []
		for bandwidth in bandwidth_list:
			loss_dict[alg]['Bdwth = %s' % str(bandwidth)] = []
			loss_dict[alg]['Bdwth = %s 25' % str(bandwidth)] = []
			loss_dict[alg]['Bdwth = %s 75' % str(bandwidth)] = []
		loss_ctrl[alg] = get_loss(f_train, l_train, f_test, l_test, alg)

	print(loss_ctrl)
	
	for dim in sketch_dim:
		print('Dimension %i' % dim)
		for alg in algs:
			loss_dict[alg]['Dimension'].append(dim)

		for bandwidth in bandwidth_list:
			print("Bandwidth %i" % bandwidth)
			for alg in algs:
				trial_dict[alg] = []

			for trial in range(num_trials):
				print("Trial Number %i" % (trial + 1))
				omega, beta, f_train_rff = get_rffs(f_train, dim, bandwidth)
				f_test_rff = 2 ** (0.5) * np.cos(np.matmul(f_test, omega) + beta)

				# Make the features binary
				f_train_rff = rand_round(f_train_rff)
				f_test_rff = rand_round(f_test_rff)

				for alg in algs:
					trial_dict[alg].append(get_loss(f_train_rff, l_train, f_test_rff, l_test, alg))

			for alg in algs:
				loss_dict[alg]['Bdwth = %s' % str(bandwidth)].append(median(trial_dict[alg]))
				loss_dict[alg]['Bdwth = %s 25' % str(bandwidth)].append(median(trial_dict[alg]) - np.percentile(trial_dict[alg], 25))
				loss_dict[alg]['Bdwth = %s 75' % str(bandwidth)].append(np.percentile(trial_dict[alg], 75) - median(trial_dict[alg]))

	for alg in algs:
		alg_df = pd.DataFrame(loss_dict[alg])
		alg_df = alg_df.set_index('Dimension')
		alg_df = alg_df / loss_ctrl[alg]
		print(alg_df)

		file = 'adult_rffbinary_bandwidth_%s_trials=%i' % (alg.lower(), num_trials)
		alg_df.to_csv('%s.csv' % file)
		shift = -0.05
		shift += 0.01
		for bandwidth in bandwidth_list:
			plt.errorbar(alg_df.index + shift, alg_df['Bdwth = %s' % str(bandwidth)], \
				yerr = alg_df[['Bdwth = %s 25' % str(bandwidth), 'Bdwth = %s 75' % str(bandwidth)]].to_numpy().T, label = 'Bdwth = %s' % str(bandwidth))
			shift += 0.01

		plt.xlabel("Dimension")
		plt.ylabel("(Loss With Dim Reduction) / (Actual Loss)")
		plt.legend(loc = "lower right")
		plt.savefig('%s.jpg' % file)
		plt.close()