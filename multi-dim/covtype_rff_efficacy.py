import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import median

from sklearn import metrics 
from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from experiment_allsketches_utils import prep_data

method_to_obj = {'Naive Bayes': GaussianNB(),
				'Decision Tree': DecisionTreeClassifier(),
				'Logistic Regression': LogisticRegression(),
				'SVM': SVC(),
				'AdaBoost': AdaBoostClassifier(), 
				'Random Forest': RandomForestClassifier()}

def get_rffs(mat, dim):
	omega = 2 ** (0.5) * np.random.normal(size = (mat.shape[1], dim))
	beta = np.random.uniform(0, 2 * np.pi, dim).reshape(1, -1)
	return omega, beta, 2 ** (0.5) * np.cos(np.matmul(mat, omega) + beta)

def get_loss(f_train, l_train, f_test, l_test, alg = 'Logistic Regression'):
	classifier = method_to_obj[alg]
	classifier.fit(f_train, l_train)
	pred = classifier.predict(f_test)
	return metrics.accuracy_score(l_test, pred) 

if __name__ == "__main__":
	num_trials = 15

	file = 'data/covtype.csv'
	l_name = ['Cover_Type']
	f_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', \
		'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', \
		'Vertical_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', \
		'Horizontal_Distance_To_Fire_Points']
	f_train, l_train, f_test, l_test = prep_data(file, l_name)

	f_train = f_train[f_names]
	l_train = l_train[l_train['Cover_Type'] in [6, 7]]
	f_train = f_train.loc[f_train.index]
	print(l_train.value_counts())
	print(l_test.value_counts())
	
	f_test, l_test = f_test[f_names], l_test[l_name].loc[f_test.index]
	l_train = l_train.replace(6, 0)
	l_train = l_train.replace(7, 1)
	l_test = l_test.replace(6, 0)
	l_test = l_test.replace(7, 1)

	sketch_dim = [2, 3, 4, 5]
	algs = ['SVM'] # ['Naive Bayes', 'Logistic Regression', 'SVM', 'AdaBoost']

	trial_dict = {}
	loss_dict = {}
	loss_ctrl = {}
	for alg in algs:
		loss_dict[alg] = {}
		loss_dict[alg]['Epsilon'] = []
		loss_dict[alg]['PCA'] = []
		loss_dict[alg]['RFFs'] = []
		loss_dict[alg]['RFFs 25'] = []
		loss_dict[alg]['RFFs 75'] = []
		loss_ctrl[alg] = get_loss(f_train.to_numpy(), l_train.to_numpy(), f_test.to_numpy(), l_test.to_numpy(), alg)

	for dim in sketch_dim:
		print('Sketch Dimension %i' % dim)
		pca = PCA(n_components = dim)
		f_train_pca = pca.fit_transform(f_train.to_numpy())
		f_test_pca = pca.transform(f_test.to_numpy())
		print(f_train_pca.shape)

		for alg in algs:
			trial_dict[alg] = []
		
		for trial in range(num_trials):
			print('Trial %i' % (trial + 1))
			# TODO: Maybe create a fit_transform and transform function for RFFs too.
			omega, beta, f_train_rff = get_rffs(f_train.to_numpy(), dim)
			f_test_rff = 2 ** (0.5) * np.cos(np.matmul(f_test.to_numpy(), omega) + beta)
			print(f_train_rff.shape)

			for alg in algs:
				trial_dict[alg].append(get_loss(f_train_rff, l_train.to_numpy(), f_test_rff, l_test.to_numpy()))
				print(trial_dict[alg][-1])	

		for alg in algs:
			loss_dict[alg]['Epsilon'].append(dim)
			loss_dict[alg]['PCA'].append(get_loss(f_train_pca, l_train.to_numpy(), f_test_pca, l_test.to_numpy()))
			loss_dict[alg]['RFFs'].append(median(trial_dict[alg]))
			loss_dict[alg]['RFFs 25'].append(median(trial_dict[alg]) - np.percentile(trial_dict[alg], 25))
			loss_dict[alg]['RFFs 75'].append(np.percentile(trial_dict[alg], 75) - median(trial_dict[alg]))
		print()

	for alg in algs:
		alg_df = pd.DataFrame(loss_dict[alg])
		alg_df = alg_df.set_index('Epsilon')
		alg_df = alg_df / loss_ctrl[alg]
		print(alg_df)

		file = 'adult_checkRFF_%s_trials=%i' % (alg.lower(), num_trials)
		alg_df.to_csv('%s.csv' % file)
		shift = -0.02
		plt.errorbar(alg_df.index + shift, alg_df['PCA'], \
			yerr = np.zeros(shape = (2, len(alg_df))), label = 'PCA')
		shift += 0.01
		plt.errorbar(alg_df.index + shift, alg_df['RFFs'], \
			yerr = alg_df[['RFFs 25', 'RFFs 75']].to_numpy().T, label = 'RFFs')

		plt.xlabel("Epsilon")
		plt.ylabel("(Loss With DP) / (Actual Loss)")
		plt.legend(loc = "lower right")
		plt.savefig('%s.jpg' % file)
		plt.close()
