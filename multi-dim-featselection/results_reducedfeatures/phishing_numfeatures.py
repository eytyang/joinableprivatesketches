import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from experiment_numfeatures import Experiment
from experiment_numfeatures_utils import center, prep_data, uniform_subsample, plot_results

import warnings
warnings.filterwarnings("ignore", message = "A column-vector y was passed when a 1d array was expected.")
warnings.filterwarnings("ignore", message = "X has feature names")

if __name__ == "__main__":
	num_trials = 15

	file = 'data/phishing.csv'
	l_name = ['result']
	experiment_list = ['Naive Bayes', 'Decision Tree', 'Logistic Regression', 'AdaBoost']
	f_train, l_train, f_test, l_test = prep_data(file, l_name)

	f_names = list(f_train.columns)
	f_names.remove('length_of_url')
	f_names.remove('sub_domains')
	f_names.remove('sfh-domain')
	f_names.remove('web_traffic')
	f_names.remove('links_pointing')

	# f_train = f_train[f_train['favicons'] == -1]
	# l_train = l_train.loc[f_train.index]
	f_names.remove('favicons')
	f_train = f_train[f_names]

	# f_test = f_test[f_test['favicons'] == -1]
	f_test, l_test = f_test[f_names], l_test[l_name].loc[f_test.index]
	# print(l_test[l_name].value_counts())
	# print(len(f_train), len(l_train))

	f_train = f_train.replace(0, -1)
	l_train = l_train.replace(0, -1)
	f_test = f_test.replace(0, -1)
	l_test = l_test.replace(0, -1)

	experiment = Experiment(experiment_list, f_train, l_train, f_test, l_test, f_names, l_name)

	results = {}
	for experiment_name in experiment_list:
		results[experiment_name] = pd.DataFrame()

	eps_list = [5.0, 10.0, 15.0, 20.0]
	reduced_features_list = [8] # [1, 8, 15, 22]
	for eps in eps_list:
		print('Epsilon: %s' % str(eps))

		loss_dict = experiment.run_dp_sketch_experiments(eps, reduced_features_list, num_trials)
		for experiment_name in experiment_list:
			loss_dict[experiment_name]['Epsilon'] = [eps]
			eps_df = pd.DataFrame(loss_dict[experiment_name]).set_index('Epsilon')
			results[experiment_name] = pd.concat([results[experiment_name], eps_df])
		print()

	for experiment_name in experiment_list:
		loss_ctrl = experiment.get_loss(experiment_name)
		results[experiment_name] = results[experiment_name] / loss_ctrl
		save_file = 'phishingnohall_%s_trials=%i_largeeps' % (experiment_name.strip(' ').lower(), num_trials)
		plot_results(results[experiment_name], save_file, reduced_features_list, len(f_names))