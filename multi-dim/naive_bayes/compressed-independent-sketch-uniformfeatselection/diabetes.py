import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from experiment import Experiment
from experiment_utils import center, prep_data, uniform_subsample, plot_results

import warnings
warnings.filterwarnings("ignore", message = "A column-vector y was passed when a 1d array was expected.")
warnings.filterwarnings("ignore", message = "X has feature names")

if __name__ == "__main__":
	num_trials = 25
	num_features = 14

	file = '../data/diabetes_binary_5050split.csv'
	l_name = ['Diabetes_binary']
	experiment_list = ['Naive Bayes', 'Naive Bayes - Numerical Correction']
	f_train, l_train, f_test, l_test = prep_data(file, l_name)

	f_names = list(f_train.columns)
	f_names.remove('BMI')
	f_names.remove('GenHlth')
	f_names.remove('MentHlth')
	f_names.remove('PhysHlth')
	f_names.remove('Age')
	f_names.remove('Education')
	f_names.remove('Income')

	f_train = f_train[f_names]

	# TODO: WHY DO THIS?
	# f_test = f_test[f_test['favicons'] == -1]
	f_test, l_test = f_test[f_names], l_test[l_name].loc[f_test.index]
	print(l_test[l_name].value_counts())
	print(len(f_train), len(l_train))

	f_train = f_train.replace(0.0, -1)
	l_train = l_train.replace(0.0, -1)
	f_test = f_test.replace(0.0, -1)
	l_test = l_test.replace(0.0, -1)

	experiment = Experiment(experiment_list, f_train, l_train, f_test, l_test, f_names, l_name)
	loss_ctrl = experiment.get_loss(len(f_names))
	print(loss_ctrl)

	results_df = pd.DataFrame()

	eps_list = [0.25, 0.5]
	for eps in eps_list:
		print('Epsilon: %s' % str(eps))
		eps_memb = eps / (num_features + 1)
		eps_val = eps - eps_memb

		loss_dict = experiment.run_dp_sketch_experiments(eps_memb, eps_val, num_features, num_trials)
		loss_dict['Epsilon'] = [eps]

		eps_df = pd.DataFrame(loss_dict).set_index('Epsilon')
		results_df = pd.concat([results_df, eps_df])
		print()

	results_df = results_df / loss_ctrl
	save_file = 'diabetes_5050split_trials=%i_epsm=even_minisculeeps_feat=%i' % (num_trials, num_features)
	plot_results(results_df, experiment_list, save_file)