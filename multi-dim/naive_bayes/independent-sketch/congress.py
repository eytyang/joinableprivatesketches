import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None  

from experiment import Experiment
from experiment_utils import center, prep_data, uniform_subsample, plot_results

import warnings
warnings.filterwarnings("ignore", message = "A column-vector y was passed when a 1d array was expected.")
warnings.filterwarnings("ignore", message = "X has feature names")

if __name__ == "__main__":
	num_trials = 25

	file = 'house-votes-84.data'
	l_name = ['party']
	experiment_list = ['Naive Bayes', 'Naive Bayes - Numerical Correction', 'Test Time Correction']
	f_train, l_train, f_test, l_test = prep_data(file, l_name)

	f_names = list(f_train.columns)
	l_train['party'] = l_train['party'].replace(['democrat', 'republican'], [1, -1])
	l_test['party'] = l_test['party'].replace(['democrat', 'republican'], [1, -1])

	f_train = f_train[f_train['el-salvador-aid'] == 1]
	f_names.remove('el-salvador-aid')
	f_train = f_train[f_names]
	f_test = f_test[f_test['el-salvador-aid'] == 1]
	f_test, l_test = f_test[f_names], l_test[l_name].loc[f_test.index]

	experiment = Experiment(experiment_list, f_train, l_train, f_test, l_test, f_names, l_name)
	loss_ctrl = experiment.get_loss()
	print(loss_ctrl)

	results_df = pd.DataFrame()

	eps_list = [2.5, 5.0, 7.5, 10.0, 12.5] 
	for eps in eps_list:
		print('Epsilon: %s' % str(eps))
		eps_memb = 1.0 # eps / (len(f_names) + 1)
		eps_val = eps - eps_memb

		loss_dict = experiment.run_dp_sketch_experiments(eps_memb, eps_val, num_trials)
		loss_dict['Epsilon'] = [eps]

		eps_df = pd.DataFrame(loss_dict).set_index('Epsilon')
		results_df = pd.concat([results_df, eps_df])
		print()

	results_df = results_df / loss_ctrl
	save_file = 'congress_trials=%i_epsm=1.0' % num_trials
	plot_results(results_df, save_file)