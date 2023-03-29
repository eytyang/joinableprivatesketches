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
	num_trials = 10

	file = '../data/phishing.csv'
	l_name = ['result']
	experiment_list = ['AdaBoost', 'AdaBoost - Numerical Correction']
	f_train, l_train, f_test, l_test = prep_data(file, l_name)

	f_names = list(f_train.columns)
	f_names.remove('length_of_url')
	f_names.remove('sub_domains')
	f_names.remove('sfh-domain')
	f_names.remove('web_traffic')
	f_names.remove('links_pointing')

	f_train = f_train[f_train['favicons'] == -1]
	f_names.remove('favicons')
	f_train = f_train[f_names]
	f_test = f_test[f_test['favicons'] == -1]
	f_test, l_test = f_test[f_names], l_test[l_name].loc[f_test.index]

	experiment = Experiment(experiment_list, f_train, l_train, f_test, l_test, f_names, l_name)
	loss_ctrl = experiment.get_loss()
	print(loss_ctrl)

	results_df = pd.DataFrame()

	eps_list = [8.0, 12.0, 16.0, 20.0, 24.0]
	for eps in eps_list:
		print('Epsilon: %s' % str(eps))
		eps_memb = eps / (1 + len(f_names))
		eps_val = eps - eps_memb

		loss_dict = experiment.run_dp_sketch_experiments(eps_memb, eps_val, num_trials)
		loss_dict['Epsilon'] = [eps]

		eps_df = pd.DataFrame(loss_dict).set_index('Epsilon')
		results_df = pd.concat([results_df, eps_df])
		print()

	results_df = results_df / loss_ctrl
	save_file = 'phishing_trials=%i_epsm=even_5boost_hugeeps' % num_trials
	plot_results(results_df, experiment_list, save_file)