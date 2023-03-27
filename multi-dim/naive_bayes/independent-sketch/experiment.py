import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  

from statistics import median
from dp_sketch import DP_Join
from sklearn import metrics 
from naive_bayes import NB
from naive_bayes_adjusted import AdjustedNB
from test_time_correction import TestTimeCorrection

import copy
import warnings
warnings.filterwarnings("ignore", message = "A column-vector y was passed when a 1d array was expected.")
warnings.filterwarnings("ignore", message = "X has feature names")

control_experiment = 'Naive Bayes'

method_to_obj = {'Naive Bayes': NB(),
	'Naive Bayes - Numerical Correction': AdjustedNB(), 
	'Test Time Correction': TestTimeCorrection(NB())}

class Experiment:
	def __init__(self, experiment_list, f_train, l_train, f_test, l_test, f_names, l_name):
		self.experiment_list = experiment_list
		self.f_train = f_train
		self.l_train = l_train
		self.f_test = f_test.replace(-1, 0)
		self.l_test = l_test.replace(-1, 0)
		self.f_names = f_names
		self.l_name = l_name 
		self.df_ctrl = f_train.join(l_train, how = 'inner').replace(-1, 0)
		# print('Control', len(self.df_ctrl[self.df_ctrl[l_name[0]] == 1]), len(self.df_ctrl[self.df_ctrl[l_name[0]] == 0]))
		self.df_dp = None
		self.df_dp_unfiltered = None

	# Get control loss
	def get_loss(self, experiment = control_experiment, params = {}, is_dp = False):
		classifier = method_to_obj[experiment]

		if is_dp and experiment == 'Test Time Correction':
			classifier.fit(self.df_dp_unfiltered.df[self.f_names], self.df_dp_unfiltered.df[self.l_name], **params)
		elif is_dp and experiment != control_experiment:
			classifier.fit(self.df_dp.df[self.f_names], self.df_dp.df[self.l_name], **params)
		elif is_dp and experiment == control_experiment:
			classifier.fit(self.df_dp.df[self.f_names], self.df_dp.df[self.l_name])
		else:
			classifier.fit(self.df_ctrl[self.f_names], self.df_ctrl[self.l_name])
		
		pred = classifier.predict(self.f_test.to_numpy())
		return metrics.accuracy_score(self.l_test, pred) 

	# Run any experiments on the joinable sketch
	def run_dp_sketch_experiments(self, eps_memb, eps_val, num_trials = 25):
		trial_dict = {}
		for experiment in self.experiment_list:
			trial_dict[experiment] = []

		for trial in range(num_trials):
			print('Trial Number %i' % (trial + 1))
			# print(self.l_test.to_numpy().flatten())

			self.df_dp = DP_Join(eps_memb, eps_val)
			self.df_dp.join(self.l_train, self.f_train)
			self.df_dp_unfiltered = copy.copy(self.df_dp)

			self.df_dp.drop_entries()
			self.df_dp_unfiltered.flip_labels(self.l_name[0])
			self.df_dp.df = self.df_dp.df.replace(-1, 0)
			self.df_dp_unfiltered.df = self.df_dp_unfiltered.df.replace(-1, 0)

			params = {'full_labels': self.l_train.replace(-1, 0), 
				'eps_memb': eps_memb,
				'eps_val': eps_val} 
			for experiment in self.experiment_list:
				print(self.get_loss(experiment, params, True))
				trial_dict[experiment].append(self.get_loss(experiment, params, True))

		loss_dict = {}
		for experiment in self.experiment_list:
			loss_dict[experiment] = median(trial_dict[experiment])
			loss_dict[experiment + ' 25'] = loss_dict[experiment] - np.percentile(trial_dict[experiment], 25)
			loss_dict[experiment + ' 75'] = np.percentile(trial_dict[experiment], 75) - loss_dict[experiment]
		return loss_dict

