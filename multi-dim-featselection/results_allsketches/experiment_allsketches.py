import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  

from statistics import median
from dp_sketch import DP_Join
from sklearn import metrics 
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

import warnings
warnings.filterwarnings("ignore", message = "A column-vector y was passed when a 1d array was expected.")
warnings.filterwarnings("ignore", message = "X has feature names")

method_to_obj = {'Naive Bayes': BernoulliNB(),
				'Decision Tree': DecisionTreeClassifier(),
				'Logistic Regression': LogisticRegression(),
				'SVM': SVC(),
				'AdaBoost': AdaBoostClassifier()}

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
		self.df_dp = {}

	# Get control loss
	def get_loss(self, experiment_name, is_dp = False, reduced_features = 1, test_string = 'Ind_Same'):
		classifier = method_to_obj[experiment_name]

		if is_dp:
			df = self.df_dp[test_string].df
			# TODO: THIS IS HACKY; FIX
			if test_string == 'Ind_Same' or test_string == 'WeightedInd_Same' or test_string == 'Dep_Same':
				temp_f_names = [name for name in df.columns if (name not in self.l_name and name != 'membership' and name != 'sign')]
				classifier.fit(df[temp_f_names], df[self.l_name])
			else:
				temp_f_names = self.f_names
				classifier.fit(df[temp_f_names], df[self.l_name])
		else:
			temp_f_names = self.f_names
			classifier.fit(self.df_ctrl[self.f_names], self.df_ctrl[self.l_name])
		
		pred = classifier.predict(self.f_test[temp_f_names].to_numpy())
		return metrics.accuracy_score(self.l_test, pred) 

	# Run experiments on the joinable sketch for all sketch and feature types
	def run_dp_sketch_experiments(self, eps, reduced_features, num_trials = 25):
		eps_memb = eps / (reduced_features + 1)
		eps_val = eps - eps_memb

		trial_dict = {}
		loss_dict = {}
		for experiment_name in self.experiment_list:
			trial_dict[experiment_name] = {}
			loss_dict[experiment_name] = {}

		types = [('Ind', 'Same'), ('WeightedInd', 'Same'), ('Dep', 'Same'), ('Ind', 'Unif'), ('Dep', 'Unif'), ('Ind', 'NonUnif'), ('Dep', 'NonUnif')]
		for test in types:
			test_string = '%s_%s' % (test[0], test[1])
			print(test_string)
			for experiment_name in self.experiment_list:
				trial_dict[experiment_name][test_string] = []

			for trial in range(num_trials):
				self.df_dp[test_string] = DP_Join(eps_memb, eps_val)
				self.df_dp[test_string].join(self.l_train, self.f_train, reduced_features, test[0], test[1])
				self.df_dp[test_string].flip_labels(self.l_name[0])
				self.df_dp[test_string].df = self.df_dp[test_string].df.replace(0, 0.5)
				self.df_dp[test_string].df = self.df_dp[test_string].df.replace(-1, 0)
				
				for experiment_name in self.experiment_list:
					loss = self.get_loss(experiment_name, True, reduced_features, test_string)
					print('Trial Number %i, %s: %f' % (trial + 1, experiment_name, loss))
					trial_dict[experiment_name][test_string].append(loss)
			print()

		for experiment_name in self.experiment_list:
			for test in types:
				test_string = '%s_%s' % (test[0], test[1])
				loss_dict[experiment_name][test_string] = median(trial_dict[experiment_name][test_string])
				loss_dict[experiment_name][str(test_string) + ' 25'] = loss_dict[experiment_name][test_string] - np.percentile(trial_dict[experiment_name][test_string], 25)
				loss_dict[experiment_name][str(test_string) + ' 75'] = np.percentile(trial_dict[experiment_name][test_string], 75) - loss_dict[experiment_name][test_string]
		return loss_dict

