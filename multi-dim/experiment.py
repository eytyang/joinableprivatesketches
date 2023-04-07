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
	def get_loss(self, experiment_name, is_dp = False, reduced_features = 1, dp_type = 'Ind_Unif'):
		classifier = method_to_obj[experiment_name]

		if is_dp:
			df = self.df_dp[reduced_features].df
			classifier.fit(df[self.f_names], df[self.l_name])
		else:
			classifier.fit(self.df_ctrl[self.f_names], self.df_ctrl[self.l_name])
		
		pred = classifier.predict(self.f_test.to_numpy())
		return metrics.accuracy_score(self.l_test, pred) 

	# Run experiments on the joinable sketch for all sketch and feature types
	def run_dp_sketch_experiments(self, eps, reduced_features_list, num_trials = 25):
		trial_dict = {}
		loss_dict = {}
		for experiment_name in self.experiment_list:
			trial_dict[experiment_name] = {}
			loss_dict[experiment_name] = {}

		for reduced_features in reduced_features_list:
			print("Number of Features (Reduced): %i" % reduced_features)
			for experiment_name in self.experiment_list:
				trial_dict[experiment_name][reduced_features] = []

			for trial in range(num_trials):
				self.df_dp[reduced_features] = DP_Join(eps / (reduced_features + 1), eps - eps / (reduced_features + 1))
				self.df_dp[reduced_features].join(self.l_train, self.f_train, reduced_features, 'Dep', 'NonUnif')
				self.df_dp[reduced_features].flip_labels(self.l_name[0])
				self.df_dp[reduced_features].df = self.df_dp[reduced_features].df.replace(-1, 0)
				
				for experiment_name in self.experiment_list:
					loss = self.get_loss(experiment_name, True, reduced_features, 'Dep_NonUnif')
					print('Trial Number %i, %s: %f' % (trial + 1, experiment_name, loss))
					trial_dict[experiment_name][reduced_features].append(loss)
			print()

		for experiment_name in self.experiment_list:
			for reduced_features in reduced_features_list:
				loss_dict[experiment_name][reduced_features] = median(trial_dict[experiment_name][reduced_features])
				loss_dict[experiment_name][str(reduced_features) + ' 25'] = loss_dict[experiment_name][reduced_features] - np.percentile(trial_dict[experiment_name][reduced_features], 25)
				loss_dict[experiment_name][str(reduced_features) + ' 75'] = np.percentile(trial_dict[experiment_name][reduced_features], 75) - loss_dict[experiment_name][reduced_features]
		return loss_dict

