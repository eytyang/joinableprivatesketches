import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  

from math import exp, log
from statistics import median
from dp_sketch import DP_Join
from sklearn import metrics 
from stump import Stump

class AdaBoost:
	def __init__(self, n_estimators = 10, count_minimum = 1):
		self.minimum = 0.0
		self.n_estimators = n_estimators
		self.hall_prob_memb = None
		self.flip_prob_val = None
		self.stump_list = []
		self.stump_weights = []

	# features and labels come from the noisy join
	# full_labels is the set of full labels. 
	def fit(self, features, labels):
		f_names = features.columns
		l_name = labels.columns[0]
		self.stump_list = []
		self.stump_weights = []

		# Initialize all weights to 1.0
		features['weight'] = pd.Series(np.ones(len(features)), index = features.index)
		labels['weight'] = pd.Series(np.ones(len(features)), index = labels.index)
		
		for t in range(self.n_estimators):
			# Train an adjusted weak learner on the weighted data
			weak_learner = Stump()
			weak_learner.fit(features, labels)
			# print(weak_learner.stump, weak_learner.choice)

			# Compute the error of the weak learner and the weight multiplier
			labels['pred'] = pd.Series(weak_learner.predict(features[f_names]), index = labels.index)
			err = labels[labels[l_name] != labels['pred']]['weight'].sum() / labels['weight'].sum()
			if err < 0.01:
				err = 0.01
			weight_multiplier = (1 - err) / err
			labels['weight'] *= np.exp((labels[l_name] + labels['pred']) % 2 * log(weight_multiplier))

			# Normalize the weights, propagate weights to the features DataFrame as well
			normalizer = labels['weight'].sum() / len(labels)
			labels['weight'] = labels['weight'] / normalizer
			features['weight'] = labels['weight']

			self.stump_list.append(weak_learner)
			self.stump_weights.append(log(weight_multiplier))

	# Returns the prediction of a single instance of features
	def get_prediction(self, f_vec):
		score = [0.0, 0.0]
		# Follows the AdaBoost algorithm
		for i in range(len(self.stump_list)):
			pred = self.stump_list[i].predict_array(f_vec)
			score[pred] += self.stump_weights[i]
		if score[0] > score[1]:
			return 0
		return 1

	# Returns prediction for an array of features, corresponding to multiple test instances
	def predict(self, f_test):
		pred = []
		for f_vec in f_test:
			pred.append(self.get_prediction(f_vec))
		return pred
