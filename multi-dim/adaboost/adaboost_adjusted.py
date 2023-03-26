import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  

from math import exp, log
from statistics import median
from dp_sketch import DP_Join
from sklearn import metrics 
from stump_adjusted import AdjustedStump

class AdjustedAdaBoost:
	def __init__(self, n_estimators = 50, count_minimum = 1):
		self.minimum = 0.0
		self.n_estimators = n_estimators
		self.hall_prob_memb = None
		self.flip_prob_val = None
		self.label_halls = {}
		self.feat_label_halls = {}
		self.stump_list = []
		self.stump_weights = []

	# Estimate how many rows of each label or (feature, label) pair are hallucinated
	# Initialize label_halls and feat_label_halls assuming weights 1.0 throughout. 
	def est_hall_counts(self, f_names, labels, eps_memb, full_labels):
		l_name = labels.columns[0]

		for i in range(len(f_names)):
			self.feat_label_halls[i] = np.zeros((2, 2))

		self.hall_prob_memb = 1.0 / (1.0 + exp(eps_memb))
		# Count how many rows have each label in both labels and full_labels
		label_count = labels[l_name].value_counts()
		full_label_count = full_labels[l_name].value_counts()

		for l_val in [0, 1]:
			# Estimate the number of hallucinations with each label
			# If we estimate a negative quantity, default to 2 * self.minimum
			self.label_halls[l_val] = max(self.hall_prob_memb * (label_count[l_val] - \
										(1 - self.hall_prob_memb) * full_label_count[l_val]) / \
										(2 * self.hall_prob_memb - 1), 2 * self.minimum)

			# Estimate the number of hallucinations with each (feature, label) combination
			for i in range(len(f_names)):
				for f_val in [0, 1]:
					# Divide by two since the features are uniformly distributed
					self.feat_label_halls[i][f_val][l_val] = self.label_halls[l_val] / 2.0

	# Update estimates of hallucination weights based on the weight multiplier
	def update_hall_counts(self, f_names, labels, weak_classifier, weight_multiplier, normalizer):
		weight_multiplier *= 0.5

		# Case: the stump does not classify based on a feature
		if weak_classifier.stump is None:
			# Update the weights of the misclassified rows 
			self.label_halls[1 - weak_classifier.choice] *= weight_multiplier

			# Update weights of (feature, label) pairs only for the misclassified rows
			for i in range(len(f_names)):
				for f_val in [0, 1]:
					self.feat_label_halls[i][f_val][1 - weak_classifier.choice] *= weight_multiplier
		# Case: the stump does classify based on feature weak_classifier.stump
		else:
			for l_val in [0, 1]:
				# Update weights of each label. 
				# In expectation, half should be classified correctly and half should be misclassified
				self.label_halls[l_val] *= (1 + weight_multiplier) / 2.0
				for i in range(len(f_names)):
					for f_val in [0, 1]:
						if i == weak_classifier.stump and l_val != weak_classifier.choice[f_val]:
							self.feat_label_halls[i][f_val][l_val] *= weight_multiplier
						elif i != weak_classifier.stump:
							self.feat_label_halls[i][f_val][l_val] *= (1 + weight_multiplier) / 2.0

		# Apply the normalizer:
		for l_val in [0, 1]:
			self.label_halls[l_val] /= (normalizer)
			for i in range(len(f_names)):
				for f_val in [0, 1]:
					self.feat_label_halls[i][f_val][l_val] /= (normalizer)

	# features and labels come from the noisy join
	# full_labels is the set of full labels. 
	def fit(self, features, labels, eps_memb, eps_val, full_labels):
		f_names = features.columns
		l_name = labels.columns[0]
		self.est_hall_counts(f_names, labels, eps_memb, full_labels)
		self.flip_prob_val = exp(-1.0 * eps_val / len(features.columns)) / 2.0
		self.stump_list = []
		self.stump_weights = []

		# Initialize all weights to 1.0
		features['weight'] = pd.Series(np.ones(len(features)), index = features.index)
		labels['weight'] = pd.Series(np.ones(len(features)), index = labels.index)
		
		for t in range(self.n_estimators):
			# Train an adjusted weak learner on the weighted data
			weak_learner = AdjustedStump(self.label_halls, self.feat_label_halls)
			weak_learner.fit(features, labels, eps_memb, eps_val, full_labels)
			# print(weak_learner.stump, weak_learner.choice)

			# Compute the error of the weak learner and the weight multiplier
			labels['pred'] = pd.Series(weak_learner.predict(features[f_names]), index = labels.index)
			# print(labels['weight'].sum(), self.label_halls)
			if weak_learner.stump is not None:
				err = (labels[labels[l_name] != labels['pred']]['weight'].sum() - \
					self.feat_label_halls[weak_learner.stump][0][1 - weak_learner.choice[0]] - \
					self.feat_label_halls[weak_learner.stump][1][1 - weak_learner.choice[1]]) / \
					(labels['weight'].sum() - self.feat_label_halls[weak_learner.stump][0][0] - \
					self.feat_label_halls[weak_learner.stump][0][1] - \
					self.feat_label_halls[weak_learner.stump][1][0] - \
					self.feat_label_halls[weak_learner.stump][1][1])
			else:
				err = (labels[labels[l_name] != labels['pred']]['weight'].sum() - \
					self.label_halls[1 - weak_learner.choice]) / \
					(labels['weight'].sum() - self.label_halls[0] - self.label_halls[1])
			if err < 0.01:
				err = 0.01
			weight_multiplier = (1 - err) / err 
			labels['weight'] *= np.exp((labels[l_name] + labels['pred']) % 2 * log(weight_multiplier))

			# Normalize the weights, propagate weights to the features DataFrame as well
			normalizer = labels['weight'].sum() / len(labels)
			labels['weight'] = labels['weight'] / normalizer
			features['weight'] = labels['weight']

			# Update the expected weights of the hallucinations
			self.update_hall_counts(f_names, labels, weak_learner, weight_multiplier, normalizer)
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
