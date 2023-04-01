from math import log
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  

from math import exp, log
from sklearn import metrics 
from sklearn.naive_bayes import MultinomialNB

from dp_sketch import DP_Join

class NB_Weighted:
	def __init__(self, num_features, minimum = 1.0):
		self.minimum = minimum
		self.num_features = num_features
		self.label_counts = {}
		self.feat_label_counts = {}
		self.non_nan_counts = {}

	# REFACTOR
	def populate_counts(self, features, labels, probabilities = None):
		f_names = features.columns
		l_name = labels.columns[0]
		dp_label_count = labels[l_name].value_counts()

		if probabilities is None:
			probabilities = {}
			for col in f_names:
				probabilities[col] = 1.0

		for l_val in [0, 1]:
			if l_val in dp_label_count:
				self.label_counts[l_val] = dp_label_count[l_val] + self.minimum
			else:
				self.label_counts[l_val] = self.minimum

			labels_restricted = labels[labels[l_name] == l_val]
			features_restricted = features.loc[labels_restricted.index]
			for i in range(len(f_names)):
				if f_names[i] not in features_restricted:
					dp_label_feat_count = {0: self.minimum, 1: self.minimum}
					continue

				dp_label_feat_count = features_restricted[f_names[i]].value_counts()
				if 0 not in dp_label_feat_count:
					if 1 not in dp_label_feat_count:
						dp_label_feat_count = {0: self.minimum, 1: self.minimum}
					else:
						dp_label_feat_count = {0: self.minimum, 1: dp_label_feat_count[1]}
				if 1 not in dp_label_feat_count and 0 in dp_label_feat_count:
					dp_label_feat_count = {0: dp_label_feat_count[0], 1: self.minimum}

				for f_val in [0, 1]:
					if probabilities[f_names[i]] < 0.00:
						self.feat_label_counts[i][f_val][l_val] = self.minimum
					else:
						self.feat_label_counts[i][f_val][l_val] = max(dp_label_feat_count[f_val] + self.minimum, self.minimum)

		for i in range(len(f_names)):
			self.non_nan_counts[i] = self.feat_label_counts[i].sum()

	def fit(self, features, labels, probabilities = None):
		for i in range(len(features.columns)):
			self.feat_label_counts[i]= np.zeros((2, 2))

		self.populate_counts(features, labels, probabilities)
		# print(self.label_counts, self.feat_label_counts)

	def compute_log_likelihood(self, feature_vec, l_val):
		label_count = self.label_counts[l_val]
		if label_count <= 0.0:
		 	return -1 * np.inf
		log_likelihood = log(label_count / (self.label_counts[0] + self.label_counts[1])) * ((self.label_counts[0] + self.label_counts[1]) * self.num_features / 22.0)
		for i in range(len(feature_vec)):
			if self.feat_label_counts[i][feature_vec[i]][l_val] == 0:
				return -1 * np.inf
			log_likelihood += self.non_nan_counts[i] * log(self.feat_label_counts[i][feature_vec[i]][l_val] / \
				self.label_counts[l_val])
		return log_likelihood

	def classify(self, feature_vec):
		zero_log_likelihood = self.compute_log_likelihood(feature_vec, 0)
		one_log_likelihood = self.compute_log_likelihood(feature_vec, 1)
		# print(zero_log_likelihood, one_log_likelihood)
		if zero_log_likelihood > one_log_likelihood:
			return 0
		elif zero_log_likelihood == one_log_likelihood:
			if self.label_counts[0] > self.label_counts[1]:
				return 0
		return 1

	def predict(self, f_test):
		pred_bayes = []
		for f_vec in f_test:
			pred_bayes.append(self.classify(f_vec))
		return pred_bayes
