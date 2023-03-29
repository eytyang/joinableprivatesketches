import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  

from math import exp, log
from sklearn import metrics 
from sklearn.naive_bayes import MultinomialNB

from dp_sketch import DP_Join

class AdjustedNB:
	def __init__(self, minimum = 1):
		self.minimum = minimum
		self.label_counts = {}
		self.feat_label_counts = {}
		self.hall_prob_memb = None 
		self.flip_prob_val = None

	def populate_counts(self, features, labels, full_labels):
		f_names = features.columns
		l_name = labels.columns[0]
		dp_label_count = labels[l_name].value_counts()
		original_label_count = full_labels[l_name].value_counts()

		for l_val in [0, 1]:
			adjusted_label_count = max(self.minimum, (dp_label_count[l_val] - \
				self.hall_prob_memb * original_label_count[l_val]) / (1 - 2 * self.hall_prob_memb))
			self.label_counts[l_val] = adjusted_label_count

			labels_restricted = labels[labels[l_name] == l_val]
			features_restricted = features.loc[labels_restricted.index]
			for i in range(len(f_names)):
				dp_label_feat_count = features_restricted[f_names[i]].value_counts()
				if 0 not in dp_label_feat_count:
					dp_label_feat_count[0] = 0
				if 1 not in dp_label_feat_count:
					dp_label_feat_count[1] = 0

				for f_val in [0, 1]:
					self.feat_label_counts[i][f_val][l_val] = \
						max((dp_label_feat_count[f_val] - self.flip_prob_val * (1.0 - self.hall_prob_memb) * adjusted_label_count \
						- (self.hall_prob_memb / 2.0) * (original_label_count[l_val] - adjusted_label_count)) \
						/ ((1.0 - self.hall_prob_memb) * (1.0 - 2.0 * self.flip_prob_val)), self.minimum)

	def fit(self, features, labels, eps_memb, eps_val, full_labels):
		self.hall_prob_memb = 1.0 / (1.0 + exp(eps_memb))
		self.flip_prob_val = exp(-1.0 * eps_val / len(features.columns)) / 2.0

		for i in range(len(features.columns)):
			self.feat_label_counts[i]= np.zeros((2, 2))

		self.populate_counts(features, labels, full_labels)

	def compute_likelihood(self, feature_vec, l_val):
		label_count = self.label_counts[l_val]
		prod = label_count
		if label_count == self.minimum:
		 	return 0.0
		for i in range(len(feature_vec)):
			prod = prod * self.feat_label_counts[i][feature_vec[i]][l_val] / label_count
		return prod

	def classify(self, feature_vec):
		zero_likelihood = self.compute_likelihood(feature_vec, 0)
		one_likelihood = self.compute_likelihood(feature_vec, 1)
		if zero_likelihood > one_likelihood:
			return 0
		elif zero_likelihood == one_likelihood:
			if self.label_counts[0] > self.label_counts[1]:
				return 0
		return 1

	def predict(self, f_test):
		pred_bayes = []
		for f_vec in f_test:
			pred_bayes.append(self.classify(f_vec))
		return pred_bayes