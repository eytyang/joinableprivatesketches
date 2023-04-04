from math import isnan
from random import choice
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  

from math import exp, log, sqrt
from sklearn import metrics 
from sklearn.base import BaseEstimator
import warnings
warnings.filterwarnings("ignore", message = "X does not have valid feature names")

class Stump(BaseEstimator):
	def __init__(self, label_halls = None, feat_label_halls = None, count_minimum = 0):
		self.minimum = count_minimum
		self.label_counts = {}
		self.feat_label_counts = {}
		self.stump = None
		self.choice = {}

	def populate_counts(self, features, labels):
		f_names = features.columns
		l_name = labels.columns[0]
		dp_label_count = labels.groupby(l_name)['weight'].sum()
		
		for l_val in [0, 1]:
			self.label_counts[l_val] = max(dp_label_count[l_val], self.minimum)

			labels_restricted = labels[labels[l_name] == l_val]
			features_restricted = features.loc[labels_restricted.index]
			for i in range(len(f_names) - 1):
				dp_label_feat_count = features_restricted.groupby(f_names[i])['weight'].sum()
				if 0 not in dp_label_feat_count:
					if 1 not in dp_label_feat_count:
						dp_label_feat_count = {0: 1, 1: 1}
					else:
						dp_label_feat_count = {0: 1, 1: dp_label_feat_count[1]}
				if 1 not in dp_label_feat_count and 0 in dp_label_feat_count:
					dp_label_feat_count = {0: dp_label_feat_count[0], 1: 1}

				for f_val in [0, 1]:
					self.feat_label_counts[i][f_val][l_val] = max(dp_label_feat_count[f_val], self.minimum)

	def pick_stump(self, features):
		gini = []
		f_names = features.columns
		for i in range(len(f_names) - 1):
			gini.append(0.0)
			total_count = self.feat_label_counts[i][0][0] + self.feat_label_counts[i][0][1] \
							+ self.feat_label_counts[i][1][0] + self.feat_label_counts[i][1][1]
			if total_count == 0:
				gini[i] = np.inf
			else:
				for f_val in [0, 1]:
					feat_count = self.feat_label_counts[i][f_val][0] + self.feat_label_counts[i][f_val][1]
					if feat_count == 0:
						gini[i] = np.inf
					else:
						gini_contribution = (feat_count / total_count) * (1 - (self.feat_label_counts[i][f_val][0] / feat_count) ** 2 \
							- (self.feat_label_counts[i][f_val][1] / feat_count) ** 2)
						gini[i] += gini_contribution
		if min(gini) == np.inf:
			self.stump = np.random.choice(list(range(len(f_names) - 1)))
		else:
			self.stump = np.argmin(np.array(gini))

	def fill_choice(self):
		for stump_val in [0, 1]:
			if self.feat_label_counts[self.stump][stump_val][0] > self.feat_label_counts[self.stump][stump_val][1]:
				self.choice[stump_val] = 0
			elif self.feat_label_counts[self.stump][stump_val][0] < self.feat_label_counts[self.stump][stump_val][1]:
				self.choice[stump_val] = 1
			else:
				self.choice[stump_val] = np.random.choice([0, 1])
		if self.choice[0] == self.choice[1]:
			self.stump = None
			self.choice = self.choice[0]

	def fit(self, features, labels):
		# Off by 1 due to 'weight' column
		for i in range(len(features.columns) - 1):
			self.feat_label_counts[i] = np.zeros((2, 2))

		self.populate_counts(features, labels)
		self.pick_stump(features)
		self.fill_choice()
		return self

	def get_err(self):
		if self.stump is None:
			return (self.label_counts[1 - self.choice]) / \
				(self.label_counts[0] + self.label_counts[1])
		else:
			i = self.stump
			return (self.feat_label_counts[i][0][1 - self.choice[0]] + self.feat_label_counts[i][1][1 - self.choice[1]]) / \
				(self.feat_label_counts[i][0][0] + self.feat_label_counts[i][0][1] + \
				self.feat_label_counts[i][1][0] + self.feat_label_counts[i][1][1])

	def predict_array(self, f_vec):
		if self.stump is not None:
			return self.choice[f_vec[self.stump]]
		else:
			return self.choice

	def predict(self, f_test):
		pred = []
		for i in f_test.index:
			if self.stump is not None:
				if isnan(f_test.loc[i][f_test.columns[self.stump]]):
					pred.append(choice([0, 1]))
				else:
					pred.append(self.choice[f_test.loc[i][f_test.columns[self.stump]]])
			else:
				pred.append(self.choice)
		return pred
