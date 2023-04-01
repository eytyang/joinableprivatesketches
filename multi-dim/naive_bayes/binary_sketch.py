import numpy as np
import pandas as pd
from math import exp
from random import choice
from hashes import Hashes
from feature_selection import Feature_Selection

# Returns a sample from the two-sided geometric distribution
def two_sided_geom(p):
		return np.random.geometric(1.0 - p) - np.random.geometric(1.0 - p) 

# Implement DP sign flips:
def sample_dp_signs(eps, num_features):
	keep_prob = (exp(eps) - 1) / (exp(eps) + 2 ** (num_features) - 1)
	if np.random.uniform() < keep_prob:
		return [1 for i in range(num_features)]
	else:
		return [choice([-1, 1]) for i in range(num_features)]

class Binary_Sketch:
	def __init__(self, eps, index_universe, num_buckets = None):
		if num_buckets is None:
			num_buckets = len(index_universe)

		self.eps = eps
		self.index_universe = index_universe
		self.num_buckets = num_buckets
		self.features = None
		self.probabilities = None

	def get_features(self, df, num_features):
		feature_selector = Feature_Selection(self.eps, self.index_universe)
		feature_selector.populate(df, num_features)
		self.features = feature_selector.features
		self.probabilities = feature_selector.probabilities
		return self.features, self.probabilities

	def get_signs(self, col_names, num_features):
		flips = {}
		for col in col_names:
			flips[col] = []

		for i in self.index_universe:
			signs = sample_dp_signs(self.eps, num_features)
			counter = 0
			for col in col_names:
				if i in self.features[col]:
					flips[col].append(signs[counter])
					counter += 1
				else:
					flips[col].append(1)

		return pd.DataFrame(flips, index = self.index_universe)
