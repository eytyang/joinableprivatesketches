import numpy as np
import pandas as pd
from math import exp
from random import choice, choices
from hashes import Hashes
from feature_selection import Feature_Selection

# Returns a sample from the two-sided geometric distribution
def two_sided_geom(p):
		return np.random.geometric(1.0 - p) - np.random.geometric(1.0 - p) 

# Implement DP sign flips:
def sample_dp_signs(eps, num_features, total_features):
	dist = [exp(-1.0 * eps / num_features) / 2.0, 1.0]
	return [choices([-1, 1], weights = dist)[0] for i in range(total_features)]

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
		flips = pd.DataFrame(index = self.index_universe, columns = col_names)

		for i in self.index_universe:
			flips.loc[i] = sample_dp_signs(self.eps, num_features, len(col_names))

		return flips
