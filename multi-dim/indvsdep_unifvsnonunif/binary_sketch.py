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
def sample_dp_signs(eps, reduced_features, total_features, sketch_type):
	if sketch_type == 'Ind':
		dist = [exp(-1.0 * eps / reduced_features) / 2.0, 1.0 - exp(-1.0 * eps / reduced_features) / 2.0]
		return [choices([-1, 1], weights = dist)[0] for i in range(total_features)]

	keep_prob = (exp(eps) - 1) / (exp(eps) + (2 ** (reduced_features)) - 1)
	if np.random.uniform() < keep_prob:
		return [1 for i in range(total_features)]
	else:
		return [choice([-1, 1]) for i in range(total_features)]

class Binary_Sketch:
	def __init__(self, eps, index_universe, num_buckets, sketch_type, feat_type):
		if num_buckets is None:
			num_buckets = len(index_universe)

		self.eps = eps
		self.index_universe = index_universe
		self.num_buckets = num_buckets
		self.features = None
		self.probabilities = None
		self.sketch_type = sketch_type
		self.feat_type = feat_type

	def get_features(self, df, reduced_features):
		feature_selector = Feature_Selection(self.eps, self.index_universe, self.feat_type)
		feature_selector.populate(df, reduced_features)
		self.features = feature_selector.features
		self.probabilities = feature_selector.probabilities
		return self.features, self.probabilities

	def get_signs(self, col_names, reduced_features):
		flips = pd.DataFrame(index = self.index_universe, columns = col_names)

		for i in self.index_universe:
			flips.loc[i] = sample_dp_signs(self.eps, reduced_features, len(col_names), self.sketch_type)

		return flips
