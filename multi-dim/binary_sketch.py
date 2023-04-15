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
def sample_dp_signs(eps_list, reduced_features, total_features, sketch_type):
	if sketch_type == 'Ind' or sketch_type == 'WeightedInd':
		dist = {}
		for i in range(len(eps_list)):
			dist[i] = [exp(-1.0 * eps_list[i]) / 2.0, 1.0 - exp(-1.0 * eps_list[i]) / 2.0]
		return [choices([-1, 1], weights = dist[i])[0] for i in range(total_features)]

	if sketch_type == 'Dep':
		eps = eps_list
		keep_prob = (exp(eps) - 1) / (exp(eps) + (2 ** (reduced_features)) - 1)
		if np.random.uniform() < keep_prob:
			return [1 for i in range(total_features)]
		else:
			return [choice([-1, 1]) for i in range(total_features)]

class Binary_Sketch:
	def __init__(self, eps, index_universe, num_buckets, sketch_type, feat_type):
		if num_buckets is None:
			num_buckets = len(index_universe)

		self.feat_type = feat_type
		if self.feat_type != 'Unif':
			self.feat_eps = 0.25 * eps
		else:
			self.feat_eps = 0.0
		# TODO: Allow for user to pass in feat_eps
		
		self.eps = eps - self.feat_eps 
		self.index_universe = index_universe
		self.num_buckets = num_buckets
		self.probabilities = None
		self.sketch_type = sketch_type
		self.features = None


	def get_features(self, df, reduced_features):
		feature_selector = Feature_Selection(self.feat_eps, self.index_universe, self.feat_type)
		feature_selector.populate(df, reduced_features)
		self.features = feature_selector.features
		return feature_selector.features, feature_selector.p

	def get_signs(self, col_names, reduced_features, p = None):
		flips = pd.DataFrame(index = self.index_universe, columns = col_names)

		for i in self.index_universe:
			if self.sketch_type == 'Ind':
				eps_list = [self.eps / reduced_features for i in range(len(col_names))]
			if self.sketch_type == 'Dep':
				# TODO: KIND OF HACKY THAT THE TYPES ARE OFF
				eps_list = self.eps
			if self.sketch_type == 'WeightedInd':
				eps_list = [p[i] if i in self.features else 0.0 for i in range(len(col_names))]
				sum_eps_list = sum(eps_list)
				eps_list = [self.eps * eps_list[i] / sum_eps_list for i in range(len(col_names))]
			flips.loc[i] = sample_dp_signs(eps_list, reduced_features, len(col_names), self.sketch_type)

		return flips
