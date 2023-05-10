import numpy as np
import pandas as pd
from math import exp
from random import choice, choices
from hashes import Hashes
from feature_selection import Feature_Selection

# Returns a sample from the two-sided geometric distribution
def two_sided_geom(p):
		return np.random.geometric(1.0 - p) - np.random.geometric(1.0 - p) 

class Binary_Sketch:
	def __init__(self, eps, index_universe, num_buckets, sketch_type = 'Ind'):
		if num_buckets is None:
			num_buckets = len(index_universe)

		self.eps = eps
		self.index_universe = index_universe
		self.num_buckets = num_buckets
		self.probabilities = None
		self.sketch_type = sketch_type
		self.features = None

	def get_signs(self, col_names, p = None):
		bernoulli = np.random.binomial(1, 1.0 - exp(-1.0 * self.eps / len(col_names)) / 2.0, size = (len(self.index_universe), len(col_names)))
		return bernoulli * 2 - 1
