import numpy as np
import pandas as pd
# pd.options.mode.chained_assignment = None  
from math import exp
from hashes import Hashes

# Returns a sample from the two-sided geometric distribution
def two_sided_geom(p):
		return np.random.geometric(1.0 - p) - np.random.geometric(1.0 - p) 

class Member_Sketch:
	def __init__(self, eps, index_universe, num_buckets = None):
		if num_buckets is None:
			num_buckets = len(index_universe)

		self.eps = eps
		self.hashes = Hashes(index_universe)
		self.num_buckets = num_buckets
		self.sketch = [0.0 for i in range(num_buckets)]

	def populate(self, index):
		for i in index:
			self.sketch[self.hashes.buckets[i]] += self.hashes.signs[i] * 1

		for b in range(self.num_buckets):
			self.sketch[b] += two_sided_geom(exp(-1.0 * self.eps))


