import numpy as np
import pandas as pd
# pd.options.mode.chained_assignment = None  
from math import exp
from hashes import Hashes

# Returns a sample from the two-sided geometric distribution
def two_sided_geom(p):
		return np.random.geometric(1.0 - p) - np.random.geometric(1.0 - p) 

class Binary_Sketch:
	def __init__(self, eps, index_universe, num_buckets = None):
		# Allows us to enforce no hash collisions by allocating enough buckets.
		if num_buckets is None:
			num_buckets = len(index_universe)

		self.eps = eps
		self.hashes = Hashes(index_universe)
		self.num_buckets = num_buckets
		self.sketch = [0.0 for i in range(num_buckets)]

	# Returns a vector indexed by hash bucket. 
	# Each bucket entry contains the CountSketch value of that bucket
	# along with the noise added to maintain privacy. 
	def populate(self, df_col):
		for i in df_col.index:
			self.sketch[self.hashes.buckets[i]] += self.hashes.signs[i] * df_col.loc[i]

		for b in range(self.num_buckets):
			self.sketch[b] += two_sided_geom(exp(-1.0 * self.eps)) 
			if self.sketch[b] == 0:
				self.sketch[b] = np.random.choice([-1, 1]) 
			else:
				self.sketch[b] = np.sign(self.sketch[b])

