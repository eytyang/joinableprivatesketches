from random import choice
import numpy as np
import pandas as pd
import scipy as sc
from math import exp
from hashes import Hashes

def rand_unit_vec(length, scaling):
	gaussian = np.random.normal(size = length)
	return scaling * gaussian / np.linalg.norm(gaussian)

# Takes in a query vector and a dictionary mapping between 
# column name and the column vector
def get_closest_vec(mat, vec_norm):
	min_dist = np.inf
	closest_col = []
	for i in range(mat.shape[0]):
		query = rand_unit_vec(mat.shape[0], vec_norm)
		dist = np.linalg.norm(query - mat[:, i])
		if dist < min_dist:
			min_dist = dist
			closest_col = i
	return closest_col

def sample_closest_vecs(num_features, mat, vec_norm):
	if num_features == mat.shape[0]:
		return [1.0 for i in range(mat.shape[0])]
	sample = []
	while len(sample) < num_features:
		new_col = get_closest_vec(mat, vec_norm)
		if new_col not in sample:			
			sample.append(new_col)
	return [None if i not in sample else 1.0 for i in range(mat.shape[0])]

class Feature_Selection:
	def __init__(self, eps, index_universe):
		self.eps = eps
		self.index_universe = index_universe
		self.num_buckets = len(index_universe)
		self.features = {}
		self.probabilities = {}

	# Here, we select num_features features per index
	# TODO: REFACTOR THIS AND INCORPORATE DP
	def populate(self, df, num_features):
		col_names = df.columns
		num_rows = len(df.index)

		# Get columns of df as a list of vectors
		data = df.to_numpy()
		gram = np.matmul(data.T, data)
		mat = sc.linalg.sqrtm(gram)
		vec_norm = len(data) ** 0.5
		print("Lin Alg Preprocessing Complete")	

		counter = 0
		# scale_factor = (len(df)) ** (0.5)
		self.features = pd.DataFrame(index = self.index_universe, columns = col_names)
		for i in self.index_universe:
			self.features.loc[i] = sample_closest_vecs(num_features, mat, vec_norm)
			counter += 1
			if counter % 1000 == 0:
				print("Sampled %i Rows" % counter)

		for col in col_names:
			self.probabilities[col] = len(self.features[col].dropna()) / len(self.index_universe)

