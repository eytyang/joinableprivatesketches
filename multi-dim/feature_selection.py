from random import choice, sample
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

def get_col_dist(mat, vec_norm, num_samples = 10000):
	rand_vecs = np.random.normal(size = (num_samples, mat.shape[0]))
	norms = np.linalg.norm(rand_vecs, axis = 1)
	rand_unit_vecs = vec_norm * rand_vecs / norms[:, np.newaxis]

	samples = list(np.argmax(np.matmul(rand_unit_vecs, mat), axis = 1))
	# Guarantee that each feature gets counted at least once.
	samples.extend(list(range(mat.shape[1]))) 
	return np.unique(samples, return_counts = True)[1]

def sample_closest_vecs(reduced_features, mat, vec_norm, p = None):
	if reduced_features == mat.shape[0]:
		return [1.0 for i in range(mat.shape[0])]
	
	if p is None:
		subsample = sample(range(mat.shape[0]), reduced_features)
	else:	
		subsample = np.random.choice(mat.shape[0], size = reduced_features, replace = False, p = p)
	return [0 if i not in subsample else 1.0 for i in range(mat.shape[0])]

class Feature_Selection:
	def __init__(self, eps, index_universe, feat_type):
		self.eps = eps
		self.index_universe = index_universe
		self.num_buckets = len(index_universe)
		self.features = None
		self.probabilities = {}
		self.feat_type = feat_type
		self.col_subset = None

	# Here, we select reduced_features features per index
	# TODO: INCORPORATE DP
	def populate(self, df, reduced_features):
		col_names = df.columns
		num_rows = len(df.index)

		# Get columns of df as a list of vectors
		data = df.to_numpy()
		gram = np.matmul(data.T, data)
		mat = sc.linalg.sqrtm(gram)
		vec_norm = len(data) ** 0.5

		counter = 0
		self.features = pd.DataFrame(index = self.index_universe, columns = col_names)
		if self.feat_type == "Unif":
			for i in self.index_universe:
				self.features.loc[i] = sample_closest_vecs(reduced_features, mat, vec_norm)
				counter += 1
		else:
			p = get_col_dist(mat, vec_norm, 25 * len(data))
			p = p / np.sum(p)
			if self.feat_type == "NonUnif":
				for i in self.index_universe:
					self.features.loc[i] = sample_closest_vecs(reduced_features, mat, vec_norm, p)
					counter += 1
			if self.feat_type == "Same":
				p = get_col_dist(mat, vec_norm, 25 * len(data))
				p = p / np.sum(p)
				self.col_subset = np.argpartition(p, reduced_features)[-1 * reduced_features:]

		for col in col_names:
			self.probabilities[col] = len(self.features[col].dropna()) / len(self.index_universe)

