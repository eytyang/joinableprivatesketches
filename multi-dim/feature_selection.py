from random import choice, sample
import numpy as np
import pandas as pd
import scipy as sc
from math import exp
from hashes import Hashes

def get_col_distr(mat, vec_norm, feat_eps = 0.01, num_samples = 10000):
	rand_vecs = np.random.normal(size = (num_samples, mat.shape[0]))
	norms = np.linalg.norm(rand_vecs, axis = 1)
	rand_vecs = vec_norm * rand_vecs / norms[:, np.newaxis]

	dot_prods = np.matmul(rand_vecs, mat)
	dists = np.sqrt(2 * (vec_norm ** 2) * np.ones(shape = (num_samples, mat.shape[1])) - 2 * dot_prods)
	priv_dists = dists + np.random.laplace(scale = ((2.0 ** 0.5) * mat.shape[1] / feat_eps), size = (num_samples, mat.shape[1]))
	samples = list(np.argmin(priv_dists, axis = 1))

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
	return [choice([-1, 1]) if i not in subsample else 1.0 for i in range(mat.shape[0])]

class Feature_Selection:
	def __init__(self, feat_eps, index_universe, feat_type):
		self.feat_eps = feat_eps
		self.index_universe = index_universe
		self.num_buckets = len(index_universe)
		self.features = None
		self.feat_type = feat_type
		self.col_subset = None
		self.p = None

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
		if self.feat_type == "Unif":
			self.features = pd.DataFrame(index = self.index_universe, columns = col_names)
			for i in self.index_universe:
				self.features.loc[i] = sample_closest_vecs(reduced_features, mat, vec_norm)
				counter += 1
		else:
			self.p = get_col_distr(mat, vec_norm, self.feat_eps, 100 * len(data))
			self.p = self.p / np.sum(self.p)
			if self.feat_type == "NonUnif":
				self.features = pd.DataFrame(index = self.index_universe, columns = col_names)
				for i in self.index_universe:
					self.features.loc[i] = sample_closest_vecs(reduced_features, mat, vec_norm, self.p)
					counter += 1
			if self.feat_type == "Same":
				self.features = np.argpartition(self.p, -1 * reduced_features)[-1 * reduced_features:]
				print(self.features)