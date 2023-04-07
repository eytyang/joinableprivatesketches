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

def sample_closest_vecs(reduced_features, mat, vec_norm, feat_type):
	if reduced_features == mat.shape[0]:
		return [1.0 for i in range(mat.shape[0])]
	
	if feat_type == 'Unif':
		subsample = sample(range(mat.shape[0]), reduced_features)
	else:	
		subsample = []
		while len(subsample) < reduced_features:
			new_col = get_closest_vec(mat, vec_norm)
			if new_col not in subsample:			
				subsample.append(new_col)
	return [choice([-1, 1]) if i not in subsample else 1.0 for i in range(mat.shape[0])]

class Feature_Selection:
	def __init__(self, eps, index_universe, feat_type):
		self.eps = eps
		self.index_universe = index_universe
		self.num_buckets = len(index_universe)
		self.features = {}
		self.probabilities = {}
		self.feat_type = feat_type

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
		for i in self.index_universe:
			self.features.loc[i] = sample_closest_vecs(reduced_features, mat, vec_norm, self.feat_type)
			counter += 1
			# if counter % 5000 == 0:
			# 	print("Sampled %i Rows" % counter)

		for col in col_names:
			self.probabilities[col] = len(self.features[col].dropna()) / len(self.index_universe)

