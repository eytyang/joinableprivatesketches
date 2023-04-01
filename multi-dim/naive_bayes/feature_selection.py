from random import choice
import numpy as np
import pandas as pd
import scipy as sc
from math import exp
from hashes import Hashes

def rand_unit_vec(length):
	gaussian = np.random.normal(size = length)
	# TODO: FIX THIS HACK
	return ((11274) ** (0.5)) * gaussian / np.linalg.norm(gaussian) 

# Takes in a query vector and a dictionary mapping between 
# column name and the column vector
def get_closest_vec(query, col_vecs):
	min_dist = np.inf
	closest_col = []
	for col in col_vecs:
		if np.linalg.norm(query - col_vecs[col]) == min_dist:
			closest_col.append(col) 
		elif np.linalg.norm(query - col_vecs[col]) < min_dist:
			min_dist = np.linalg.norm(query - col_vecs[col])
			closest_col = [col]
	return choice(closest_col)

def sample_closest_vecs(num_features, vec_len, col_vecs, col_dict, i):
	sample = []
	while len(sample) < num_features:
		new_col = get_closest_vec(rand_unit_vec(vec_len), col_vecs)
		if new_col not in sample:
			col_dict[new_col].append(i)
			sample.append(new_col)
	return col_dict

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
		col_vecs = {}
		cov_matrix = np.matmul(df.to_numpy().T, df.to_numpy())
		sqrt_matrix = sc.linalg.sqrtm(cov_matrix)
		for i in range(len(col_names)):
			self.features[col_names[i]] = []
			col_vecs[col_names[i]] = sqrt_matrix[:,i]
		print("Lin Alg Preprocessing Complete")	

		counter = 0
		for i in self.index_universe:
			self.features = sample_closest_vecs(num_features, len(col_names), col_vecs, self.features, i)
			counter += 1
			if counter % 1000 == 0:
				print("Sampled %i Rows" % counter)

		for col in col_names:
			self.probabilities[col] = len(self.features[col]) / (self.num_buckets)

