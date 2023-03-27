from random import choice
import numpy as np
import pandas as pd
from math import exp
from hashes import Hashes

def rand_binary_vec(length):
	gaussian = np.random.normal(length)
	return np.sign(gaussian)

# Takes in a query vector and a dictionary mapping between 
# column name and the column vector
def get_closest_vec(query, col_vecs, already_seen = []):
	min_dist = np.inf
	closest_col = []
	for col in col_vecs:
		if col in already_seen:
			continue
		if np.linalg.norm(query - col_vecs[col]) <= min_dist:
			min_dist = np.linalg.norm(query - col_vecs[col])
			closest_col.append(col) 
	return choice(closest_col)

def sample_closest_vecs(num_features, num_rows, col_vecs, col_dict, i):
	sample = []
	while len(sample) < num_features:
		new_col = get_closest_vec(rand_binary_vec(num_rows), col_vecs, sample)
		if new_col not in sample:
			col_dict[new_col].append(i)
			sample.append(new_col)
	
	return col_dict

class Feature_Selection:
	def __init__(self, eps, num_buckets = None):
		self.eps = eps
		self.num_buckets = num_buckets
		self.features = {}

	# Here, we select num_features features per index
	# TODO: REFACTOR THIS AND INCORPORATE DP
	def populate(self, df, num_features):
		num_rows = len(df.index)

		# Get columns of df as a list of vectors
		col_vecs = {}
		for col in df.columns:
			col_vecs[col] = df[col].to_numpy()
			self.features[col] = []

		for i in range(self.num_buckets):
			self.features = sample_closest_vecs(num_features, num_rows, col_vecs, self.features, i)


