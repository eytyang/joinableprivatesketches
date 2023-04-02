import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None  

from math import exp, log
from random import randint
from feature_selection import Feature_Selection
from membership_sketch import Member_Sketch
from binary_sketch import Binary_Sketch

def decode_sketch(df, col, sketch_obj):
	decoded_col = {}
	for i in df.index:
		bucket_index = sketch_obj.hashes.buckets[i]
		decoded_col[i] = sketch_obj.hashes.signs[i] * sketch_obj.sketch[bucket_index]
	df[col] = pd.Series(decoded_col)
	return df

class DP_Join:
	def __init__(self, eps_memb, eps_val, sens = 1.0, num_buckets = None):
		self.eps_memb = eps_memb
		self.eps_val = eps_val
		self.sens = sens
		self.num_buckets = num_buckets
		self.df = None
		self.features = None
		self.probabilities = None
		self.known_cols = None

	# Combines the value sketch and the membership sketch to perform a join
	def join(self, df_known, df_private, num_features):
		index_universe = df_private.index.union(df_known.index)
		df_dp = df_known.copy()
		self.known_cols = df_known.columns

		memb = Member_Sketch(self.eps_memb, index_universe)
		memb.populate(df_private)

		# TODO: THIS IS A HACK; FIX THIS
		self.num_buckets = memb.num_buckets
		df_dp = decode_sketch(df_dp, 'membership', memb)
		
		val = Binary_Sketch(self.eps_val, index_universe, self.num_buckets)
		self.features, self.probabilities = val.get_features(df_private, num_features)
		# TODO: Make this more streamlined, somehow
		for col in self.known_cols:
			self.features[col] = pd.Series([1.0 for i in index_universe], index = index_universe)
		self.features['membership'] = pd.Series([1.0 for i in index_universe], index = index_universe)
		
		signs = val.get_signs(df_private.columns, num_features)
		df_private = df_private.mul(signs)
		df_dp = df_dp.join(df_private)
		self.df = df_dp.applymap(lambda x: x if not np.isnan(x) else np.random.choice([-1, 1]))

	# TODO: DO THIS MORE CLEANLY USING LAMBDAS / MAPS
	def populate_nans(self):
		self.df = self.df.mul(self.features)

	def drop_entries(self):
		self.df = self.df[self.df['membership'] >= 1]

	def flip_labels(self, l_name):
		self.df = self.df[self.df['membership'] != 0]
		self.df['sign'] = np.sign(self.df['membership'])
		self.df[l_name] = self.df[l_name].multiply(self.df['sign'], axis = 'index')

