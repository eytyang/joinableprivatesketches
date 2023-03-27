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
		self.known_cols = None

	# Combines the value sketch and the membership sketch to perform a join
	def join(self, df_known, df_private, num_features = 3):
		index_universe = df_private.index.union(df_known.index)
		df_dp = df_known.copy()
		self.known_cols = df_known.columns

		memb = Member_Sketch(self.eps_memb, index_universe, self.num_buckets)
		memb.populate(df_private, num_features)
		self.features = memb.features
		df_dp = decode_sketch(df_dp, 'membership', memb)
		
		num_cols = len(df_private.columns)
		for col in df_private.columns:
			new_col = {}
			val = Binary_Sketch(self.eps_val / num_features, index_universe, self.num_buckets)
			val.populate(df_private[col])
			df_dp = decode_sketch(df_dp, col, val)				
		self.df = df_dp

	# TODO: DO THIS MORE CLEANLY USING LAMBDAS / MAPS
	def populate_nans(self):
		for col in self.df.columns:
			if col in self.known_cols or col == 'membership':
				continue
			self.df[col].iloc[self.features[col]] = None

	def drop_entries(self):
		self.df = self.df[self.df['membership'] >= 1]

	def flip_labels(self, l_name):
		self.df = self.df[self.df['membership'] != 0]
		self.df['sign'] = np.sign(self.df['membership'])
		self.df[l_name] = self.df[l_name].multiply(self.df['sign'], axis = 'index')

