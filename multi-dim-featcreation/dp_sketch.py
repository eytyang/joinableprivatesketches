import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None  

from math import exp, log
from random import randint
from feature_selection import Feature_Selection
from membership_sketch import Member_Sketch
from binary_sketch import Binary_Sketch
from realvalued_sketch import RealValued_Sketch

# TODO: This function behaves a little awkwardly
def decode_sketch(df, col, sketch_obj):
	decoded_col = {}
	for i in df.index:
		bucket_index = sketch_obj.hashes.buckets[i]
		decoded_col[i] = sketch_obj.hashes.signs[i] * sketch_obj.sketch[bucket_index]
	df[col] = pd.Series(decoded_col)
	return df

class DP_Join:
	def __init__(self, eps_memb, eps_val, sens_list = None, num_buckets = None):
		self.eps_memb = eps_memb
		self.eps_val = eps_val
		self.sens_list = sens_list
		self.num_buckets = num_buckets
		self.features = None
		self.labels = None
		self.known_cols = None

	# Combines the value sketch and the membership sketch to perform a join
	def join(self, df_known, df_private, data_type = 'Real', dim = 0, bandwidth = 1.0):
		index_universe = df_private.index.union(df_known.index)
		df_dp = df_known.copy()
		self.known_cols = df_known.columns

		memb = Member_Sketch(self.eps_memb, index_universe)
		memb.populate(df_private)

		# TODO: THIS IS A HACK; FIX THIS
		self.num_buckets = len(index_universe)
		df_dp = decode_sketch(df_dp, 'membership', memb)
		df_dp = df_dp.join(df_private)
		if data_type == 'Real' or data_type == 'Real Clip':
			df_dp = df_dp.fillna(0)
		elif data_type == 'Binary':
			df_dp = df_dp.fillna(1)
		df_dp = df_dp[df_dp['membership'] != 0]
		df_dp['sign'] = np.sign(df_dp['membership'])
		df_dp[self.known_cols[0]] = df_dp[self.known_cols[0]].multiply(df_dp['sign'], axis = 'index')
		
		if data_type == 'Real' or data_type == 'Real Clip':
			val = RealValued_Sketch(self.eps_val, self.sens_list, df_dp.index, self.num_buckets)
			noise = val.get_noise(df_private.columns)

			df_dp = df_dp.applymap(lambda x: x if not np.isnan(x) else 0)
			self.features = df_dp[df_private.columns].to_numpy() + noise

			if data_type == 'Real Clip':
				self.features = np.clip(self.features, -1 * 2 ** 0.5, 2 ** 0.5)
		elif data_type == 'Binary':
			val = Binary_Sketch(self.eps_val, df_dp.index, self.num_buckets)
			signs = val.get_signs(df_private.columns)

			df_dp = df_dp.applymap(lambda x: x if not np.isnan(x) else np.random.choice([-1, 1]))
			self.features = np.multiply(df_dp[df_private.columns].to_numpy(), signs)
		self.labels = df_dp[df_known.columns]
	# TODO: ADD FLIP LABELS BACK

