import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  

from math import exp

import warnings
warnings.filterwarnings("ignore", message = "A column-vector y was passed when a 1d array was expected.")
warnings.filterwarnings("ignore", message = "X has feature names")

# Estimate the true intersection size using the number of features and the join size
def est_size(full_data_size, dp_join_size, hall_prob):
	return (dp_join_size - hall_prob * full_data_size) / (1 - 2.0 * hall_prob)

class AdjustedLinearRegression:
	def __init__(self, conservative = True, fit_intercept = False):
		self.data_dependent = data_dependent
		self.fit_intercept = fit_intercept
		self.slope = None
		self.intercept = 0.0 # Hard-coding this for now.

	def fit(self, X, y, eps_memb, full_y = None):
		hall_prob = exp(-1.0 * eps_memb) / (1 + exp(-1.0 * eps_memb))
		
		full_data_size, dp_join_size = len(full_y), len(y)
		join_size = est_size(full_data_size,dp_join_size, hall_prob)
		num_hallucinations = hall_prob * (full_data_size - join_size)
		
		X_mat, y_vec = X.to_numpy(), y.to_numpy()
		dot = np.matmul(X_mat.T, y_vec)

		if self.conservative:
			inv = np.linalg.inv(np.matmul(X_mat.T, X_mat) \
			 	- num_hallucinations * 2 * ((X_sens / eps_val) ** 2) * np.identity(1))
		else:	
			inv = np.linalg.inv(np.matmul(X_mat.T, X_mat) \
				- dp_join_size * 2 * ((X_sens / eps_val) ** 2) * np.identity(1))
		
		self.slope = np.matmul(inv, dot)

	def predict(self, X):
		return np.matmul(X, self.slope)