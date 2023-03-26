import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  

from math import exp, log
from sklearn import metrics 
from sklearn.naive_bayes import MultinomialNB

import warnings
warnings.filterwarnings("ignore", message = "X does not have valid feature names, but MultinomialNB was fitted with feature names")

class TestTimeCorrection:
	def __init__(self, classifier, num_votes = 201):
		self.num_votes = num_votes
		self.classifier = classifier
		self.flip_prob_val = None

	def fit(self, features, labels, eps_memb, eps_val, **kwargs):
		self.flip_prob_val = exp(-1.0 * eps_val) / 2.0
		self.classifier.fit(features, labels)

	def predict(self, f_test):
		votes = []
		for i in range(self.num_votes):
			rr = np.random.choice([1, -1], \
				size = f_test.shape, \
				p = [1 - self.flip_prob_val, self.flip_prob_val])

			f_test_noisy = np.multiply(f_test, rr)
			np.place(f_test_noisy, f_test_noisy == -1, [0])

			votes.append(self.classifier.predict(f_test_noisy))
		# print(votes)
		votes = np.array(votes)
		votes = np.median(votes, axis = 0).flatten().tolist()
		# print(votes)
		return votes
