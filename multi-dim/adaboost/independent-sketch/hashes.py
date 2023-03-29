import numpy as np
import pandas as pd
# pd.options.mode.chained_assignment = None  
from random import randint

def bucket_hash(index, num_buckets):
	count = 0
	buckets = {}
	for i in set(index):
		if num_buckets is None:
			buckets[i] = count
			count += 1
		else:
			buckets[i] = np.random.randint(0, num_buckets)
	return buckets

# Returns a dictionary that represents a random sign function
def sign_hash(index):
	signs = {}
	for i in index:
		signs[i] = np.random.choice([-1, 1])
	return signs

class Hashes:
	# Returns a dictionary that maps index element to a bucket
	# If num_buckets = None, we enforce no hash collisions 
	def __init__(self, index, num_buckets = None):
		self.index = index
		self.num_buckets = num_buckets
		self.buckets = bucket_hash(index, num_buckets)
		self.signs = sign_hash(index)

