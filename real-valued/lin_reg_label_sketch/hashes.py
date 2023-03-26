''' 
The overall goal of this class is to simulate the behavior of a hash function without
actually using a hash function.

Given a list of ids (stored in the variable 'index', which is a pandas.Index object), 
this class initializes two functions: 

    (1) A function from 'index' to {1, 2, ..., num_buckets}. 
    If 'num_buckets' is specified, this is a random hash function. 
    If num_buckets is None, we save a 1-1 map from {ids} to {1, 2, ..., |index|} 
    and enforce no hash collisions. 

    (2) A function from 'index' to {-1, 1}. An output of -1 or 1 is chosen 
    uniformly at random for each id. 
'''

import numpy as np

def bucket_hash(index, num_buckets):
    '''
    Returns a dictionary mapping elements of 'index' to integers.
    If 'num_buckets' is specified, each element of 'index' maps uniformly at random to 
        an integer in {1, 2, ..., num_buckets}.
    If 'num_buckets' is None, we map 'index' to {1, 2, ..., |index|} in a 
        1-1 fashion, which simulates what happens with no hash collisions. 
    '''
    count = 0
    buckets = {}
    for i in set(index):
        # This case enforces no hash collisions
        if num_buckets is None:
            buckets[i] = count
            count += 1
        # This case allows for hash collisions
        else:
            buckets[i] = np.random.randint(0, num_buckets)
    return buckets

def sign_hash(index):
    '''
    Returns a dictionary mapping elements of 'index' randomly to -1 or +1
    '''
    signs = {}
    for i in index:
        signs[i] = np.random.choice([-1, 1])
    return signs

class Hashes:
    def __init__(self, index, num_buckets = None):
        '''
        The Hashes object stores the following:

            index:          a list (pandas.Index object) of ids that is the domain 
                                of functions (1), (2) defined above
            num_buckets:    the size of the co-domain for function (1)
            buckets:        the initialized function (1)
            sign_hash:      the initialized function (2)
        '''
        self.index = index
        self.num_buckets = num_buckets
        self.buckets = bucket_hash(index, num_buckets)
        self.signs = sign_hash(index)
