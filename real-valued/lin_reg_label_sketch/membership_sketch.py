'''
Given a list of ids ('index_universe', which is a pandas.Index object), 
this class initializes a 'sketch' vector that:

    (1) is indexed by {1, 2, ..., num_buckets} if num_buckets is specified,
        otherwise indexed by {1, 2, ..., |index_universe|}

    (2) stores S(i) + Z(i) at each entry, where S(i) = sum([sign(id) : hash(id) = i])
        and Z(i) is independent noise that depends on 'eps'. 
        The functions sign() and hash() are stored in a Hashes object. 
'''

import numpy as np
from math import exp
from hashes import Hashes

def two_sided_geom(p):
    '''
    Returns a sample from the two-sided geometric distribution with parameter 'p'
    '''
    return np.random.geometric(1.0 - p) - np.random.geometric(1.0 - p) 

class Member_Sketch:
    def __init__(self, eps, index_universe, num_buckets = None):
        '''
        The Member_Sketch object stores the following:

            - eps:      a positive real number, representing a privacy parameter, 
                            that affects the amount of noise added to the final vector
            - hashes:   a Hashes object containing a hash function and a sign function
                            whose domains are both 'index_universe'
            - sketch:   the sketch vector defined above. 
        '''
        self.eps = eps
        self.hashes = Hashes(index_universe, num_buckets = num_buckets)
        
        if num_buckets is None:
            num_buckets = len(index_universe)
        self.sketch = [0.0 for i in range(num_buckets)]

    def populate(self, index):
        '''
        Creates the 'sketch' vector indexed by hash bucket and saves it to self.sketch. 
        The value stored at each vector entry is described by (2), above. 
        
        The index of 'df_col' is a subset of 'index_universe', input above. 
        '''
        for i in index:
            self.sketch[self.hashes.buckets[i]] += self.hashes.signs[i]

        for b in range(len(self.sketch)):
            self.sketch[b] += two_sided_geom(exp(-1.0 * self.eps)) 
