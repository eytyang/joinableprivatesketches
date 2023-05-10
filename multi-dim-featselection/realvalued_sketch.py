'''
Given a list of ids ('index_universe', which is a pandas.Index object), 
this class initializes a 'sketch' vector that:

    (1) is indexed by {1, 2, ..., num_buckets} if num_buckets is specified,
        otherwise indexed by {1, 2, ..., |index_universe|}

    (2) stores S(i) + Z(i) at each entry, where 
        S(i) = sum([sign(id) * value(id) : hash(id) = i])
        and Z(i) is independent noise that depends on 'eps' and 'sens'. 
        The functions sign() and hash() are stored in a Hashes object, 
        and value() comes from 'df_col', an input into the 'populate' function.
'''

import numpy as np
from hashes import Hashes

class RealValued_Sketch:
    def __init__(self, eps, sens, index_universe, num_buckets = None):
        '''
        The RealValued_Sketch object stores the following:

            - eps:      a positive real number, representing a privacy parameter, 
                            that affects the amount of noise added to the final vector
            - sens:     a positive real number, representing an upper bound on the 
                            magnitude of value().
            - hashes:   a Hashes object containing a hash function and a sign function
                            whose domains are both 'index_universe'
            - sketch:   the sketch vector defined above. 
        '''
        if num_buckets is None:
            num_buckets = len(index_universe)

        self.eps = eps
        self.sens = sens
        self.hashes = Hashes(index_universe)
        self.sketch = [0.0 for i in range(num_buckets)]

    def populate(self, df_col):
        '''
        Creates the 'sketch' vector indexed by hash bucket and saves it to self.sketch. 
        The value stored at each vector entry is described by (2), above.

        The index of 'df_col' is a subset of 'index_universe', input above. 
        '''
        for i in df_col.index:
            self.sketch[self.hashes.buckets[i]] += self.hashes.signs[i] * df_col.loc[i]

        for b in range(len(self.sketch)):
            self.sketch[b] += np.random.laplace(scale = 1.0 * self.sens / self.eps)
