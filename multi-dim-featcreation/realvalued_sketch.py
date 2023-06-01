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
import pandas as pd

class RealValued_Sketch:
    def __init__(self, eps, sens_list, index_universe, num_buckets):
        '''
        The RealValued_Sketch object stores the following:

            - eps:      a positive real number, representing a privacy parameter, 
                            that affects the amount of noise added to the final vector
            - sens:     a positive real number, representing an upper bound on the 
                            magnitude of value().
        '''
        if num_buckets is None:
            num_buckets = len(index_universe)

        self.eps = eps
        self.index_universe = index_universe
        self.sens_list = np.array(sens_list)

    def get_noise(self, col_names, p = None):
        sens_matrix = np.tile(self.sens_list, (len(self.index_universe), 1))
        laplace = np.random.laplace(scale = len(col_names) / self.eps, size = (len(self.index_universe), len(col_names)))
        return np.multiply(sens_matrix, laplace)
