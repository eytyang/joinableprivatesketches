''' 
Implement an adjusted version of Ordinary Least Squares to account for hallucinations.
We support two functions:

    (1) We support a fit function, where given feature matrix X and label vector y,
        we estimate an Ordinary Least Squares estimator that accounts for hallucinations.

        (a) If 'data_dependent' is False, we obtain an estimator under the assumption that
            the true join's features follow the same distribution as the set of all features. 

        (b) If 'data_dependent' is True, we obtain an estimator that does not make the 
            above assumption, and implements a data-dependent correction. 

    (2) We also support a predict function, where we predict the labels for features X. 
'''

import numpy as np
from math import exp

def est_size(full_data_size, dp_join_size, hall_prob):
    '''
    Estimate the true intersection size using the number of features and the private join size
    Use the following equation:
        (private join size) = (1 - hall_prob) * (true size) + hall_prob * (total features - true size)
    '''
    return (dp_join_size - hall_prob * full_data_size) / (1 - 2.0 * hall_prob)

class AdjustedLinearRegression:
    def __init__(self, data_dependent = True):
        '''
        The AdjustedLinearRegression object stores the following:

            - data_dependent:   A boolean variable. If 'False', we follow the assumptions of 
                                    (1a) above. Otherwise, we follow assumptions of (1b).
            - estimator:        A matrix that represents the Ordinary Least Squares estimator
        '''
        self.data_dependent = data_dependent
        self.estimator = None

    def fit(self, X, y, eps_memb, full_X = None):
        '''
            Given the features X and the labels y as pandas DataFrames, we compute a 
            modified version of the Ordinary Least Squares estimator. 
        '''

        # Compute the hallucination probability
        hall_prob = exp(-1.0 * eps_memb) / (1 + exp(-1.0 * eps_memb))
        
        # Convert everything to numpy
        X_mat, y_vec = X.to_numpy(), y.to_numpy()

        # This block of code implements the oblivious correction
        if not self.data_dependent:
            full_data_size, dp_join_size = len(full_X), len(X)
            join_size = est_size(full_data_size, dp_join_size, hall_prob)
            scale = (dp_join_size - hall_prob * (full_data_size - join_size)) / (dp_join_size) 

            cov = scale * np.matmul(X_mat.T, X_mat)
        # This block of code implements the data-dependent correction
        else:   
            full_X_mat = full_X.to_numpy() 
            cov = (np.matmul(X_mat.T, X_mat) - hall_prob * np.matmul(full_X_mat.T, full_X_mat)) \
                * (1 - hall_prob) / (1 - 2 * hall_prob)
        
        self.estimator = np.linalg.solve(cov, np.matmul(X_mat.T, y_vec))

    def predict(self, X):
        '''
        Given features X as a numpy matrix, output the vector of predictions
        made using 'self.estimator'.
        '''
        return np.matmul(X, self.estimator)
