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

def get_sqdistance_matrix(M1, M2, bandwidth = 1.0):
    allsqnorms = np.linalg.norm(np.vstack([M1,M2]), axis=1).reshape(-1, 1) ** 2
    M1sqnorms = allsqnorms[:M1.shape[0], :]
    M2sqnorms = allsqnorms[M1.shape[0]:,:].reshape(1, -1)
    dm = M1sqnorms + M2sqnorms - 2.0 * np.dot(M1, M2.T)
    dm[dm < 0.0] = 0.0
    return dm / (bandwidth ** 2)

def gaussianKernelMatrix(dataset, queries, bandwidth = 1.0):
    return np.exp(-1 * get_sqdistance_matrix(dataset, queries, bandwidth))

def get_features(dataset, queries, num_output_features = 0, bandwidth = 1.0):
    mat = gaussianKernelMatrix(dataset, queries, bandwidth)
    if num_output_features == 0:
        return mat
    else:
        eigvals, eigvecs = np.linalg.eigh(mat)
        return eigvecs[:, -1 * num_output_features:]

# TODO: Optimize by saving the kernel matrix
def hrw_mechanism(dataset, epsilon, delta, num_output_features = 0, bandwidth = 1.0):
    n = dataset.shape[0]
    kernel_matrix = gaussianKernelMatrix(dataset, dataset, bandwidth)
    noise_coefficient = 2 * (np.log(2 / delta)) ** 0.5 / epsilon
    noise_matrix = np.zeros(kernel_matrix.shape)
    noise_matrix += noise_coefficient * np.random.multivariate_normal(np.zeros(n), kernel_matrix, n)
    noise_matrix += noise_coefficient * np.random.multivariate_normal(np.zeros(n), kernel_matrix, n).T

    if num_output_features == 0:
        return kernel_matrix + noise_matrix
    else:
        eigvals, eigvecs = np.linalg.eigh(kernel_matrix + noise_matrix)
        return eigvecs[:, -1 * num_output_features:]

