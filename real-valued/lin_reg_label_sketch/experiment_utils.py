'''
This file compiles a variety of functions for pre- and post- processing the experiment. 
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def center(df, cols):
    '''
    Centers every column of 'df' that is listed in 'cols'
    '''
    for col in cols:
        df[col] = df[col] - df[col].mean() 
    return df

def prep_data(file_name, f_names, l_name, index_name = None, test_size = 0.2, center_data = True):
    '''
    Load the dataset from 'file_name', and perform a train-test split. 
    If 'center_data' is true, center all of the features and the labels in the training and test sets.
    '''
    if index_name is not None:
        df = pd.read_csv(file_name).set_index(index_name)
    else:
        df = pd.read_csv(file_name)

    df_train, df_test = train_test_split(df, test_size = test_size)
    if center_data:
        center(df_train, f_names)
        center(df_train, l_name)
        center(df_test, f_names)
        center(df_test, l_name)
    f_train, l_train = df_train[f_names], df_train[l_name]
    f_test, l_test = df_test[f_names], df_test[l_name]
    return f_train, l_train, f_test, l_test

def uniform_subsample(df_known, df_private, frac_keep):
    '''
    Only keep a random 'frac_keep' fraction of labels, to force hallucinations.
    In a next version, I want to implement keeping a subset of the labels
    based one one of the features, i.e. keep only labels whose feature 'f' has value 'f_val'
    '''
    if frac_keep < 1.0:
        num_instances = len(df_private)
        private_indices = np.random.choice(df_private.index.to_numpy(), size = round(num_instances * frac_keep), replace = False)
        df_private = df_private.loc[private_indices]
    return df_private

def plot_results(results, file_name):
    '''
    Plot the median loss results over the private join trials. 
    Also saves a .csv file of the results. 
    '''
    results.plot()
    results.to_csv('%s.csv' % file_name)
    plt.xlabel("Epsilon")
    plt.ylabel("(Loss With DP) / (Actual Loss)")
    plt.legend(loc = "upper right")
    plt.savefig('%s.jpg' % file_name)
