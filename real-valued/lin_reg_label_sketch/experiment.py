''' 
This class provides a framework for running experiments on both ordinarily joined 
features and labels data, and on privately joined features and labels data. 

The main function that gets called is 'run_dp_sketch_experiments'. It simulates
'num_trials' number of private joins, and then runs every linear regression algorithm 
in 'self.experiment_list' on each of them. 

It computes the median mean squared error loss of each algorithm using the 'get_loss' function.
'''

import pandas as pd
from statistics import median

from dp_sketch import DP_Join
from sklearn import metrics 
from sklearn.linear_model import LinearRegression
from regression_adjusted import AdjustedLinearRegression

unmodified_experiment = 'Linear Regression'

# Initialize objects for performing linear regression tasks. 
# Each object comes with a fit() and a predict() function.
method_to_obj = {'Linear Regression': LinearRegression(fit_intercept = False),
    'Adjusted Regression - Oblivious': AdjustedLinearRegression(data_dependent = False),
    'Adjusted Regression - Data Dep.': AdjustedLinearRegression(data_dependent = True)}

def get_loss(reg, f_test, l_test):
    '''
    Compute the MSE loss on 'self.f_test' and 'self.l_test' for the model 'reg'.
    '''
    pred = reg.predict(f_test.to_numpy())
    return metrics.mean_squared_error(l_test, pred) 

class Experiment:
    def __init__(self, experiment_list, f_train, l_train, f_test, l_test, f_names, l_name):
        '''
        The Experiment object stores the following:

            - experiment_list:  A list of strings that correspond to different versions of 
                                    linear regression to run on joined data. 
                                    These need to exactly match the keys in 'method_to_obj' in
                                    experiment.py
            - f_train:          A DataFrame with the training set features.
            - l_train:          A DataFrame with the training set labels.
            - f_test:           A DataFrame with the test set features.
            - l_test:           A DataFrame with the test set labels.
            - f_names:          A list of column names corresponding to features.
            - l_name:           A list consisting of the label column name
            - ctrl:             The true join between f_train and l_train
            - dp:               The private join between f_train and l_train. It gets 
                                    repopulated every trial in 'run_dp_sketch_experiments'. 
        '''
        self.experiment_list = experiment_list
        self.f_train = f_train
        self.l_train = l_train
        self.f_test = f_test
        self.l_test = l_test
        self.f_names = f_names
        self.l_name = l_name 
        self.ctrl = f_train.join(l_train, how = 'inner')
        self.dp = None

    def fit_model(self, experiment, params = {}):
        '''
        Fits a model, given by 'experiment', on 'self.f_train' and 'self.l_train'.
        We need to pass in additional parameters when running AdjustedLinearRegression.
        '''
        reg = method_to_obj[experiment]
        if experiment == unmodified_experiment:
            reg.fit(self.dp.df_joined[self.f_names].to_numpy(), self.dp.df_joined[self.l_name])
        else:
            reg.fit(self.dp.df_joined[self.f_names], self.dp.df_joined[self.l_name], **params)
        return reg

    def run_dp_sketch_experiments(self, sens, eps_memb, eps_val, num_trials = 25):
        '''
        Computes the loss for each experiment in 'self.experiment_list' for 'num_trials' trials. 
        In each trial, we run an independent private join of 'self.f_train' and 'self.l_train'
    
        Returns a dictionary where the keys are the experiment name and the values are 
        the median MSE loss. 
        '''
        trial_dict = {}
        for experiment in self.experiment_list:
            trial_dict[experiment] = []

        for trial in range(num_trials):
            print('Trial Number %i' % (trial + 1))
            # Perform the DP join. 
            self.dp = DP_Join(eps_memb, eps_val, sens)
            self.dp.join(self.f_train, self.l_train)
            self.dp.drop_entries()

            # Compute the loss experienced by the model on the DP join
            params = {'eps_memb': eps_memb,
                'full_X': self.f_train[self.f_names]} 
            for experiment in self.experiment_list:
                reg = self.fit_model(experiment, params)
                trial_dict[experiment].append(get_loss(reg, self.f_test, self.l_test))

        loss_dict = {}
        # Compute the median loss for each experiment
        for experiment in self.experiment_list:
            loss_dict[experiment] = median(trial_dict[experiment])
        return loss_dict
