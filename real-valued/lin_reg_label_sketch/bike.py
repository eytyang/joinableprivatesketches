'''
This file runs all of the different experiments from experiment.py on hourly bikeshare data,
downloaded from the UCI Machine Learning Repository. Instructions for downloading the data 
are in the README file. 

We set the following parameters to start:
- experiment_list:  The list of experiments from experiment.py to test. 
                        These need to exactly match the keys in 'method_to_obj' in
                        experiment.py
- num_trials:       The number of independent private joins to test before taking
                        the median loss of each experiment.
- data_file_name:   The path to the hourly bike data.
- f_names:          A list of columns corresponding to features. In general, can contain
                        more than one column. 
- l_names:          A list consisting only of the label_name. 
- ysens:            In this case, it is a generous upper bound on the magnitude of 
                        label values. In the future, we could set this using some 
                        private maximum computation. 
- frac_labels_keep: A real number between 0 and 1 that tells us what fraction of the
                        training labels to keep. This forces us to hallucinate some features. 
''' 

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression 

from experiment import Experiment, get_loss
from experiment_utils import prep_data, uniform_subsample, plot_results

if __name__ == "__main__": 
    experiment_list = ['Linear Regression', 'Adjusted Regression - Oblivious', 'Adjusted Regression - Data Dep.'] 
    num_trials = 25
    data_file_name = 'data/hour.csv'
    f_names, l_name = ['atemp'], ['cnt']
    y_sens = 1000
    frac_labels_keep = 0.25
    
    # Pre-process the data
    f_train, l_train, f_test, l_test = prep_data(data_file_name, f_names, l_name, center_data = True)
    l_train = uniform_subsample(f_train, l_train, frac_labels_keep)

    # Train on the control join (no privacy)
    join_ctrl = f_train.join(l_train, how = 'inner')
    reg_ctrl = LinearRegression(fit_intercept = False)
    reg_ctrl.fit(join_ctrl[f_names].to_numpy(), join_ctrl[l_name])
    loss_ctrl = get_loss(reg_ctrl, f_test, l_test)

    # Run experiments for every epsilon
    experiment = Experiment(experiment_list, f_train, l_train, f_test, l_test, f_names, l_name)
    results_df = pd.DataFrame()
    eps_list = [1.0, 2.0, 3.0, 4.0, 5.0] 
    for eps in eps_list:
        print('Epsilon: %s' % str(eps))
        eps_memb = eps * 0.5
        eps_val = eps * 0.5

        loss_dict = experiment.run_dp_sketch_experiments(y_sens, eps_memb, eps_val, num_trials)
        loss_dict['Epsilon'] = [eps]

        eps_df = pd.DataFrame(loss_dict).set_index('Epsilon')
        results_df = pd.concat([results_df, eps_df])
        print()

    results_df = results_df / loss_ctrl
    file_name = f'bikedaily_centered_avgloss_sens={y_sens}_trials={num_trials}_rand{frac_labels_keep}'
    plot_results(results_df, file_name)
