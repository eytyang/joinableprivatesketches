import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None  

from sklearn.model_selection import train_test_split
from experiment import Experiment

# Function for centering columns of a DataFrame
def center(df, cols):
	for col in cols:
		df[col] = df[col] - df[col].mean() 
	return df

def prep_data(file, l_name, index_name = None, f_names = None, test_size = 0.2, center_data = False):
	# Load dataset 
	if index_name is not None:
		df = pd.read_csv(file).set_index(index_name)
	else:
		df = pd.read_csv(file)
	
	if f_names is None:
		f_names = list(df.columns)
		f_names.remove(l_name[0])
	if center_data:
		df = center(center(df, f_names), l_name)

	df_train, df_test = train_test_split(df, test_size = test_size)
	f_train, l_train = df_train[f_names], df_train[l_name]

	if center_data:
		f_test, l_test = center(df_test[f_names], f_names), center(df_test[l_name], l_name)
	else:
		f_test, l_test = df_test[f_names], df_test[l_name]
	return f_train, l_train, f_test, l_test

def uniform_subsample(df_known, df_private, frac_keep):
	# Only keep a random gamma fraction of labels
	if frac_keep < 1.0:
		num_instances = len(df_known)
		private_indices = np.random.choice(df_private.index.to_numpy(), size = round(num_instances * frac_keep), replace = False)
		df_private = df_private.loc[private_indices]
	return df_private

def plot_results(results, file, reduced_features, total_features):
	print(results)
	print()
	results.to_csv('%s.csv' % file)
	shift = -0.02
	sketch_types = ['Ind', 'Dep']
	if reduced_features == 1:
		sketch_types = ['Ind']
	feat_types = ['Unif', 'NonUnif']
	if reduced_features == total_features:
		feat_types = ['Unif']

	for sketch_type in sketch_types:
		for feat_type in feat_types:
			dp_type = '%s_%s' % (sketch_type, feat_type)
			plt.errorbar(results.index + shift, results[dp_type], \
				yerr = results[[dp_type + ' 25', dp_type + ' 75']].to_numpy().T, label = dp_type)
			shift += 0.01

	plt.xlabel("Epsilon")
	plt.ylabel("(Loss With DP) / (Actual Loss)")
	plt.legend(loc = "lower right")
	plt.savefig('%s.jpg' % file)
	plt.close()