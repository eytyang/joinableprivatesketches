import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

algs_to_string = {
	'KNN': 'KNN', 
	'AdaBoost': 'AdaBoost', 
	'RandomForest': 'Rand. For.',
	'LogisticRegression': 'Log. Reg.', 
	'MultiLayerPerceptron': '2-Layer NN'
}
dataset_to_string = {
	'mnist49': 'MNIST 4/9',
	'covtype67': 'Forest Cover 6/7', 
	'epileptic': 'Epilepsy Dataset'
}

dataset = 'covtype67'
algs = ['RandomForest'] # ['AdaBoost', 'RandomForest', 'KNN']
# algs = ['LogisticRegression', 'MultiLayerPerceptron']
num_trials = 25
total_eps_list = [1.0, 3.0, 5.0]
for alg in algs:
	file = '%s_rffrealclip_%s_trials=%i_copy' % (dataset, alg.lower(), num_trials)
	alg_df = pd.read_csv('%s.csv' % file)
	alg_df = alg_df.set_index('Dimension')
	print(alg_df)

	shift = -0.25
	# Edit!
	plt.ylim((0.45, 1.05))
	plt.xlim((0, 25))
	plt.errorbar(np.array([1, 10, 15, 20, 25]) + shift, alg_df['Original Features'], \
		yerr = np.zeros(shape = (2, len(alg_df))), label = 'Original Features', linestyle = 'dashed')
	# plt.errorbar(alg_df.index + shift, alg_df['PCA'], \
	# 	yerr = np.zeros(shape = (2, len(alg_df))), label = 'Reduced Feats. (No Privacy)')
	plt.errorbar(alg_df.index + shift, alg_df['RFF Real'], \
	  	yerr = alg_df[['RFF Real 25', 'RFF Real 75']].to_numpy().T, label = 'Real RFFs (No Privacy)')
	shift += 0.05
	for total_eps in total_eps_list:
		plt.errorbar(alg_df.index + shift, alg_df['Eps = %s' % str(total_eps)], \
			yerr = alg_df[['Eps = %s 25' % str(total_eps), 'Eps = %s 75' % str(total_eps)]].to_numpy().T, label = 'Eps = %s' % str(total_eps))
		shift += 0.05

	plt.xlabel("# RFFs")
	plt.ylabel("Accuracy")
	# plt.title('Accuracy of %s on %s (PCA + Memb Unknown)' % (algs_to_string[alg], dataset_to_string[dataset]))
	# plt.legend(loc = "upper right")
	plt.savefig('%s.jpg' % file)
	plt.close()