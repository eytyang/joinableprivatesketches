import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

algs = ['AdaBoost']
num_trials = 25
total_eps_list = [1.0, 2.0, 3.0, 4.0, 5.0]
for alg in algs:
	alg_df = pd.read_csv('covtype67_pca_centeredclip_%s_trials=%i.csv' % (alg.lower(), num_trials))
	alg_df = alg_df.set_index('Dimension')
	print(alg_df)

	file = 'covtype67_pca_centeredclip_%s_trials=%i' % (alg.lower(), num_trials)
	alg_df.to_csv('%s.csv' % file)
	shift = -0.25

	# Edit!
	plt.ylim((0.45, 1.05))
	plt.errorbar(alg_df.index + shift, alg_df['Original Features'], \
		yerr = np.zeros(shape = (2, len(alg_df))), label = 'Original Features', linestyle = 'dashed')
	plt.errorbar(alg_df.index + shift, alg_df['PCA'], \
		yerr = np.zeros(shape = (2, len(alg_df))), label = 'PCA (No Privacy)')
	# plt.errorbar(alg_df.index + shift, alg_df['RFF Real'], \
	# 	yerr = alg_df[['RFF Real 25', 'RFF Real 75']].to_numpy().T, label = 'Real RFFs (No Privacy)')
	shift += 0.05
	for total_eps in total_eps_list:
		if total_eps == 2.0 or total_eps == 4.0:
			continue
		plt.errorbar(alg_df.index + shift, alg_df['Eps = %s' % str(total_eps)], \
			yerr = alg_df[['Eps = %s 25' % str(total_eps), 'Eps = %s 75' % str(total_eps)]].to_numpy().T, label = 'Eps = %s' % str(total_eps))
		shift += 0.05

	plt.xlabel("Reduced Dimension")
	plt.ylabel("Accuracy")
	plt.title('Accuracy of AdaBoost on Forest Cover, 6 vs. 7 (PCA)')
	plt.legend(loc = "lower left")
	plt.savefig('%s.jpg' % file)
	plt.close()