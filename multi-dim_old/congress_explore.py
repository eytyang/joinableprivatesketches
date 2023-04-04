import numpy as np 
import pandas as pd

from sklearn.cluster import DBSCAN

def simhash(df, cols, vec_len, num_hashes = 1):
	hash_output = {}
	for col in cols:
		hash_output[col] = []
	for num_hash in range(num_hashes):
		gaussian = np.random.normal(size = vec_len)
		for col in cols:
			hash_output[col].append(np.sign(np.dot(df[col].to_numpy(), gaussian)))

	return hash_output

if __name__ == "__main__":
	df = pd.read_csv('house-votes-84.data')
	cols = list(df.columns)

	df['party'] = df['party'].replace(['democrat', 'republican'], [1, -1])
	for col in cols:
		if col == 'party':
			continue
		print('party', col, df['party'].corr(df[col]))
	print()

	cols.remove('party')
	num_cols = len(cols)
	df = df[cols]
	corr_matrix = np.ones((num_cols, num_cols)) - np.absolute(df.corr(method = 'pearson'))
	clusters = DBSCAN(min_samples = 1)
	clusters = clusters.fit(corr_matrix)
	hash_output = clusters.labels_

	# hash_output = simhash(df, cols, len(df), 6)
	# reverse_hash = {}
	# for col in hash_output:
	# 	if hash_output[col][0] == -1:
	# 		hash_output[col] = [-1 * i for i in hash_output[col]]
	# 	hash_string = ','.join([str(item) for item in hash_output[col]])
	# 	if hash_string not in reverse_hash:
	# 		reverse_hash[hash_string] = [col]
	# 	else:
	# 		reverse_hash[hash_string].append(col)
	# print(reverse_hash)

	reverse_hash = {}
	for i in range(len(hash_output)):
		if hash_output[i] not in reverse_hash:
			reverse_hash[hash_output[i]] = []
		reverse_hash[hash_output[i]].append(cols[i])
	
	for key in reverse_hash:
		print(reverse_hash[key])
	print()

	for key in reverse_hash:
		bucket_cols = reverse_hash[key]
		# for col1 in bucket_cols:
		# 	for col2 in bucket_cols:
		# 		if col2 == col1:
		# 			continue
		# 		print(col1, col2, df[col1].corr(df[col2]))

		if len(bucket_cols) == 1:
			continue
		df_bucket = df[bucket_cols]
		row_dict = {}
		for index, row in df_bucket.iterrows():
			row_list = row.values.flatten().tolist()
			row_list = [str(item) for item in row_list]
			row_string = ','.join(row_list)
			if row_string in row_dict:
				row_dict[row_string] += 1
			else:
				row_dict[row_string] = 1 

		print(bucket_cols)
		row_dict = {k: v for k, v in sorted(row_dict.items(), key = lambda item: item[1])}
		for key in row_dict:
			if row_dict[key] > 5:
				print(row_dict[key], key)
				keep_rows = []
				for i in df.index:
					if df[bucket_cols].loc[i].values.flatten().tolist() != [int(i) for i in key.split(',')]:
						keep_rows.append(i)
				df = df.loc[keep_rows]
		
		corr_matrix = np.ones((len(bucket_cols), len(bucket_cols))) - np.absolute(df[bucket_cols].corr(method = 'pearson'))
		clusters = DBSCAN(min_samples = 1)
		clusters = clusters.fit(corr_matrix)
		hash_output = clusters.labels_
		print(hash_output)

