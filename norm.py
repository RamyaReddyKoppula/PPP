
import numpy as np
# read csv file to np
data_SFE=np.genfromtxt('data_nor.csv',delimiter=',')
# Find the min and max values for each column
def dataset_MinMax(dataset):
	Min_Max = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		Min_value = min(col_values)
		Max_value = max(col_values)
		Min_Max.append([Min_value, Max_value])
	return Min_Max
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, min_max):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])
 
Min_Max = dataset_MinMax(data_SFE)
# Normalize columns
normalize_dataset(data_SFE, Min_Max)
print(data_SFE[0])
