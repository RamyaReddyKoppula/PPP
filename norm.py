
import numpy as np
# read csv file to np
data_SFE=np.genfromtxt('data_nor.csv',delimiter=',')
# Min and Max
def dataset_MinMax(dataset):
	Min_Max = list()
	for i in range(len(dataset[0])):
		col_val = [row[i] for row in dataset]
		Min_val = min(col_val)
		Max_val = max(col_val)
		Min_Max.append([Min_val, Max_val])
	return Min_Max
 
# Rescaling, or min-max normalization:we scale the data into range: [0,1]
def normalization(dataset, Min_Max):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - Min_Max[i][0]) / (Min_Max[i][1] - Min_Max[i][0])
    
 
Min_Max = dataset_MinMax(data_SFE)
# Normalize columns
normalization(data_SFE, Min_Max)
print(data_SFE[0])
