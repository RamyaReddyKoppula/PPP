
import numpy as np
# read csv file to np
data_SFE=np.genfromtxt('data_nor.csv',delimiter=',')
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
		return row

    
def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot
def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy
def Sigmoid(a):
        return 1 / (1 + np.exp(-a))
#x = [0, 2, 1, 3]
#y = [0, 1, 2, 3]
#hr=accuracy_score(x,y)
#print(hr)