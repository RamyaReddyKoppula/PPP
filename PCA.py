import numpy as np
import math
import matplotlib #scientific plots
import matplotlib.pyplot as plt
#data_SFE=np.genfromtxt('data_nor.csv',delimiter=',')

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

def PCA(data , n_components):
     #mean- each row w.r.t coloumn
    m = data - np.mean(data , axis = 0)
     #covariance matrix
    cm = np.cov(m , rowvar = False)
     
    #Eigenvalues and Eigenvectors
    eigenvalues , eigenvectors = np.linalg.eigh(cm)
    #linalg.eigh( ) gives eigenvalues and eigenvectors of a complex Hermitian (conjugate symmetric) or a real symmetric matrix
    #arranging the eigenvalues and eigenvectors in descending order
    arranged_Index = np.argsort(eigenvalues)[::-1]
    arranged_eigenvalues = eigenvalues[arranged_Index]
    arranged_eigenvectors = eigenvectors[:,arranged_Index]
     
    #select the first n eigenvectors, n is desired dimension
    eigenvector_ncom = arranged_eigenvectors[:,0:n_components]
     
   #finally reducing the dimension
    reduced_dim = np.dot(eigenvector_ncom.transpose() , m.transpose() ).transpose()
     
    return reduced_dim