import numpy as np
import math
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