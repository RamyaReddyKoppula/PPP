import numpy as np
import math
import matplotlib #scientific plots
import matplotlib.pyplot as plt
data_SFE=np.genfromtxt('data_nor.csv',delimiter=',')
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
mat_reduced = PCA(data_SFE , 3)
fig = plt.figure(figsize=(10,8))
#ax = fig.add_subplot(1,2,1, projection='3d')
ax = plt.axes(projection ="3d")
#gs = gridspeci.GridSpec(1, 2,width_ratios=[10,2])
#ax = plt.subplot(gs[0], projection='3d')
#ax2 = plt.subplot(gs[1])


ax.scatter(mat_reduced[:, 0], mat_reduced[:, 1], mat_reduced[:, 2], s=50, c="red" ,  edgecolor='', cmap='brg', depthshade=False)
ax.set_xlabel("1st eigenvector")
ax.set_xlim(mat_reduced[:,0].min()-0.1,mat_reduced[:,0].max()+0.1)
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.set_ylim(mat_reduced[:,1].min()-0.1,mat_reduced[:,1].max()+0.1)
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.set_zlim(mat_reduced[:,2].min()-0.1,mat_reduced[:,2].max()+0.1)
ax.w_zaxis.set_ticklabels([])
ax.view_init(15,120)
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
# blue_patch = mpatches.Patch(color=[0,0,1], label='Low SFE')
# red_patch = mpatches.Patch(color=[1,0,0], label='Medium SFE')
# green_patch = mpatches.Patch(color=[0,1,0], label='High SFE')'
#plt.axis('off')
#ax2.legend(handles=[blue_patch,red_patch,green_patch ], loc='lower right')
plt.show()