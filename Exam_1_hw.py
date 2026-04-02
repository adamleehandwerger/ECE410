import numpy as np
np.set_printoptions(precision=1)
def phi(i):
    M=np.array([[1,0,.5],[0,1,.5],[.5,.5,1]])
    eigenvalues,eigenvectors=np.linalg.eig(M)

    return np.array([0, np.sqrt(eigenvalues[0])*eigenvectors[i,0],np.sqrt(eigenvalues[1])\
        *eigenvectors[i,1],np.sqrt(eigenvalues[2])*eigenvectors[i,2]])
Phi=[np.zeros([1,4]),*[phi(i) for i in range(3)]]
K=np.zeros([4,4])
for i in range(4):
    for j in range(4):
        K[i,j]=np.round(np.dot(np.array(Phi[i]),np.array(Phi[j]).T),4)


