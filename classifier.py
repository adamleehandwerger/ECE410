import random as r
import numpy as np
from matplotlib import pyplot as plt
ss=20
x=np.array(range(1,ss+1))
xp=np.square(np.array(r.sample(range(20,50), int(ss/2))))
xn=np.square(np.array(r.sample(range(1,40),int(ss/2))))
xt=np.concatenate((np.array(xp),np.array(xn)))
X=np.vstack((x,xt))
y=np.concatenate((np.ones((1,len(xp))),-np.ones((1,len(xn)))),axis=1)
plt.scatter(x[0:np.size(xp)],xp,color='r',facecolors='none')
plt.scatter(x[np.size(xn):None],xn,color='b',facecolors='none')

def phi1(x):
    phi=np.array([x[0]**2,np.sqrt(2)*x[0]*x[1],x[1]**2])
    return phi
w=np.array([0,0,0])
for i in range(0,ss):
    w=w+y[0,i]* phi1(X[:,i])
n=1000
xg=np.linspace(0,np.size(x),n)
yg=np.linspace(0,np.max(xp),n)
V,W=np.meshgrid(xg,yg)
coordinates = np.stack((V, W), axis=-1)
Z=np.array(np.zeros((np.size(xg),np.size(yg))))
for i in range(0,n):
    for j in range(0,n):
        Z[i,j]=np.dot(w,phi1(coordinates[i,j,:]))

plt.contour(V,W,Z)
plt.show()





