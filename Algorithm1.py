import random as 
import numpy as np
from matplotlib import pyplot as plt
ss=20
x=np.array(range(1,ss+1))
xp=np.array(r.sample(range(25,50), int(ss/2)))
xn=np.array(r.sample(range(1,25),int(ss/2)))
xt=np.concatenate((np.array(xp),np.array(xn)))
X=np.vstack((x,xt))
y=np.concatenate((np.ones((len(xp))),-np.ones((len(xn)))))
l=np.size(y)
plt.scatter(x[0:np.size(xp)],xp,color='r',facecolors='none')
plt.scatter(x[np.size(xn):None],xn,color='b',facecolors='none')
#algorithm
xg=np.linspace(0,np.size(x),l)
yg=np.linspace(0,np.max(xp),l)
w=y[0]*X[:,0]
for i in r.sample(list(range(0,l)),l-1):
    if y[i]*np.dot(w,X[:,i])<0:
        w=w+y[i]*X[:,i]
       
#plt.plot(xg,s*xg,color='k')  
n=1000
xg=np.linspace(0,np.size(x),n)
yg=np.linspace(0,np.max(xp),n)
V,W=np.meshgrid(xg,yg)
coordinates = np.stack((V, W), axis=-1)
Z=np.array(np.zeros((np.size(xg),np.size(yg))))
for i in range(0,n):
   for j in range(0,n):
       Z[i,j]=np.dot(w,coordinates[i,j,:])
plt.scatter(x[0:np.size(xp)],xp,color='r',facecolors='none')
plt.scatter(x[np.size(xn):None],xn,color='b',facecolors='none')
s=-w[0]/w[1]
plt.plot(xg,s*xg,color='k')
plt.xlim(0,np.size(xp)+10)
plt.ylim(-10,np.max(xp)+10)
plt.contour(V,W,Z)
plt.show()






