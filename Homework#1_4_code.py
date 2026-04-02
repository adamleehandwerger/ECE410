import random as r
import numpy as np
from matplotlib import pyplot as plt
ss=20
x=np.array(range(1,ss+1))
xp=np.array(r.sample(range(20,50),int(ss/2)))
xn=np.array(r.sample(range(1,30),int(ss/2)))
xt=np.concatenate((np.array(xp),np.array(xn)))
X=np.vstack((x,xt))
y=np.concatenate((np.ones((len(xp))),-np.ones((len(xn)))))
l=np.size(y)
plt.scatter(x[0:np.size(xp)],xp,color='r',facecolors='none')
plt.scatter(x[np.size(xn):None],xn,color='b',facecolors='none')
xg=np.linspace(0,np.size(x),l)
yg=np.linspace(0,np.max(xp),l)
alpha=np.zeros(np.size(y))
#Kernels
##Kernel,d=1
#flag=1
#def K(x,y):
    #return np.dot(x,y)
#Kernel,d=1,c=1
#flag=2
#def K(x,y):
    #return np.dot(x,y)+1
#def K(x,y):
    #return (np.dot(x,y)+1)
#def K(x,y)
#Kernel, guassian
flag=3
def K(x,y):
    global sigma 
    sigma= 1
    XminusYSquared =np.sum(x*x,axis=0)\
    -2*np.dot(x,y.T)+np.sum(y*y,axis=0)

    return np.exp(-XminusYSquared/(2*sigma**2))

#learn objective function from the data
alpha=.1*np.ones(np.size(y))
alpha[0]=y[0]
yhat=0

for k in range(10):
    yhat=0
    for i in r.sample(list(range(1,l)),l-1):
        for j in range(0,l):
            yhat = yhat + alpha[j]*K(X[:,i],X[:,j])
        if y[i]*yhat<0:
            alpha[i]=alpha[i]+y[i]

#create grid for plotting level sets of objective function
n=250
xg=np.linspace(0,np.size(x),n)
yg=np.linspace(0,np.max(xp),n)
V,W=np.meshgrid(xg,yg)
coordinates = np.stack((V, W), axis=-1)
Z=np.array(np.zeros((np.size(xg),np.size(yg))))

for i in range(0,n):
   for j in range(0,n):
        f=0
        for k in range(0,l):
            f=f+alpha[k]*K(X[:,k],coordinates[i,j,:])
        Z[i,j]=f
plt.scatter(x[0:np.size(xp)],xp,color='r',facecolors='none')
plt.scatter(x[np.size(xn):None],xn,color='b',facecolors='none')
#plot
plt.xlim(0,np.size(xp)+10)
plt.ylim(-10,np.max(xp)+10)
plt.contour(V,W,Z,levels=5)
plt.xlabel('X0')
plt.ylabel('X1')
if flag==1:
    title='Linear Kernel'
if flag==2:
    title='Affine Kernel'
if flag==3:
    title='Perceptron Classifier with Guassian Kernel'
plt.title(title)
plt.show()
pfig, ax = plt.subplots(subplot_kw={"projection": "3d"})
from matplotlib import colormaps as cmaps
from matplotlib.colors import CenteredNorm
ax.plot_surface(V, W,Z,cmap='seismic',norm=CenteredNorm()) #ignore
plt.show()