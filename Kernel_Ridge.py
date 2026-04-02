import numpy as np
import pandas as pd
import cvxopt
from cvxopt import solvers
from cvxopt import matrix
import matplotlib.pyplot as plt
import seaborn
from numpy.random import randn # Gaussian
from scipy.stats import norm

kapha=0.1;delta=1 #create kernel functions
def alpha(k,x,y):
    n=len(x)
    lam=0.01
    K=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j]=k(x[i],x[j])
    return np.linalg.inv(K+lam*np.identity(n)).dot(y)

def k_p(x,y):   #polynomial kernel
    return (np.dot(x.T,y)+1)**3

def k_g(x,y):   #exponential kernel
    return np.exp(-(x-y)**2/2)

def k_s(x,y):   #sigmoid kernel
    return np.tanh(kapha*np.dot(x.T,y)-delta)

n=50 #number of data points
beta=0.5 #noise scaling
lam=0.01 #ridge scaling
x=np.random.randn(n) #gernerate data point
y=.25*(1+x+x**3+beta*np.random.randn(n)) #generate fake response

alpha_p=alpha(k_p,x,y)
alpha_g=alpha(k_g,x,y)
alpha_s=alpha(k_s,x,y)

z=np.sort(x);l=[];u=[];v=[];w=[] #create kernel for each data point
for j in range(n):
    S=0
    for i in range(n):
        S=S+alpha_p[i]*k_p(x[i],z[j])
    u.append(S)
    S=0
    for i in range(n):
        S=S+alpha_g[i]*k_g(x[i],z[j])
    v.append(S)
    S=0
    for i in range(n):
        S=S+alpha_s[i]*k_s(x[i],z[j])
    w.append(S)
    
onevec=np.ones((n,))
lam2=.01
X=np.stack((x,onevec),axis=1) #create matrix for linear regression
invmat=np.linalg.inv(np.matmul(X,X.T)+lam2*np.identity(n))
slope=np.matmul(invmat,X).T.dot(y) #find least squares solution with ridge

plt.scatter(x,y,facecolors='none',edgecolors='k',marker='o') #plot data points
seaborn.set_theme()
####plot each kernel and linear regression together
plt.plot(z,(slope[0]*z+slope[1]),c='k',label='Linear Ridge Regression')
plt.plot(z,u,c='r',label='Polynomial Kernel and Ridge')
plt.plot(z,v,c='b',label='Gaussian Kernel and Ridge')
plt.plot(z,w,c='g',label='Sigmoid Kernel and Ridge')
plt.xlim(-2,2)
plt.ylim(-1,2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ridge Regression using some Common Kernels')
plt.legend(loc='upper left',frameon=True,prop={'size':12})