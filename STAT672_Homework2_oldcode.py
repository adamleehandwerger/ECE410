import math
import random
import numpy as np
import matplotlib.pyplot as plt

#generate some random data
def generate_points_on_circle(radius, num_points,noise):
    xpoints=[]
    ypoints=[]
    for _ in range(num_points):
        # Generate a random angle in radians
        theta = random.uniform(0, 2 * math.pi)
        
        # Calculate x and y coordinates using polar coordinates
        x =  (radius+np.random.normal(0,noise,1)) * math.cos(theta)
        y =  (radius+np.random.normal(0,noise,1)) * math.sin(theta)
        
        xpoints.append(x)
        ypoints.append(y)
    
    return np.array(xpoints), np.array(ypoints)
#Parameteers
categories=3
factor=25
radius=[1,4,6]
color=['green','green','green']
noise=.3
#Contruct X(:,0) annd X(:,1)
Xp=np.empty((0,1))
Yp=np.empty((0,1))

for c in range(categories):
    num_points=20*radius[c]
    xpoints,ypoints = generate_points_on_circle(radius[c], int(num_points),noise)
    Xp=np.vstack((Xp,xpoints))
    Yp=np.vstack((Yp,ypoints))
    #Scatter plot each color category
    #plt.scatter(xpoints,ypoints,s=2,color=color[c]) 
plt.axis('equal')  # To maintain aspect ratio 
plt.title('Random Points on Circle') 
plt.xlabel('X-axis') 
plt.ylabel('Y-axis')

#Construct X Matrix from data
X=np.hstack((Xp,Yp))
n_samples = X.shape[0]
One_matrix=np.ones((n_samples,n_samples))
Id_matrix=np.identity((n_samples))
#Variety of Kernels for PCA
def polynomial_kernel(X, Y, degree=1, c=0):
    return (np.dot(X, Y) + c) ** degree
def rbf_kernel(X, Y, gamma=.01):
    return np.exp(-gamma * np.linalg.norm(X-Y)**2)
def sigmoid_kernel(X,Y,gamma=1,c=1):
    return np.tanh(gamma*np.dot(X,Y)+c)
#k=sigmoid_kernel
#k=polynomial_kernel
k=rbf_kernel

K=3
iter=3

#initilize with K randomn Points
mu=np.zeros([K,1])
##mu=X[np.random.choice(range(0,n_samples), size=K,replace=False)]
dist=np.zeros([n_samples,K])
#calculate distances to mu and assign points to clusters 
s=np.zeros([n_samples,1])
c=np.zeros([K,1])
for i in range(0,n_samples):
    for j in range(0,K):
        #dist[i,j]=k(X[i,:],X[i,:])-2*k(X[i,:],mu[j])+k(mu[j],mu[j])
        dist[i,j]=np.linalg.norm(X[np.random.choice(range(0,n_samples),1)])
        
for _ in range(1,iter):
    for i in range(0,n_samples):
        s[i]=np.argmin(dist[i])
        c[int(s[i])]=c[int(s[i])]+1
        
    ksum2=0
    for j in range(0,K):
            m=np.where(j==s)[0]
            for p in m:
                for q in m:
                    ksum2=ksum2+k(X[p,:],X[q,:])
    for i in range(0,n_samples):
        for j in range(0,K):
            ksum1=0  
            m=np.where(j==s)[0]
            for p in m:
                ksum1=ksum1+k(X[i,:],X[p,:])
            dist[i,j]=k(X[i,:],X[i,:])-(2/c[j])*ksum1+(1/c[j]**2)*ksum2
    
     
    
    ksum3=0
    G=0
    for l in range(0,K):
        m=np.where(s==l)[0]
        for i in m:
            for j in m:
                ksum3=ksum3+k(X[i,:],X[j,:])
        G=G+(1/c[l])*ksum3
    print(G)
  
    
    
#plt.scatter(mu[:,0],mu[:,1],marker='o',color='k',s=8,facecolors='none')
color=['red','green','blue']
for l in range(0,K):
    plt.scatter(X[np.where(s==l),0],X[np.where(s==l),1],s=2,color=color[l])
    #plt.scatter(mu[l,0],mu[l,1],marker='o',color=color[l],s=15,facecolors='none')
    
    #plt.scatter(X[np.where(s==l),0],X[np.where(s==l),1],s=2,color=color[l])

plt.show()
# from tslearn.clustering import KernelKMeans
# kmeans = KernelKMeans(n_clusters=3, kernel="rbf",kernel_params={gamma=1})
# kmeans.fit(X)
# labels=kmeans.labels_
# for l in range(0,K):
#     plt.scatter(X[np.where(labels==l),0],X[np.where(labels==l),1],s=2,color=color[l])
# plt.show()

            
    



