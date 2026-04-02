---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    custom_cell_magics: kql
    text_representation:
      extension: .py
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: base
    language: python
    name: python3
---

```python
#Calculate the Eigenvalues for Each Kernel Matrix
#for a 1000 Randomly Generated Data Sets and Check If Any 
#Eigenvalues Are Negative.
```

```python

#K(x,y)=math.exp(-gamma*np.abs(np.linalg.norm(x)-np.linalg.norm(y)))
```

```python
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import SpectralClustering
import math #generate some random data def generate_points_on_circle(radius, num_points,noise):
from matplotlib import pyplot as plt
import math #generate some random data def generate_points_on_circle(radius, num_points,noise):

def generate_points_on_circle(radius, num_points,noise):
    xpoints=[] 
    ypoints=[]
    #np.random.seed(37)
    for _ in range(num_points):
    # Generate a random angle in radians
        theta =np.random.uniform(0, 2 * math.pi)

    # Calculate x and y coordinates using polar coordinates

        x =(radius+np.random.normal(0,noise,1)) * math.cos(theta)

        y =(radius+np.random.normal(0,noise,1)) * math.sin(theta)

        xpoints.append(x)
        ypoints.append(y)

    return np.array(xpoints), np.array(ypoints)

#Parameteers
categories=3 
factor=25
radius=[1,4,6]
noise=.5
count=0
for t in range(0,999):
    
    #Contruct X(:,0) annd X(:,1)
    Xp=np.empty((0,1))
    Yp=np.empty((0,1))

    for c in range(categories):
        num_points=factor*radius[c]
        xpoints,ypoints = generate_points_on_circle(radius[c],int(num_points),noise)
        Xp=np.vstack((Xp,xpoints))
        Yp=np.vstack((Yp,ypoints))

    X=np.hstack((Xp,Yp))
    def custom_kernel(x,y,gamma=.5):
        return math.exp(-gamma*np.abs(np.linalg.norm(x)-np.linalg.norm(y)))
    affinity_matrix = pairwise_kernels(X,X, metric=custom_kernel)

    eigvalues,eigenvectors=np.linalg.eigh(affinity_matrix)
    #print(np.min(eigvalues))
    if np.any(eigvalues<0)==True:
        count=count+1
print(count!=0)
```

```python
#K(x,y)=np.abs(np.cos(gamma*np.linalg.norm(x-y))) +epsilon*kdelta(x,y)
#where kdelta is the Kronecker delta function
```

```python
import math as m
import numpy as np
#Parameteers
num_points=100
angles=[0,2*m.pi/3,4*m.pi/3]
noise=.5
#generate some random data def generate_points_on_circle(angles, num_points,noise):
def generate_points_on_circle(angles, num_points,noise):
    xpoints=[]
    ypoints=[]
    for _ in range(num_points):
        # Generate a random angle in radians randi =np.random.randint(0,3)
        randi =np.random.randint(0,3)
        # Calculate x and y coordinates using polar coordinate
        x = np.sqrt(np.random.uniform(0,10)) * np.cos(angles[randi])+np.random.normal(0,noise)
        y = np.sqrt(np.random.uniform(0,10)) * np.sin(angles[randi])+np.random. normal(0,noise)

        xpoints.append(x)
        ypoints.append(y)

    return np.array(xpoints), np.array(ypoints) #Parameteers num_points=100 angles=[0,2*m.pi/3,4*m.pi/3] noise=.5

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import SpectralClustering

def custom_kernel(x,y,gamma=.1):
    return np.abs(np.cos(gamma*np.linalg.norm(x-y)))


epsilon=.2
count=0
for t in range(0,999):
    xpoints,ypoints = generate_points_on_circle(angles,num_points,noise)
    X=(np.vstack((xpoints,ypoints))).T
    affinity_matrix = pairwise_kernels(X,X, metric=custom_kernel)+epsilon*np.identity(np.shape(X)[0])
    eigvalues,eigenvectors=np.linalg.eigh(affinity_matrix)
    #print(np.min(eigvalues))
    if np.any(eigvalues<0)==True:
        count=count+1
print(count!=0)

```

```python
# K(x,y)=math.exp(-gamma*abs(m*x[0]-x[1]-m*y[0]+y[1])*(1+m**2)**(-1/2))
# were m is slope of the linear regression through cluster that the largest
# eccentricity, i.e. third cluster.
```

```python
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import numpy as np
import math

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import SpectralClustering

categories=3
count=0
for t in range(0,999):
    X, Y = make_classification(n_features=2, n_redundant=0, random_state=1,n_informative=2,n_clusters_per_class=1,n_classes=3)
    clustering = SpectralClustering(n_clusters=3, affinity='rbf', gamma=10,random_state=0).fit(X)
    index=np.where(clustering.labels_==2)[0]
    Xprime=np.hstack((X[index,0].reshape(-1,1),np.ones((len(index),1))))
    slope=np.linalg.pinv(Xprime)@X[index,1]
    def custom_kernel(x,y,m=slope[0],gamma=.5):
        return math.exp(-gamma*abs(m*x[0]-x[1]-m*y[0]+y[1])*(1+m**2)**(-1/2))
    affinity_matrix = pairwise_kernels(X,X, metric=custom_kernel)
    eigvalues,eigenvectors=np.linalg.eigh(affinity_matrix)
    #print(np.min(eigvalues))
    if np.any(eigvalues<0)==True:
        count=count+1
print(count!=0)
```
