
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
sns.set_style("ticks")
n = 250
x = 10* np.random.uniform(size = n)
y = np.sin(2 * np.pi * x) + np.random.normal(size = n) / 10
def K(x, y, sigma2) :
    return np.exp(-np. linalg .norm(x - y)**2/2/sigma2)

def F(z, sigma2) :
    S=0; T=0
    for i in range(n) :
        S = S + K(x[i] , z, sigma2) * y[i]
        T = T + K(x[i] , z, sigma2)
    return S / T
plt.title('Nadaraya–Watson Kernel Regression')
ax = plt.gca() # Get the current axes
ax.set_aspect(2)
z=np.linspace(int(np.floor(min(x))),int(np.ceil(max(x))),num=2000)
plotz=np.zeros(len(z))

for i in range(len(z)):
    plotz[i]=F(z[i],.001)

plt.plot(z,plotz)
