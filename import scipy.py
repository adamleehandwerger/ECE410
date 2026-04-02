import scipy.special as sp
import scipy.integrate as integrate
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag


theta_00=9*np.pi/128
x_0=[theta_00,0]
m=np.sin(theta_00/2)**2
omega_0=.1*2.5
K=integrate.quad(lambda z:((1-z**2)*(1-m*z**2))**(-1/2),0,1)[0]
t=range(100)
n_trajectories=15
n_samples=10
F_theta=np.zeros([np.size(t),1])
F_omega=np.zeros([np.size(t),1])
tt=omega_0*np.array(range(0,np.size(t)))
def fx(t,x,omega_0):
    return np.array([x[1],-omega_0**2*np.sin(x[0])])
Gmatrix=[]
for i in range(0,n_trajectories):
    theta_0=i*theta_00
    m=np.sin(theta_0/2)**2
    K=integrate.quad(lambda z:((1-z**2)*(1-m*z**2))**(-1/2),0,1)[0]
    xt=solve_ivp(fx,t_span=(0,np.size(t)),t_eval=[i for i in range(100)],y0=([theta_0,0]),args=(omega_0,),vectorize=True)
    for j in range(np.size(t)):
        F_theta[j]=xt.y[0][j]
        #F_theta[j]=2*np.arcsin(np.sin(theta_0/2)*sp.ellipj(K-omega_0*j,(np.sin(theta_0/2)**2))[0])
        # if xt.y[1][j]>=.01:
        #     F_omega[j]=2*omega_0**2*np.sqrt(np.cos(F_theta[j])-np.cos(theta_0))
        # if xt.y[1][j]<-.01:
        #      F_omega[j]=-2*omega_0**2*np.sqrt(np.cos(F_theta[j])-np.cos(theta_0))
           
        # if -.01<xt.y[1][j]<.01:
        #        F_omega[j]=0
        F_omega[j]=xt.y[1][j]
    plt.plot(tt,F_theta)
    plt.plot(tt,F_omega)
    plt.show()
    print(f"{9*i/128}π")
    G=np.column_stack([F_theta[0:np.size(tt):int(np.size(tt)/n_samples)],F_omega[0:np.size(tt):int(np.size(tt)/n_samples)]])
    Gmatrix.append(G)
Gmatrix=np.squeeze(np.stack(Gmatrix))
breakpoint()
delta=Gmatrix[:,-1]-Gmatrix[:,0]
def k(x,y,gamma=1,p=20):
     return np.exp(-gamma*np.linalg.norm(x-y)**2)
     #return gamma*(np.exp(-2*(np.sin(np.abs(np.pi*(x-y)/20))**2)))
d=2
#M=np.zeros([n_trajectories*d,n_trajectories*d])
Marray=[]
Mprime=np.identity(d)
for i in range(n_trajectories):
    for j in range(n_trajectories):
        k1 = k(Gmatrix[i,0],Gmatrix[j,0])
        k2 = k(Gmatrix[i,0],Gmatrix[j,-1])
        k3 = k(Gmatrix[i,-1],Gmatrix[j,0])
        k4 = k(Gmatrix[i,-1],Gmatrix[j,-1])
        #M[i,j]=(np.size(t)**2/4)*(k1+k2+k3+k3)
        Mprime[0,0]=(np.size(t)**2/4)*(k1+k2+k3+k3)
        Mprime[d,d]=(np.size(t)**2/4)*(k1+k2+k3+k3)
    Marray.append(Mprime)
M=block_diag(Marray)
l=.0001
Mmatrix=M+l*n_trajectories*np.identity(d*n_trajectories)
alpha=np.linalg.solve(Mmatrix,delta)
y_0=[theta_00*i for i in range(n_trajectories)]
def fstar(t,y,Gmatrix,alpha,tspan,n_examples):
    def k(x,y,gamma=1):
         return np.exp(-gamma*np.linalg.norm(x-y)**2)
         #return gamma*(np.exp(-2*(np.sin(np.abs(np.pi*(x-y)/20))**2)))
        
    sol=0
    for i in range(n_examples):
        sol=sol+(tspan/2)*(k(y,Gmatrix[i,0])+k(y,Gmatrix[i,-1])*alpha[i])
    return sol


#for i in range(n_trajectories):
y0=np.array([y_0[0]])
yt=solve_ivp(fstar,t_span=(0,np.size(t)),t_eval=[i for i in range(100)],y0=y0,args=(Gmatrix,alpha,np.size(t),n_trajectories),dense_output=True,vectorized=False)
plt.plot(yt.t,np.squeeze(yt.y))
plt.show()
    

        


