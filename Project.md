---
marp:true
math:mathjax
theme:"default"
pageinate:true

---
Mercer's Theorem and Applications
Statistical Learning, Fall 2024

---
Mercer Kernels

Properties:

-K is positive definite

-K is continuous

-$\mathcal{X}$ is a compact metric space

---

Defintion of $T_K$ operator.


$$(T_K\psi)(x)= \int_\mathcal{X}  K(x,t) \psi(t)  d\nu(t)$$


Properties of $T_K$ when K is Mercer


-$T_K:L_2(\mathcal{X})\rightarrow C^1(\mathcal{X})\subset L_2(\mathcal{X})$

-$T_K$ is a bounded linear operator

-$T_K$ is a compact operator

-$T_K$ is self-adjoint

---
Theorem 1 (Spectral Theorem)

 For every compact self-adjoint operator $T_k$ on a real or complex  
Hilbert space H, there exists an orthonormal basis of H consisting  
of eigenvectors of $T_K$. More specifically, the orthogonal complement  
of the kernel of $T_K$ admits either a finite orthonormal basis of  
eigenfunctions of $T_K$, or a countably infinite orthonormal basis  
${\psi_k}$ of eigenfunctions of $T_K$, with corresponding eigenvalues  
${\lambda_k} \in  \mathbb{R}$, such that $\lambda_k \rightarrow 0$.

 ---
Mercer's Theorem

Theorem 2 (Mercer). Let $\mathcal{X}$ be a compact metric space and  
$\nu$ be a non-degenerate Borel measure on $\mathcal{X}$, and K a  
continuous p.d. kernel. Then all eigenfunctions of $T_k$ are continuous  
and for all eigenvalues ${\lambda_i}>0$ and any $x,t \in \mathcal{X}$

$$K(x,t)=\sum\limits_{k=0}^{\infty}\lambda_k \psi_k(x) \psi_k(t)$$

where the series converges absolutely for each $(x,t) \in \mathcal{X} \times \mathcal{X}$  
 and uniformly on each compact subset of $\mathcal{X}$

---

Sketch of Proof.
Let $\mathcal{H}$ be the RKHS of K.

1. Show thay or any $k \geq 0$ such that $\lambda_k >0, \psi_k \in \mathcal{H}$

2. Show that $\{\sqrt{\lambda_k} \psi_k: \lambda_k>0\}$ forms an orthonormal basis of $\mathcal{H}$

3. Show that for any $x\in\mathcal{X}$ and some constant $C_K$
$$
\sum\limits_{k:\lambda>0} \lambda_k \psi_k(x)^2 \leq C_K
$$
4. Show that for any $x\in\mathcal{X}$ the mapping $t \rightarrow \sum_k \lambda_k \psi_k(x) \psi_k(t)\\$  
converges uniformly to a continuous function $g_x$.

5. Show that $K(x,\cdot)=K_x=g_x\in L^2$

---

Implications

Let 
$$
\begin{aligned}
&\Psi(x)=(\sqrt{\lambda_0}\psi_0(x),\sqrt{\lambda_1} \psi_{1,1}(x),\sqrt{\lambda_1}\psi_{1,2}(x)
,...,\\
&\sqrt{\lambda_2} \psi_{1,N_1}(x),\sqrt{\lambda_1} \psi_{2,1}(x),\sqrt{\lambda_1}\psi_{2,2}(x),...,\sqrt{\lambda_2} \psi_{2,N_2}(x),....)
\end{aligned}
$$

Then $\Psi(x)$ defines a feature map and

$$K(x,y)= \langle \Psi(x),\Psi(t)\rangle_{l^2}$$

---
Example 1

Consider the Kernel
$$K(x,t)=\exp(-\frac{\Vert x-t \Vert}{\sigma^2})$$ 
with $\mathcal{X}=S^1$ and uniform measure, $\nu$ on $\mathcal{X}$. To find the eigenfunctions of $T_K$  we guess that for $k\geq1$
$$
\begin{aligned}
&\psi_0(x)=\frac{1}{\sqrt{2\pi}}\\
&\psi_{2k}(x)=\frac{1}{\sqrt{\pi}}\cos(kx)\\
&\psi_{2k-1}(x)=\frac{1}{\sqrt{\pi}}\sin(kx)
\end{aligned}
$$
---
Since $K$ is symmetric $K(x-t)=K(-(x-t))\implies K$ is an even function.
Also K is periodic with period $2\pi$ on $S^1$. If we define $\kappa(x-t)=K(x,t)$, then $\kappa$ can be expanded as a Fourier Series:

$$\kappa(x)=\sum\limits_{k=0}^{\infty} \hat{\kappa}_{2k} \psi_{2k}(x),\hspace{1cm} x\in[0,2\pi]$$
$\implies$

$$(T_K \psi_{2k})(x)=\sum\limits_{l=1}^{\infty} \hat{\kappa}_{2l} \int_{0}^{2\pi}\psi_{2l}(x-t)\psi_{2k}(t)d\nu(t)=\\$$
$$\sum\limits_{l=1}^{\infty}\hat{\kappa}_{2l}\int_{0}^{2\pi} (\psi_{2l}(x)\psi_{2l}(t)+\psi_ {2l-1}(x)\psi_{2l-1}(t))\psi_{2k}(t)d\nu(t)=\\$$
$$\sum\limits_{l=0}^{\infty} \hat{\kappa}_{2l} \psi_{2l}(x) \delta_{kl}=\hat{\kappa}_{k}\psi_{2k}(x)$$

---
In the figures below the surface depicts $|\psi_{k}(x)|$ and different colors
represent whether $\psi_{k}(x)$ is positive or negative.

![center width:20cm height:auto](../Project_plots/Project_plot5.png)

---

![width:20cm height:auto](../Project_plots/Project_plot1.png)

---

![width:20cm height:auto](../Project_plots/Project_plot2.png)

---
Feature Maps

The eigenvalues for $K(x,y)=(1+x^{\top}t)^5$ are
$$\lambda_k=\dfrac{2 \pi2^5 5!}{(5-k)!}\dfrac{\Gamma (5+\frac{1}{2})}{\sqrt{\pi}\Gamma(6+k)}$$
Using the feature map construction for the previous kernels on $S^1,\\$
 $\Psi(x)$ has the form

$\Psi(x)=(\sqrt{\lambda_0},\sqrt{\lambda_1} \psi_{1,1}(x),\sqrt{\lambda_1} \psi_{1,2}(x),...,\sqrt{\lambda_5}\psi_{5,1}(x),\sqrt{\lambda_5 }\psi_{5,2})^{\top}=\\$

$(\sqrt{\frac{42}{2\pi}},\sqrt{\frac{30}{\pi}}\sin(x),\sqrt{\frac{30}{\pi}}\cos(x),\sqrt{\frac{15}{\pi}}\sin(2x),\sqrt{\frac{15}{\pi}}\cos(2x),\\$
$\sqrt{\frac{5}{\pi}}\sin(3x),\sqrt{\frac{5}{\pi}}\cos(3x),\sqrt{\frac{1}{\pi}}\sin(4x),\sqrt{\frac{1}{\pi}}\cos(4x),\\$
$\sqrt{\frac{1}{11\pi}}\sin(5x),\sqrt{\frac{1}{11\pi}}\cos(5x))^{\top}$

---

Likewise for $K(x,t)=\exp(-\parallel x-t \parallel)$, the eigenvalues are given as  
  
$$\lambda_k=e^{-2} I_k(2)\implies$$

$\Psi(x)=(\sqrt{\frac{0.31}{\pi}}, \sqrt{\frac{0.22}{\pi}}\sin(x),\sqrt{\frac{0.22}{\pi}}\cos(x),$
$\sqrt{\frac{0.09}{\pi}}\sin(2x),\sqrt{\frac{0.09}{\pi}} \cos(2x),...)^{\top}$



---

There is a general solution for the eigenfunctions on a hypersphere,  
namely the Spherical Harmonics.

$\mathcal{X}=S^{m-1} \implies \psi_{kl}(x)=H_{kl}.\\$ 

The exact forms of $H_{kl}$ can be found in reference[3]

In general, the eigenfunctions do not depend on the Kernel, but the  
eignevalues do. Luckily, analytical solutions for the eigenvalues  
have been found and are presented below. See reference[2].

---

$K=\exp(-\frac{\Vert x-y \Vert}{\sigma^2}):$

$\lambda_k=e^{-2/\sigma^2} \sigma^{m-2} I_{k+m/2-1}(2/\sigma^2)\Gamma(\frac{m}{2})$

where $I_k$ is the modified Bessel function of the first kind of order  
k and $\Gamma$ is the Gamma function. And  

$K=(1+x^{\top}y)^d:\\$
$\lambda_k=2^{d+m-2} \dfrac{d!}{(d-k)!} \dfrac{\Gamma(d+\frac{n-1}{2})\Gamma(\frac{m}{2})}{\sqrt{\pi} \Gamma(d+k+m-1)}$

---

For both cases the $\lambda_k$ occur with multiplicity of  

$N(m,k)=\dfrac{(2k+m-2)(k+m-3)!}{k!(m-2)!}$



---

![w:20cm h:auto](../Project_plots/Project_plot4.png)

Spherical harmonics. (2024, November 22). In Wikipedia. https://en.wikipedia.org/wiki/Spherical_harmonics

---

![w:20cm h:auto](../Project_plots/Project_plot3.png)

---
Other analytical Solutions

Analytical solutions for $K=(1+x^{\top}t)^d$ are given in [2] for the following $\mathcal{X}:\\$

$\mathcal{X} =\{-1,1\}^m$ are the vertices of the surface of a hypercube.

$\mathcal{X}=B^m$ which is a ball centered at the origin in m dimensions
for small d.



---
Choosing Appropriate Kernels

In the cases that the eigenvalues are given by the Fourier transform,  
Mercer's theorem allows us to choose a Kernel with a specified decay rate.

---

For example. Let $\beta\in\mathbb{N}^+$ and $x,t\in[0,1]\\$

$$
\begin{cases}
&\hat{\kappa}_0=0\\
&\hat{\kappa}_{2n}=\sqrt{2}k^{-2\beta,}, k\geq1
\end{cases}
\implies\\
$$
$$K(x,t)=\dfrac{1}{(2\beta)!} B_{2\beta}(x-t-\lfloor x-t\rfloor)$$
were $B_{2\beta}$ are even indexed Bernoulli functions and $\lfloor \cdot \rfloor$ is the floor function.

Kernels with exponential decay rates are given in reference[1]


---
The RKHS

$$\mathcal{H} =
\begin{cases}
f=\sum\limits_{k=0}^{\infty} a_k \psi_k \hspace{1.2cm} \text{s.t.}\sum\limits_{k=0}^{\infty} \dfrac{a_k^2}{\lambda_k}<\infty\\
\end{cases}\\
$$

$\text{And for } f,g\in \mathcal{H}, \hspace{.3cm}\langle f,g \rangle=\sum\limits_{k=0}^{\infty} \dfrac{a_k a_k^{\prime}}{\lambda_k}\\$

In the RKHS, the functions $\phi_k=\sqrt{\lambda_k}\psi_k$ are orthonormal and hence the basis elements, $\psi_k \in L^2$ are orthonogal with respect to $\langle \cdot,\cdot \rangle_{\mathcal{H}}$, and  

$$
\\
\begin{cases}
\begin{aligned}
&\langle \psi_k,\psi_l \rangle_{\mathcal{H}}=0 &\text{ for } k\neq l\\
&\langle  \psi_k,\psi_k \rangle_{\mathcal{H}}=\Vert \psi_k \Vert_{\mathcal{H}}=\dfrac{1}{\sqrt{\lambda_k}} &\text{ for } k=l
\end{aligned}
\end{cases}
$$

---

![w:15cm h:auto](../Project_plots/Project_plot6.png)

---
Sketch of proof

1. Show that $\mathcal{H}$ defined above is a Hilbert Space.
2. Check the elements of $\mathcal{H}$ are (continuous) functions.
3. Show that $K(x,\cdot)=K_x =\sum_{k=0}^{\infty} \lambda_k \psi_k(x)\psi_k\in \mathcal{H}\\$
4. Show that the reproducing property holds.

   I.e. $\langle f,K_x \rangle=f(x)$

---
Characterizing the functions in the RKHS for specific kernels.

Consider the example from before.

$K(x,t)=\dfrac{1}{(2\beta)!} B_{2\beta}(x-t-\lfloor x-t\rfloor)$ for $x,t\in[0,1]\\$

---

This Kernel can be expressed like in the earlier example as a Fourier series.  
In which applying the decay rate $\lambda_k=k^{-2\beta}$ gives
$$\Vert f \Vert^2_\mathcal{H}=\sum\limits_0^\infty (\hat{f}^2_{2n-1}+\hat{f}^2_{2n})k^{2\beta}<+\infty$$

Interestingly, this characterises functions $f\in$ the Sobolev space.  
If $f^{(k)}$ denotes the kth derivative, then $f^{(k)}$ is absolutely continuous and $f^{(k)}(0)=f^{(k)}(1)$ for $k=0,1,...\beta-1$ and

$$\Vert f \Vert^2_\mathcal{H}=\pi^{2\beta}\int\limits_0^1 \left(f^{(\beta)}(x)\right)^2 dx$$
This gives a way of penalizing a particular derivative of f in the cost  
functional by choosing $\beta$ appropriately.



---

References

[1] Youtube: https://www.youtube.com/watch?v=2eEMExbpVLU, Posted Jan. 6, 2021.

[2] Quang, Minh Ha, and Yuan Yao. “Mercer’s Theorem, Feature Maps, and Smoothing.” Proceedings of Learning Theory, 19th Annual Conference on Learning Theory, COLT 2006.

[3] Wikipedia: https://en.wikipedia.org/wiki/Spherical_harmonics

