import numpy as np
import matplotlib.pyplot as plt
import math

x = np.linspace(0, 1, 500)

exact = np.exp(-x)
poly6 = sum((-1)**n / math.factorial(n) * x**n for n in range(7))

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(x, np.abs(exact - poly6), 'g-', linewidth=2, label='degree-6 error')
ax.axhline(y=0.001, color='k', linestyle=':', linewidth=1.5, label='Q6.10 LSB = 0.001')
ax.set_xlabel('x')
ax.set_ylabel('|error|')
ax.set_title('Absolute error: exp(-x) vs degree-6 Taylor polynomial on [0, 1]')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/user/Desktop/horner_errorplot.png', dpi=150)
plt.show()
print("Saved to Desktop/horner_errorplot.png")
