"""
Figure 22.1
"""

import numpy as np
from numpy.random import uniform, normal
from scipy.linalg import lu
import matplotlib.pyplot as plt
from setup import DISTRIBUTION


# Growth factor against matrix size
m_rho = []
for m in np.logspace(0.5, 3, num=50):
  for _ in range(8):
    # Set up A
    if DISTRIBUTION == 'uniform':
      A = uniform(-1, 1, (int(m), int(m)))
    if DISTRIBUTION == 'normal':
      A = normal(0, m**(-1/2), (int(m), int(m)))
    # LU factorisation
    _, _, U = lu(A)
    # growth factor
    rho = np.max(np.abs(U))/np.max(np.abs(A))
    m_rho.append([int(m), rho])

plt.figure()
plt.suptitle("Figure 22.1 for {} distribution".format(DISTRIBUTION))
plt.scatter([x[0] for x in m_rho], [x[1] for x in m_rho], s=4)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Matrix size m')
plt.ylabel('Growth factor $\\rho$')
# reference line
plt.plot(np.logspace(0.5, 3, num=50), 
  np.sqrt(np.logspace(0.5, 3, num=50)), '--k', label="$\sqrt{m}$")
plt.legend()  
plt.show()