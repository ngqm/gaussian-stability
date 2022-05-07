"""
Figure 22.2
"""

import numpy as np
from numpy.random import uniform, normal
from scipy.linalg import lu
import matplotlib.pyplot as plt
from setup import DISTRIBUTION

# Probability density of growth factor
plt.figure()
for m in [8, 16, 32]:
  rho_list = []
  for _ in range(int(1e5)):
    # Set up A
    if DISTRIBUTION == 'uniform':
      A = uniform(-1, 1, (m, m))
    if DISTRIBUTION == 'normal':
      A = normal(0, m**(-1/2), (m, m))
    # LU factorisation
    _, _, U = lu(A)
    # growth factor
    rho = np.max(np.abs(U))/np.max(np.abs(A))
    rho_list.append(rho)
  plt.hist(rho_list, bins=100, density=True, label="m = {}".format(m),
    alpha=0.5, histtype='step')
  rho_list = []
plt.xlabel("Growth factor $\\rho$")
plt.ylabel("Probability density")
plt.legend()
plt.title("Figure 22.2 for {} distribution".format(DISTRIBUTION))
plt.show()