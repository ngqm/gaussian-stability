"""
Figures 22.3 and 22.4
"""

import numpy as np
from numpy.random import uniform, normal, choice
from scipy.linalg import lu, qr, inv
import matplotlib.pyplot as plt
from setup import DISTRIBUTION

N = 128  # matrix size

# Set up A
if DISTRIBUTION == 'uniform':
  A = uniform(-1, 1, (N, N))
  std = 1/np.sqrt(3)
if DISTRIBUTION == 'normal':
  A = normal(0, N**(-1/2), (N, N))
  std = 1/np.sqrt(N)

# Visualise inverse of L
_, L, _ = lu(A)  # LU factorisation
L_inv = inv(L)  # inverse of L
L_inv_masked = (np.abs(L_inv)>=1).astype(int)  # Filter out small elements
Lt = np.array([choice([x, -x]) for x in L.flatten()]).\
  reshape(L.shape)  # randomised L
Lt_inv = inv(Lt)  # inverse of randomised L
Lt_inv_masked = (np.abs(Lt_inv)>=1).astype(int)  # Filter out small elements

plt.figure()
plt.suptitle("Figure 22.3 for {} distribution".format(DISTRIBUTION))
plt.subplot(121)
plt.imshow(L_inv_masked, cmap='Blues')
plt.title('Inverse of L (max = {:5.2f})'.format(np.max(L_inv)))
plt.subplot(122)
plt.imshow(Lt_inv_masked, cmap='Blues')
plt.title('Inverse of randomised L (max = {:5.2f})'.format(np.max(Lt_inv)))
plt.show()

# Q portraits
Q, _ = qr(L)  # QR factorisation
Q_masked = (np.abs(Q)>=std).astype(int)  # Filter out small elements
Qt = qr(Lt)[0]  # QR factorisation of randomised L
Qt_masked = (np.abs(Qt)>=std).astype(int)  # Filter out small elements

# with masking
plt.figure()
plt.suptitle("Figure 22.4 for {} distribution".format(DISTRIBUTION))
plt.subplot(121)
plt.imshow(Q_masked, cmap='Blues')
plt.title('Q portrait (max = {:5.2f})'.format(np.max(Q)))
plt.subplot(122)
plt.imshow(Qt_masked, cmap='Blues')
plt.title('Q portrait of randomised L (max = {:5.2f})'.format(np.max(Qt)))
plt.show()

# without masking
plt.figure()
plt.suptitle("Figure 22.4* for {} distribution".format(DISTRIBUTION))
plt.subplot(121)
plt.imshow(Q, cmap='Blues')
plt.title('Q (max = {:5.2f})'.format(np.max(Q)))
plt.subplot(122)
plt.imshow(Qt, cmap='Blues')
plt.title('Q of randomised L (max = {:5.2f})'.format(np.max(Qt)))
plt.show()