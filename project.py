import numpy as np
from numpy.linalg import inv
from scipy import stats
vs = np.array([[1, 0],[0,4]])
ms = np.array([[1],[-1]])
vts = 5
A = np.array([[1,-1]])

# Quiz 1
vst = inv(inv(vs) + 1/vts*A.T @ A)
mst = vst @ (inv(vs) @ ms + 1/vts*A.T*3)

# Quiz 3
vt = vts + A @ vs @ A.T
mt = A @ ms
prob = 1 - float(stats.norm.cdf(0,mt,vt**0.5))
