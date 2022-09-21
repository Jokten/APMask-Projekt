from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy import stats
import matplotlib.pyplot as plt
vs = np.array([[10, 0],[0,10]])
ms = np.array([[100],[100]])
vts = 5
A = np.array([[1,-1]])

class Model:
    def __init__(self, y) -> None:
        self.vs = np.array([[10, 0],[0,10]])
        self.ms = np.array([[100],[100]])
        self.vts = 5
        self.A = np.array([[1,-1]])
        self.s = np.array([[100],[150]])
        self.t = 0
        self.y = y

    def gen_t(self):
        self.t = stats.truncnorm.rvs(0, 10000, loc=(self.A @ self.s), scale= self.vts, )
        return self.t
    def gen_s(self):
        vst = inv(inv(self.vs) + 1/self.vts*self.A.T @ self.A)
        mst = vst @ (inv(self.vs) @ ms + 1/self.vts*self.A.T*self.t)
        self.s = stats.multivariate_normal.rvs(mean=mst.reshape(2), cov=vst)
        return self.s

L = 300

t = np.zeros(L)
s = np.zeros((2,L))
model = Model(1)
for i in range(L):
    t[i] = model.gen_t()
    s[:,i] = model.gen_s()
plt.plot(range(L),s[1,:])
plt.plot(range(L),s[0,:])
#plt.hist(t,density=True, bins=50)
plt.show()

# Quiz 1
# vst = inv(inv(vs) + 1/vts*A.T @ A)
# mst = vst @ (inv(vs) @ ms + 1/vts*A.T*3)

# # Quiz 3
# vt = vts + A @ vs @ A.T
# mt = A @ ms
# prob = 1 - float(stats.norm.cdf(0,mt,vt**0.5))
