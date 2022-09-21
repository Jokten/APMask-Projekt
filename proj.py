from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy import stats
import matplotlib.pyplot as plt
import csv
# vs = np.array([[10, 0],[0,10]])
# ms = np.array([[100],[100]])
# vts = 5
# A = np.array([[1,-1]])

# class Model:
#     def __init__(self, y) -> None:
#         self.vs = np.array([[1, 0],[0,1]])
#         self.ms = np.array([[100],[000]])
#         self.vts = 5
#         self.A = np.array([[1,-1]])
#         self.s = np.array([[100],[100]])
#         self.t = 0
#         self.y = y

#     def gen_t(self):
#         self.t = stats.truncnorm.rvs(0, 10000, loc=(self.A @ self.s), scale= np.sqrt(self.vts), )
#         return self.t
#     def gen_s(self):
#         vst = inv(inv(self.vs) + 1/self.vts*self.A.T @ self.A)
#         mst = vst @ (inv(self.vs) @ self.ms + 1/self.vts*self.A.T*self.t)
#         self.s = stats.multivariate_normal.rvs(mean=mst.reshape(2), cov=vst)
#         return self.s

class Model:
    def __init__(self) -> None:
        self.players = {}
        self.vt = 10
    
    def update(self,s1,s2,score):
        for i in [s1,s2]:
            if i not in self.players.keys():
                self.players[i] = [100, 15]
        
        L = 300
        sm = np.array([[self.players[s1][0]],[self.players[s2][0]]])
        sv = np.array([[self.players[s1][1]**2, 0],[0, self.players[s2][1]**2]])
        s = sm.copy()
        A = np.array([])
        t = stats.truncnorm.rvs(0, 10000, loc=(A @ self.s), scale= np.sqrt(self.vts), )
        

def norm_from_data(data):
    return stats.norm(loc=np.mean(data), scale=np.std(data))

def data_reader(dat):
    matches = []
    with open(dat,'r',newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for i in reader:
            if i[4] != i[5]:
                matches.append([i[2], i[3],1 if int(i[4])-int(i[5]) > 0 else -1])
    return matches

    


L = 1000
ds = data_reader('SerieA.csv')

stats.norm(loc=5, sclae=10)
t = np.zeros(L)
s = np.zeros((2,L))
#model = Model(1)
# for i in range(L):
#     t[i] = model.gen_t()
#     s[:,i] = model.gen_s()
#plt.plot(range(L),s[1,:])
#plt.plot(range(L),s[0,:])
#plt.hist(t,density=True, bins=50)
#plt.show()

print(norm_from_data(s[0,40:]).stats('mv'))
print(norm_from_data(s[1,40:]).stats('mv'))
print(norm_from_data(t[40:]).stats('mv'))

# Quiz 1
# vst = inv(inv(vs) + 1/vts*A.T @ A)
# mst = vst @ (inv(vs) @ ms + 1/vts*A.T*3)

# # Quiz 3
# vt = vts + A @ vs @ A.T
# mt = A @ ms
# prob = 1 - float(stats.norm.cdf(0,mt,vt**0.5))
