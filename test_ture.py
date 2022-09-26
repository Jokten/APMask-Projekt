from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy import stats
import matplotlib.pyplot as plt
import csv

class Model:
    def __init__(self) -> None:
        self.players = {}

    
    def update(self,match):
        for player in [match[0], match[1]]:
            if player not in self.players.keys():
                self.players[player] = [100, 15]
        
        L = 300
        self.gibbs(L, match)

    def gibbs(self, L, match):
      p1 = self.players[match[0]]
      p2 = self.players[match[1]]

      vs = np.array([[p1[1], 0],[0, p2[1]]])
      ms = np.array([[p1[0]],[p2[0]]]) 
      vts = 5
      A = np.array([[1,-1]])
      s = np.array([[1],[1]]) # Startvärden s?
      t = 0


      def gen_t():
        if match[2] > 0:
          t = stats.truncnorm.rvs(0, 10000, loc=(A @ s), scale= vts)
        else:
          t = stats.truncnorm.rvs(-10000, 0, loc=(A @ s), scale= vts)
        return t

      def gen_s():
        vst = inv(inv(vs) + 1/vts*A.T @ A)
        mst = vst @ (inv(vs) @ ms + 1/vts*A.T*t)
        s = stats.multivariate_normal.rvs(mean=mst.reshape(2), cov=vst)
        return s


      t_samples = np.zeros(L)
      s1_samples = np.zeros((L))
      s2_samples = np.zeros((L))
      for i in range(L):
          t = gen_t()
          t_samples[i] = t
          s = gen_s()
          s1_samples[i] = s[0]
          s2_samples[i] = s[1]

      self.players[match[0]] = [np.mean(s1_samples[30:]), np.std(s1_samples[30:])]
      self.players[match[1]] = [np.mean(s2_samples[30:]), np.std(s2_samples[30:])]

      return t_samples, s1_samples, s2_samples



  

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


ds = data_reader('SerieA.csv')
print(len(ds))

m = Model()
for match in ds:
    m.update(match)

for team in {k: v for k, v in sorted(m.players.items(), key=lambda item: item[1], reverse=True)}:
    print(team, m.players[team])

# Exempel efter 10 Matcher
'''
Juventus [115.78151414346745, 2.1729055082669793]
Napoli [115.31800120507921, 2.237384380520798]
Spal [115.08764612291101, 2.164630137907238]
Atalanta [113.418568643223, 3.7126936410385993]
Roma [112.99279974018427, 4.930348725694931]
Empoli [111.31371007804599, 4.416429714058609]
Sassuolo [111.10204029632692, 4.377865224544233]
Inter [89.61012050546096, 4.045781011136883]
Cagliari [88.30919373497831, 4.181165542907494]
Bologna [88.21233242780242, 4.249514857184297]
Parma [87.9604318177882, 4.269384798576261]
Chievo [87.88922901457094, 3.952498340749527]
Milan [87.87197541529078, 4.069685283549899]
Torino [87.05392692059078, 4.725311065430706]
Frosinone [86.42710368664164, 4.050094436615118]
Lazio [84.67741408027206, 2.206013665402054]
'''

# Exempel efter 100 Matcher
'''
Juventus [127.73154941222349, 1.0934636959071349]
Napoli [120.85478665319572, 1.1545003024651266]
Sassuolo [114.37280761335326, 1.1384837318199097]
Fiorentina [114.20963745767426, 1.0783337685647094]
Spal [110.28019096309053, 1.0636060875463478]
Roma [108.7654845758067, 1.1490497432379017]
Genoa [106.99507182085858, 1.1625136545097512]
Atalanta [106.56247681802611, 1.1671532699836087]
Udinese [105.6713244202438, 1.051221688167749]
Empoli [104.8135783292928, 1.0845533138039778]
Inter [96.31123594457912, 1.0900945617729452]
Milan [95.97363244551447, 1.0952502891427685]
Torino [92.41924692825336, 1.1098771337054203]
Sampdoria [90.70247136302916, 1.0106147417845617]
Lazio [89.41713485442823, 0.9915496137943405]
Cagliari [88.71113474028594, 1.0771233196573564]
Parma [87.05576277017904, 1.0970632985285078]
Bologna [79.56864746045913, 1.0915379351415588]
Frosinone [78.76519844291899, 1.0066118548412968]
Chievo [75.6720747342835, 1.1068392790049029]
'''

# Exempel efter alla matcher 
'''
Juventus [136.72849727015688, 1.120283393445484]
Napoli [128.58362562049308, 1.032878495409409]
Atalanta [117.12291942264864, 1.0997391424944543]
Roma [116.81653930174855, 1.0922278690720257]
Sassuolo [110.50915509255822, 1.0548247393285815]
Fiorentina [109.18739221201656, 1.0841045561693143]
Spal [107.88884978428474, 1.0464423095795246]
Udinese [104.7023764359229, 1.1158302784219247]
Milan [104.24966477000206, 1.0217771605075143]
Genoa [103.19364814404214, 1.1287366356963482]
Inter [100.53053439684554, 1.018815439111074]
Empoli [98.74313908698852, 1.0669320611601214]
Torino [97.64893298050666, 1.0916671566664375]
Sampdoria [91.22272283829548, 1.1719007796530216]
Lazio [90.24469285544461, 1.1684178817766266]
Cagliari [82.64590123481403, 1.123158278427206]
Parma [81.33337083531696, 1.0984238724290418]
Bologna [80.7473207909945, 1.08897912736119]
Frosinone [69.50129361466473, 1.159574076365439]
Chievo [67.57351645829287, 1.0505000801053037]
'''