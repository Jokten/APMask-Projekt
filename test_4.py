from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy import stats
import matplotlib.pyplot as plt
import csv

class Model:
    def __init__(self) -> None:
        self.players = {}

    def add_team(self, match):
      for player in [match[0], match[1]]:
            if player not in self.players.keys():
                self.players[player] = [25, 8]
    
    def update(self,match):
        L = 10000
        self.gibbs(L, match)

    def gibbs(self, L, match):
      p1 = self.players[match[0]]
      p2 = self.players[match[1]]

      vs = np.array([[p1[1]**2, 0], [0, p2[1]**2]])
      ms = np.array([[p1[0]],[p2[0]]]) 
      vts = 12 ** 2
      A = np.array([[1,-1]])
      s = np.array([[1],[1]]) # Startvärden s?

      test = A @ vs @ A.T

      s_original = s



      def gen_t():
        if match[2] > 0:
          t = stats.truncnorm.rvs(0, np.inf, loc=(A @ s_original), scale= np.sqrt(vts + test))
        else:
          t = stats.truncnorm.rvs(-np.inf, 0, loc=(A @ s_original), scale= np.sqrt(vts + test))
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
    


# 0,3 std ca 25% 

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
# print(ds)
predictions = []
m = Model()

for match in ds[:1]:
  m.add_team(match)

  # Predictions here are made by sign(E(t)) and if draw team 1 wins
  if m.players[match[0]][0] >= m.players[match[1]][0]:
    predictions.append(1)
  else:
    predictions.append(-1)
  m.update(match)


print()
for team in {k: v for k, v in sorted(m.players.items(), key=lambda item: item[1], reverse=True)}:
    print(team, m.players[team])

correct_predictions = []
for i in range(len(predictions)):
  if predictions[i] == ds[i][2]:
    correct_predictions.append(True)
  else:
    correct_predictions.append(False)

print('Accuracy: ', sum(correct_predictions)/len(correct_predictions))

# After 20 matches Accuracy: 0.55
# After 100 matches Accuracy:  0.57
# After all matches Accuracy:  0.5845588235294118
# Alla matcher med vts = 25, Accuracy:  0.5882352941176471
# Efter att ha tagit bort första 100 matcherna, Accuracy:  0.5988372093023255

'''
Med deras hyperparametrar fast nya gibbs
Juventus [30.819483771445913, 2.46401037866343]
Napoli [29.85549397693062, 2.5885399095719444]
Roma [29.114111372139657, 3.166752216998702]
Milan [28.936098859909713, 3.3511016071661306]
Torino [28.86118989481488, 3.17190521970397]
Inter [28.839231346568337, 2.3105760550771612]
Atalanta [28.614708954707293, 2.406364292418552]
Lazio [26.05746680547335, 2.227197237360983]
Bologna [25.379474676888382, 2.855490043938896]
Sampdoria [24.481042563427955, 2.5122430955953186]
Empoli [24.108699462851966, 3.071915072557221]
Udinese [23.627817057493143, 2.2769073068194907]
Sassuolo [23.102572867571308, 2.7584150202415416]
Spal [23.080668826438473, 2.5607572028704313]
Cagliari [22.829145143884258, 2.63552976130284]
Genoa [22.585200038396756, 2.704823069121432]
Fiorentina [22.578912203940774, 2.8749901165787244]
Parma [22.563500823951834, 2.9187118116862885]
Frosinone [20.069337154102175, 2.8085502728573535]
Chievo [18.796330250423434, 3.117430004030342]
Accuracy:  0.6580882352941176
'''

'''
Med deras hyperparametrar, fast utan var**2
Napoli [26.32212775679728, 0.7571613640810625]
Juventus [26.30318239715772, 0.9056498977349695]
Atalanta [26.2596355814739, 0.8575402889167864]
Milan [26.158495578500837, 0.8084750449780631]
Torino [26.13009344447151, 0.8333367737671208]
Roma [26.036676145798328, 0.8282506362284714]
Inter [25.615942712823465, 0.8345798631506479]
Lazio [25.10961433565698, 0.8913127796745772]
Bologna [25.01399534432485, 0.7988000118259175]
Udinese [24.848560092857525, 0.8337930038263134]
Empoli [24.708763427515006, 0.9050707149863129]
Sampdoria [24.628087609004993, 0.9492542200430587]
Spal [24.55547991741253, 0.8537896971782036]
Cagliari [24.32241832696075, 0.9338835651991134]
Sassuolo [24.06594294797666, 0.9042369431368567]
Genoa [23.924118608005884, 0.907588736539716]
Parma [23.692953151843376, 0.8284850181434937]
Frosinone [23.666595450613, 0.85146581454101]
Fiorentina [23.304374719629035, 1.0964077799235759]
Chievo [23.193648667057015, 0.8659384097725248]'''

''' 
First game with our own hyperparameters, 100, 10 and 15**2
Juventus [102.84899414051927, 8.982312822367252]
Chievo [97.19964127144569, 9.033976161878694]
'''

'''
Q9 med hyperparametrarna 100, 10**2 (samma som 10), 15**2
S1 mu [[103.87030861]] S1 sigma [[9.22066761]]
S2 mu [[96.12969139]] S2 sigma [[9.22066761]]
'''

'''
Med gamla gibbs, hyperparametararnar 100, 10, 15**2
Juventus [104.92801157614966, 9.68254034039855]
Chievo [94.11306051305782, 9.35596890818816]
'''