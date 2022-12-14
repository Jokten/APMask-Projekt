from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import csv

class Model:
    def __init__(self) -> None:
        self.players = {}

    def add_team(self, match):
      for player in [match[0], match[1]]:
            if player not in self.players.keys():
                self.players[player] = [100, 15]
    
    def update(self,match):
        L = 300             
        self.gibbs(L, match)

    def gibbs(self, L, match): # det finns inget L i denna GIbs
      p1 = self.players[match[0]]
      p2 = self.players[match[1]]

      vs = np.array([[p1[1], 0],[0, p2[1]]])
      ms = np.array([[p1[0]],[p2[0]]]) 
      vts = 10
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
    
    def odds(self, match):
      s1 = self.players[match[0]][0]
      s2 = self.players[match[1]][0]
      mean = s1-s2 # s1-s2
      var = 25 # vts
      t_pdf = stats.norm(loc=mean, scale=var)
      print()
      print(match[1], ' ', round(s1,2),  ' vs ', match[0], ' ', round(s2,2))
      print('Odds', match[1] , ': ', round(1/ (1-t_pdf.cdf(0)),2), ', probability: ', round(1- t_pdf.cdf(0), 2))
      print('Odds', match[0] , ': ', round(1/t_pdf.cdf(0), 2), ', probability: ', round(t_pdf.cdf(0), 2))


def data_reader(dat):
    """ Processes CSV file and produces list of list with player names who competed
     and who won. """
    ATP_2018 = pd.read_csv(dat) # read CSV to produce panda dataframe into to count number of times player has played
    matches = []
    # read CSV file line by line
    with open(dat, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for i in reader:
            # Both the players have to have played at least 8 matches on tour for it to count
            if (ATP_2018["winner_name"] == i[10]).sum() + (ATP_2018["loser_name"] == i[10]).sum() >= 8:
                if (ATP_2018["winner_name"] == i[18]).sum() + (ATP_2018["loser_name"] == i[18]).sum() >= 8:
                    matches.append([i[10], i[18], 1]) # always a won, since i[10] is the winner column and i[18] is the loser column
    return matches



# ================== # 
ds = data_reader('atp_matches_2018.csv') # downloading data
predictions = []    # create predictions list
m = Model()   


for match in ds:
  m.add_team(match)
  m.odds(match)
  if m.players[match[0]][0] >= m.players[match[1]][0]:
    predictions.append(1)
  else:
    predictions.append(-1)
  m.update(match)
 
# print in order of decreasing player skill
for team in {k: v for k, v in sorted(m.players.items(), key=lambda item: item[1], reverse=True)}:
    print(team, m.players[team])

correct_predictions = []
# Determine Number of predictions that were correct by comparing with data 
for i in range(100, len(predictions)): 
  if predictions[i] == ds[i][2]:
    correct_predictions.append(True)
  else:
    correct_predictions.append(False)

# Print Accuracy
print('Accuracy: ', sum(correct_predictions)/len(correct_predictions))

# After 20 matches Accuracy: 
# After 100 matches Accuracy:  
# After all matches Accuracy: 
# Alla matcher med vts = 25, Accuracy:  
# Efter att ha tagit bort första 100 matcherna, Accuracy:  