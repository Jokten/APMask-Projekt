from tkinter import Label
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy import stats
import matplotlib.pyplot as plt
import csv
import pandas as pd
from time import perf_counter
# Hyperparameters
NUMBER_OF_MATCHES = -1 # -1 for all matches
VTS = 15**2 # Variance for t, before sqrt
DRAW_REGION = 0 # Hyperparameter for size of region to draw, 0.06 should give 2

class Model:
    def __init__(self) -> None:
        self.players = {}

    def add_team(self, match):
      for player in [match[0], match[1]]:
            if player not in self.players.keys():
                self.players[player] = [100, 10]
    
    def update(self,match, L=1030):
        return self.gibbs(L, match)

    def gibbs(self, L, match):
      p1 = self.players[match[0]]
      p2 = self.players[match[1]]

      vs = np.array([[p1[1]**2, 0],[0, p2[1]**2]])
      ms = np.array([[p1[0]],[p2[0]]]) 
      vts = VTS
      A = np.array([[1,-1]])
      s = np.array([p1[0],p2[0]]) # Startvärden s
      s = np.array([[1],[1]])
      draw_eps = np.sqrt(vts) * DRAW_REGION

      def gen_t():
        if match[2] > 0:
          t = stats.truncnorm.rvs(draw_eps, 10000, loc=(A @ s), scale= np.sqrt(vts))
        elif match[2] < 0:
          t = stats.truncnorm.rvs(-10000, -draw_eps, loc=(A @ s), scale= np.sqrt(vts))
        elif draw_eps != 0:
          t = stats.truncnorm.rvs(-1*draw_eps, draw_eps, loc=(A @ s), scale= np.sqrt(vts))
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

      self.players[match[0]] = [np.mean(s1_samples[30:130]), np.std(s1_samples[30:130])]
      self.players[match[1]] = [np.mean(s2_samples[30:130]), np.std(s2_samples[30:130])]

      

      return t_samples, s1_samples, s2_samples
    



# 0,3 std ca 25%, var 50 ger draw region eps = 15

def data_reader(dat):
    matches = []
    with open(dat,'r',newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for i in reader:
                if i[4] != i[5]:
                  matches.append([i[2], i[3], 1 if int(i[4])-int(i[5]) > 0 else -1])
                else:
                #  matches.append([i[2], i[3], 0]) # Uncomment for including draws
                    pass
    return matches

def data_reader2(dat):
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



ds = data_reader('SerieA.csv')

# Q5
m1 = Model()
first_match = ds[0]
m1.add_team(first_match)
t_set, set1, set2 = m1.update(first_match,2030)

plt.figure(0)
plt.plot(set1, label='Player 1')
plt.plot(set2, label='Player 2')
plt.xlabel('Sample nr #')
plt.ylabel('Score')
#plt.show()

xx = np.linspace(50,150,200)
plt.figure(1)
plt.hist(set1[30:130], density=True, label='Player 1 Histogram')
plt.hist(set2[30:130], density=True, label='Player 2 Histogram')
plt.plot(xx,stats.norm.pdf(xx,loc=np.mean(set1[30:130]), scale=np.std(set1[30:130])), color='blue', label='Player 1 Gaussian')
plt.plot(xx,stats.norm.pdf(xx,loc=np.mean(set2[30:130]), scale=np.std(set2[30:130])), color='orange', label='Player 2 Gaussian')
plt.xlabel('Score')
plt.ylabel('Number of samples')
plt.title('100 samples')
plt.legend()

plt.figure(2)
plt.hist(set1[30:530], density=True, label='Player 1 Histogram')
plt.hist(set2[30:530], density=True, label='Player 2 Histogram')
plt.plot(xx,stats.norm.pdf(xx,loc=np.mean(set1[30:530]), scale=np.std(set1[30:530])), color='blue', label='Player 1 Gaussian')
plt.plot(xx,stats.norm.pdf(xx,loc=np.mean(set2[30:530]), scale=np.std(set2[30:530])), color='orange', label='Player 2 Gaussian')
plt.xlabel('Score')
plt.ylabel('Number of samples')
plt.title('500 samples')
plt.legend()

plt.figure(3)
plt.hist(set1[30:1030], density=True, label='Player 1 Histogram')
plt.hist(set2[30:1030], density=True, label='Player 2 Histogram')
plt.plot(xx,stats.norm.pdf(xx,loc=np.mean(set1[30:1030]), scale=np.std(set1[30:1030])), color='blue', label='Player 1 Gaussian')
plt.plot(xx,stats.norm.pdf(xx,loc=np.mean(set2[30:1030]), scale=np.std(set2[30:1030])), color='orange', label='Player 2 Gaussian')
plt.xlabel('Score')
plt.ylabel('Number of samples')
plt.title('1000 samples')
plt.legend()

plt.figure(4)
plt.hist(set1[30:2030], density=True, label='Player 1 Histogram')
plt.hist(set2[30:2030], density=True, label='Player 2 Histogram')
plt.plot(xx,stats.norm.pdf(xx,loc=np.mean(set1[30:2030]), scale=np.std(set1[30:2030])), color='blue', label='Player 1 Gaussian')
plt.plot(xx,stats.norm.pdf(xx,loc=np.mean(set2[30:2030]), scale=np.std(set2[30:2030])), color='orange', label='Player 2 Gaussian')
plt.xlabel('Score')
plt.ylabel('Number of samples')
plt.title('2000 samples')


plt.legend()
plt.show()

print('Maximum number of matches: ', len(ds))
predictions = []
m = Model()


for match in ds[-2:NUMBER_OF_MATCHES]:
  m.add_team(match)
  vts = VTS
  draw_eps = np.sqrt(vts) * DRAW_REGION
  # Predictions here are made by sign(E(t)) and if draw team 1 wins
  if m.players[match[0]][0] - m.players[match[1]][0] > draw_eps:
    predictions.append(1)
  elif m.players[match[0]][0] - m.players[match[1]][0] < -1*draw_eps:
    predictions.append(-1)
  else:
    # print("Expected draw!")
    # print("Actual results: ", match[2])
    # if match[2] == 0:
    #   print("Correct prediction!")
    # else:
    #   print("Wrong prediction")
    predictions.append(-1) # set 0 if draws simulated
  m.update(match,1030)

f = open('A_Ranking.csv', 'w')
writer = csv.writer(f)
for team in {k: v for k, v in sorted(m.players.items(), key=lambda item: item[1], reverse=True)}:
    writer.writerow([team,m.players[team][0],m.players[team][1]])
    print(team, m.players[team])
f.close()

correct_predictions = []
for i in range(len(predictions)):
  if predictions[i] == ds[i][2]:
    correct_predictions.append(True)
  else:
    correct_predictions.append(False)

print('Accuracy: ', sum(correct_predictions)/len(correct_predictions))
# print("Number of predicted draws: ", predictions.count(0))
# print("Number of actual draws: ", [i[2] for i in ds[:NUMBER_OF_MATCHES]].count(0))




# ================== # 
ds = data_reader2('atp_matches_2018.csv') # downloading data

print('Maximum number of matches: ', len(ds))
predictions = []
m = Model()


for match in ds[:NUMBER_OF_MATCHES]:
  m.add_team(match)
  vts = VTS
  draw_eps = np.sqrt(vts) * DRAW_REGION
  # Predictions here are made by sign(E(t)) and if draw team 1 wins
  if m.players[match[0]][0] - m.players[match[1]][0] > draw_eps:
    predictions.append(1)
  elif m.players[match[0]][0] - m.players[match[1]][0] < -1*draw_eps:
    predictions.append(-1)
  else:
    # print("Expected draw!")
    # print("Actual results: ", match[2])
    # if match[2] == 0:
    #   print("Correct prediction!")
    # else:
    #   print("Wrong prediction")
    predictions.append(-1) # set 0 if draws simulated
  m.update(match,1030)

f = open('Tennis_Ranking.csv', 'w')
writer = csv.writer(f)
for team in {k: v for k, v in sorted(m.players.items(), key=lambda item: item[1], reverse=True)}:
    writer.writerow([team,m.players[team][0],m.players[team][1]])
    print(team, m.players[team])
f.close()

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

