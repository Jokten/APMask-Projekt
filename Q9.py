#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:21:01 2022

@author: Tennis
"""
import numpy as np
from scipy.stats import truncnorm
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy import stats
import math

def multiplyGauss(m1, s1, m2, s2):
# computes the Gaussian distribution N(m,s) being propotional to N(m1,s1)*N(m2,s2)
#
# Input:
# m1, s1: mean and variance of first Gaussian # m2, s2: mean and variance of second Gaussian
#
# Output:
# m, s: mean and variance of the product Gaussian
    s = 1/(1/s1+1/s2)
    m = (m1/s1+m2/s2)*s
    return m, s

def divideGauss(m1, s1, m2, s2):
# computes the Gaussian distribution N(m,s) being propotional to N(m1,s1)/N(m2,s2)
#
# Input:
# m1, s1: mean and variance of the numerator Gaussian
# m2, s2: mean and variance of the denominator Gaussian

#
# Output:
# m, s: mean and variance of the quotient Gaussian

    m, s = multiplyGauss(m1, s1, m2, -s2) 
    return m, s

def truncGaussMM(a, b, m0, s0):
# computes the mean and variance of a truncated Gaussian distribution
#
# Input:
# a, b: The interval [a, b] on which the Gaussian is being truncated # m0,s0: mean and variance of the Gaussian which is to be truncated
#
# Output:
# m, s: mean and variance of the truncated Gaussian
# scale interval with mean and variance


    a_scaled, b_scaled = (a - m0) / np.sqrt(s0), (b - m0) / np.sqrt(s0)
    m = truncnorm.mean(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
    s = truncnorm.var(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0)) 
    return m, s

def truncGaussMMwith2(awin, bwin, alose, blose, m0, s0):
# computes the mean and variance of a truncated Gaussian distribution
#
# Input:
# a, b: The interval [a, b] on which the Gaussian is being truncated # m0,s0: mean and variance of the Gaussian which is to be truncated
#
# Output:
# m, s: mean and variance of the truncated Gaussian
# scale interval with mean and variance
    a_scaled, b_scaled = (alose - m0) / np.sqrt(s0), (blose - m0) / np.sqrt(s0)
    m_lost = truncnorm.mean(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
    s_lost = truncnorm.var(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0)) 

    a_scaled, b_scaled = (awin - m0) / np.sqrt(s0), (bwin - m0) / np.sqrt(s0)
    m_won = truncnorm.mean(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
    s_won = truncnorm.var(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0)) 
    return m_lost, s_lost, m_won, s_won


# Hyperparameters
S1_m = 25 # mean of S1
S1_s = 8**2 # varianceof S1
S2_m = 25# mean of S2 
S2_s = 8**2# variance of S2
vts  = 12 ** 2  # the variance of p(s|t)
y0   = 1   # Who won
t = 0



A = np.array([[1,-1]])
vs = np.array([[S1_s, 0],[0,S2_s]]) # variance vs standard deviation
ms = np.array([[S1_m],[S2_m]])

# Message u4 from fs1 to S1
mu4_m, mu4_s = S1_m, S1_s
# Message u3 from fs2 to S2
mu3_m, mu3_s = S2_m, S2_s
# Message u5 from S2 to fst
mu5_m, mu5_s = mu3_m, mu3_s 
# Message u6 from S1 to fst
mu6_m, mu6_s = mu4_m, mu4_s
# Message u7 from fst to t 

# Using Corollary 2, we find the marginal of t
A = np.array([[1,-1]])
vs = np.array([[S1_s, 0],[0,S2_s]]) # variance vs standard deviation
ms = np.array([[S1_m],[S2_m]])

mu7_m = A @ms
mu7_s = vts + A@vs@A.T

# Message u8 from t to fst
# Do moment matching of the marginal of t. p(t|y)
if y0 == 1:
    a, b = 0, np.Inf
else:
    a, b = np.NINF, 0 
    
pt_m, pt_s = truncGaussMM(a, b, mu7_m, mu7_s)

# Compute the message from t to fst
# Outgoing message is the approximated marginal divided by the incoming message
mu8_m, mu8_s = divideGauss(pt_m, pt_s, mu7_m, mu7_s)



# We have vs: matrix with variances 
#         A : vector do get right 
#         vts: is the number given for p(s|t) (hyperparameter)
#         mu8: p(t)

 
# # Message u9 from fst to S1

mu9_m = mu6_m  + mu8_m
mu9_s = vts + mu4_s + mu8_s

# # # Message u10 from fst to S2
mu10_m = mu5_m - mu8_m
mu10_s = vts + mu3_s + mu8_s

print(f' ({mu9_m, mu9_s}), ({mu10_m, mu10_s})')

### FUNCTIONAL #### 
# # Compute marginal of S1
pS1_m, pS1_s = multiplyGauss(mu4_m, mu4_s, mu9_m, mu9_s)
# 
# # Compute marginal of S2
pS2_m, pS2_s = multiplyGauss(mu3_m, mu3_s, mu10_m, mu10_s)



# 
print(f'S1 mu {pS1_m} S1 sigma {np.sqrt(pS1_s)}') # Output: 0.564189583548
print(f'S2 mu {pS2_m} S2 sigma {np.sqrt(pS2_s)}') # Output: 0.681690113816

'''
Med deras hyperparametrar 25, 8, 12**2
S1 mu [[28.09624689]] S1 sigma [[7.37653409]]
S2 mu [[21.90375311]] S2 sigma [[7.37653409]]
'''

'''
Deras gibbs sampler från runme, med 25, 8, 12**2
27.811058022566122 7.421126945175493
21.88485343498833 7.400920721900221
'''

'''
Våran nya gibbs sampler, med 25, 8, 12**2
Juventus [27.387139997675312, 7.211939626143361]
Chievo [22.641805054890156, 7.166475971686341]
'''

'''
Med gamla gibbs sampler, med 25, 8, 12**2
Juventus [29.121763875158127, 7.659697738498515]
Chievo [20.477450622798717, 7.60261360530237]
'''

'''
Deras gibbs
28.33878830264553 7.337790884872726
21.952483000127025 7.196589183803288
Deras mp
28.096246890578605 7.376534090789678
21.903753109421395 7.376534090789678

Våran mp
S1 mu [[28.09624689]] S1 sigma [[7.37653409]]
S2 mu [[21.90375311]] S2 sigma [[7.37653409]]

våran nya gibbs
Juventus [27.225596547376526, 7.233065760578077]
Chievo [22.634476475711182, 7.138475972440488]

deras gibbs med fler iter 
28.248969671045003 7.432406195645922
22.12572419444859 7.317644922164622

Deras gibbs utan konstig skalning
29.348295684600572 7.460666815547616
20.696032695857358 7.752838160398238

våran nya gibbs med np.inf
Juventus [27.161791504372726, 7.153589942658762]
Chievo [22.83688489426708, 7.2309119393146855]

våran mp med hyperparametrar 100, 10 ,15
S1 mu [[103.87030861]] S1 sigma [[9.22066761]]
S2 mu [[96.12969139]] S2 sigma [[9.22066761]]

nya gibbs, hyperparametrar 100, 10, 15 och np.inf och + var_s
Juventus [103.89777236120682, 9.269978986616415]
Chievo [96.24524530071454, 9.095312552174603]

våran mp med deras hyperparam 25, 8, 12
S1 mu [[28.09624689]] S1 sigma [[7.37653409]]
S2 mu [[21.90375311]] S2 sigma [[7.37653409]]

nya gibbs med deras hyperparam 25, 8, 12, np.inf och + var_s
Juventus [28.085070717385545, 7.361005234733517]
Chievo [21.89687329232059, 7.398543566046761]
'''