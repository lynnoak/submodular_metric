# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:44:28 2017

@author: victor
"""

import matplotlib.pyplot as plt


balance =[0.8834,0.7457,0.7119,0.7104]							
t_balance = [3.7123,3.4525,3.4016,3.3018]							
glass = [0.6528,0.6333,0.6147,0.6331,0.6013,0.6099,0.5087,0.614]		
t_glass =[1.2915,1.2885,1.2924,1.2928,1.1215,1.1538,1.2857,1.1914] 			
iono10 = [0.8604,0.8632,0.866,0.8604,0.6927,0.8718,0.7211,0.8632,0.8775,0.8632]	
t_iono10 =[6.1803,6.053,6.2745,6.1379,5.2793,5.4945,4.8213,5.3003,4.9421,5.2296]

sonar12 = [0.5814,0.6009,0.6009,0.6201,0.6108,0.6199,0.5572,0.6294,0.6153,0.6349,0.6245,0.6245]
iono12 = [0.8604,0.8575,0.8546,0.849,0.8147,0.8604,0.7581,0.8546,0.8689,0.8575,0.8575,0.8718]
t_sonar12 = [122.33,128.3,125.3,171.35,137.73,130.42,92.372,80.884,71.931,83.994,68.624,68.312]
t_iono12 = [126.94,130.49,138.28,175.04,130.06,116.18,94.997,81.811,72.123,70.702,70.451,71.062]


plt.figure(1)

plt.subplot(211)
plt.ylabel("Score of cross validation")
plt.xlim(0,13)
plt.xticks(range(11),range(1,12))
plt.plot(range(len(balance)),balance,label = 'balance')
plt.plot(range(len(glass)),glass,label = 'glass')
plt.plot(range(len(iono10)),iono10,label = 'iono10')
plt.legend()

plt.subplot(212)
plt.xlabel("K of k-add")
plt.ylabel("Cost of time")
plt.xlim(0,13)
plt.xticks(range(11),range(1,12))
plt.plot(range(len(t_balance)),t_balance,label = 'balance')
plt.plot(range(len(t_glass)),t_glass,label = 'glass')
plt.plot(range(len(t_iono10)),t_iono10,label = 'iono10')
plt.legend()

plt.savefig("kadd1.png")
plt.show()

plt.figure(2)

plt.subplot(211)
plt.ylabel("Score of cross validation")
plt.xlim(0,15)
plt.xticks(range(12),range(1,13))
plt.plot(range(len(sonar12)),sonar12,label = 'sonar')
plt.plot(range(len(iono12)),iono12,label = 'iono')
plt.legend()

plt.subplot(212)
plt.ylabel("Cost of time")
plt.xlabel("K of k-add")
plt.xlim(0,15)
plt.xticks(range(12),range(1,13))
plt.plot(range(len(t_sonar12)),t_sonar12,label = 'sonar')
plt.plot(range(len(t_iono12)),t_iono12,label = 'iono')
plt.legend()

plt.savefig("kadd2.png")
plt.show()
