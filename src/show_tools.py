# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:24:05 2017

@author: victor
"""

from constraints_tools import *
from metric_computation import *
import matplotlib.pyplot as plt


def ShowBar(OrgKNN,stdOrgKNN,Choq,stdChoq,S_LMNN = None ,stdS_LMNN = None,n1 = 'OrgKNN',n2 = 'Choq',n3 = 'S_LMNN',title = 'test'):
    """
    Show and save the result 
    Input: "OrgKNN,stdOrgKNN,Choq,stdChoq,S_LMNN,stdS_LMNN" The mean and std of Scores 
        with original KNN,Choquet,MultiLinear
        "title" title of the result
    """
    nData = len(OrgKNN)
    fig, ax = plt.subplots()
    index = np.arange(nData)
    bar_width = 0.2
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, OrgKNN, bar_width,
                     alpha=opacity,
                     color='g',
                     error_kw=error_config,
                     yerr = stdOrgKNN, 
                     label=n1)  

    rects2 = plt.bar(index + bar_width, Choq, bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     yerr = stdChoq, 
                     label=n2)
    if(S_LMNN):
        rects3 = plt.bar(index + 2*bar_width, S_LMNN, bar_width,
             alpha=opacity,
             color='b',
             error_kw=error_config,
             yerr = stdS_LMNN, 
             label=n3)                 

    plt.xlabel("Datasets")
    plt.ylabel("Score of cross validation")
    plt.title(title)
    plt.xticks(index + bar_width, ('glass','balance','iono','sonar','digits'))
    plt.xlim(0,nData+2)
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(title+".png")
    plt.show()
