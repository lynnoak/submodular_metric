# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:24:05 2017

@author: victor
"""

from constraints_tools import *
from metric_computation import *
import matplotlib.pyplot as plt


def ShowBar(SKNN,stdSKNN,SCho,stdSCho,SMul = None ,stdSMul = None,n1 = 'SKNN',n2 = 'SCho',n3 = 'SMul',title = 'test'):
    """
    Show and save the result 
    Input: "SKNN,stdSKNN,SCho,stdSCho,SMul,stdSMul" The mean and std of Scores 
        with original KNN,Choquet,MultiLinear
        "title" title of the result
    """
    nData = len(SKNN)
    fig, ax = plt.subplots()
    index = np.arange(nData)
    bar_width = 0.2
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, SKNN, bar_width,
                     alpha=opacity,
                     color='g',
                     error_kw=error_config,
                     yerr = stdSKNN, 
                     label=n1)  

    rects2 = plt.bar(index + bar_width, SCho, bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     yerr = stdSCho, 
                     label=n2)
    if(SMul):
        rects3 = plt.bar(index + 2*bar_width, SMul, bar_width,
             alpha=opacity,
             color='b',
             error_kw=error_config,
             yerr = stdSMul, 
             label=n3)                 

    plt.xlabel("Datasets")
    plt.ylabel("Score of cross validation")
    plt.title(title)
    plt.xticks(index + bar_width, ('balance','seeds','wine','iono','sonar'))
    plt.xlim(0,nData+2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(title+".png")
    plt.show()
