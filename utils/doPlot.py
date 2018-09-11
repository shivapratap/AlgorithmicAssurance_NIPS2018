# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:47:42 2018

@author: shivap
"""

import numpy as np
import matplotlib.pyplot as plt

def forretal08(x):    
    fact1 = (6*x - 2)**2
    fact2 = np.sin(12*x - 4)
    y     = fact1 * fact2
    return y

def forretal2(x, A=0.5, B=10, C=-5):    
    yh = forretal08(x)
    term1 = A * yh
    term2 = B * (x-0.5)
    y = term1 + term2 - C
    return y

def func3(x):        
    y = x * np.sin(12*x - 4) * 40 
    return y


def plotBOres(data, result):
    xx = np.random.uniform(0,1,1000)
    xx = np.sort(xx)
    y1 = forretal08(xx)
    y2 = forretal08(-xx)
    y3 = forretal2(xx)        
    y4 = func3(xx)
    
    pts0 = data[0]
    pts1 = data[1]
    pts2 = data[2]
    pts3 = data[3]
    
    res0 = result[0]
    res1 = result[1]
    res2 = result[2]
    res3 = result[3]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xx,y1, 'r', label="F1")
    ax.plot(xx,y2, 'g', label="F2")
    ax.plot(xx,y3, 'b', label="F3")
    ax.plot(xx,y4, 'c', label="F4")    
    ax.plot(pts0, res0, 'rs')
    ax.plot(pts1, res1, 'gs')
    ax.plot(pts2, res2, 'bs')
    ax.plot(pts3, res3, 'cs')
    ax.legend()
 
#    fig = plt.figure()
#    plt.plot(xx,y1, 'r', label="F1")
#    plt.plot(xx,y2, 'g', label="F2")
#    plt.plot(xx,y3, 'b', label="F3")
#    plt.plot(xx,y4, 'c', label="F4")    
#    plt.plot(pts0, res0, 'rs')
#    plt.plot(pts1, res1, 'gs')
#    plt.plot(pts2, res2, 'bs')
#    plt.plot(pts3, res3, 'cs')
#    plt.legend()
    
    return fig, ax

#

#%%
    
if __name__ == "__main__":
    plt.style.use('fivethirtyeight') 
    fig, ax = plotBOres(myhal.data, myhal.result)
#    ax.plot(0.6, 35, 'r*')