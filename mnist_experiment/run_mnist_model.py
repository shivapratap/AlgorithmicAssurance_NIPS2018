#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:15:05 2018

@author: shivap
"""
import numpy as np
import mnist_experiment.mnist_helper_alternate as mha

#%%

# Global variables
model_filename = "models/lenet_Aug2"
model = mha.loadModel(model_filename)

pXtest = np.load("data/testdata_paper.npy")
y_test = np.load("data/testlabels_paper.npy")

data = [pXtest, y_test]
isRotate, isZoom, isSkew = 1,0,1
transform=[isRotate, isZoom, isSkew]

# =============================================================================
#  index = index of digit [0:8 - There is no 9]
#  xx = [x_shear, y_shear, angle]
#  x_shear and y_shear bounds between -0.2 to 0.2
#  angle bounded between 0 to 360
# =============================================================================

def getDigitScore(index, xx):
    xx = np.atleast_2d(xx)
    score = np.zeros(len(xx)) 
#    print("Processing digit:", index)
    import time
    start=time.time()
    for i in range(len(xx)):
#        print("Processing
        x_shear = xx[i,0]
        y_shear = xx[i,1]
        angle = xx[i,2]
        measure=[angle, x_shear, y_shear]
        digit_err = mha.test_ErrorDigittransformMNIST(model, data, transform, measure)
        score[i] = digit_err[index] * -1
    end=time.time()
    elapse=end-start
#    print("Total time: ", elapse)        
    return score


#%%
    
#import time
#start=time.time()

#xx = [-1.04502794e-01,  -1.46367979e-01,   1.70557838e+02]
#vu=getDigitScore(7, xx)
#print(vu)
    
#end=time.time()
#elapse=end-start
#print(elapse)
