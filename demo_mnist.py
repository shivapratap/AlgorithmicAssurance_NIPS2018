#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 19:27:06 2018

@author: shivap
"""

import numpy as np
import matplotlib.pyplot as plt

from MAB.GP_EXP3 import GP_EXP3
from MAB.RoundRobin_MAB import RoundRobin_MAB
from MAB.RandomArm_MAB import RandomArm_MAB
from MAB.Oracle_BO import Oracle_BO
from MAB.Oracle_MAB import Oracle_MAB

import mnist_experiment.run_mnist_model
import time

#%%

f = mnist_experiment.run_mnist_model.getDigitScore

bounds = [
    {'name': 'x_shear', 'type': 'continuous', 'domain': (-.2,.2)},
    {'name': 'y_shear', 'type': 'continuous', 'domain': (-.2,.2)},
    {'name': 'angle',   'type': 'continuous', 'domain': (0,360)}
]

trials = 5
budget = 500
categories = 9
seed = 108

#%%
start_time = time.time()
myexp3 = GP_EXP3(objfn=f, initN=15, bounds=bounds, acq_type='LCB', C=categories, rand_seed=seed)
myexp3.runoptimBatchList(trials, budget)
exp3_time = time.time() - start_time

#%%
    
start_time = time.time()
round_robin = RoundRobin_MAB(objfn=f, initN=15, bounds=bounds, acq_type='LCB', C=categories, rand_seed=seed)
round_robin.runoptimBatchList(trials, budget)
rr_time = time.time() - start_time


#%%

start_time = time.time()
random = RandomArm_MAB(objfn=f, initN=15, bounds=bounds, acq_type='LCB', C=categories, rand_seed=seed)
random.runoptimBatchList(trials, budget)
rand_time = time.time() - start_time
    
    
#%%
start_time = time.time()
oracle = Oracle_BO(objfn=f, initN=15, bounds=bounds, acq_type='LCB', C=categories)
oracle.runOracle_Trials(100, 5)
oracle_time = time.time() - start_time

#%%
    
       