#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 19:27:06 2018

@author: shivap
"""

# =============================================================================
#  EXP3 Algorithm for MNIST digits
#  Each algorithm takes around 7 hours on a server with 
# =============================================================================

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

trials = 5      # no: of times to repeat our experiment
budget = 500    # budget for bayesian optimisation
categories = 9  # no of digits - 9 digits in total
seed = 42       # seed for random number generator

#%%

# initN - no of initial points to be generated for BO
# bounds - bounds of input for BO
# acq_type - List of valid acquisition functions for GPyopt


start_time = time.time()
myexp3 = GP_EXP3(objfn=f, initN=15, bounds=bounds, acq_type='LCB', C=categories, rand_seed=seed)
myexp3.runoptimBatchList(trials, budget)
exp3_time = time.time() - start_time
print("EXP3 finished in : ", exp3_time/3600, "hrs")

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
oracle = Oracle_BO(objfn=f, initN=1, bounds=bounds, acq_type='LCB', C=categories)
oracle.runOracle_Trials(500, 3)
oracle_time = time.time() - start_time
print("Oracle finished in : ", oracle_time/3600, "hrs")

#%% Plot the comparison
    
indx = np.arange(0, 505, 40)
indx[-1] = indx[-1] -1

plt.style.use('seaborn-white') 
plt.figure(figsize=(9,5))
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

plt.errorbar(indx, myexp3.mean_bestVals_batch[indx]*100,
             myexp3.mean_errVals_batch[indx]*100,
                         fmt='s-', ms=8,linewidth=1,
                         label="EXP3BO ")

plt.errorbar(indx, round_robin.mean_bestVals_batch[indx]*100,
             round_robin.mean_errVals_batch[indx]*100,
             fmt='o--', ms=5, linewidth=1,
             label="RoundRobin BO")

plt.errorbar(indx, random.mean_bestVals_batch[indx]*100,
             random.mean_errVals_batch[indx]*100,
             fmt='v--', ms=5, linewidth=1,
             label="Random BO")

plt.errorbar(indx, oracle.mean_bestVals_batch[indx]*100,
             oracle.mean_errVals_batch[indx]*100,
             fmt='p--', ms=5, linewidth=1,
             label="Oracle")

plt.legend()

#%% Plot the histogram of selected digits for EXP3
#myexp3.plotResults()