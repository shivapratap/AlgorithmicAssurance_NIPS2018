#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 16:31:49 2018

@author: shivap
"""


import numpy as np
import matplotlib.pyplot as plt

from MAB.GP_EXP3 import GP_EXP3

from MAB.RoundRobin_MAB import RoundRobin_MAB
from MAB.RandomArm_MAB import RandomArm_MAB
from MAB.Oracle_BO import Oracle_BO

import testFunctions.syntheticFunctions

#%%

f = testFunctions.syntheticFunctions.func

bounds = [
    {'name': 'x1', 'type': 'continuous', 'domain': (-2,2)},
    {'name': 'x2', 'type': 'continuous', 'domain': (-2,2)}
]
categories = 3


#%%

trials = 5      # no of times to repeat the experiment
budget = 60     # budget for bayesian optimisation
seed   = 42     # seed for random number generator

#%% Run EXP3 Algorithm

myexp3 = GP_EXP3(objfn=f, initN=3, bounds=bounds, acq_type='LCB', C=categories, rand_seed=seed)
myexp3.runoptimBatchList(trials, budget)

#plot the results of exp3
#myexp3.plotResults()

#%% Baseline Round-Robin BO

rr = RoundRobin_MAB(objfn=f, initN=3, bounds=bounds, acq_type='LCB', C=categories, rand_seed=seed)
rr.runoptimBatchList(trials, budget)
#plot the results of round robin
#rr.plotResults()


#%% Baseline Random arm BO

random = RandomArm_MAB(objfn=f, initN=3, bounds=bounds, acq_type='LCB', C=categories, rand_seed=seed)
random.runoptimBatchList(trials, budget)

#%% Oracle

oracle = Oracle_BO(objfn=f, initN=3, bounds=bounds, acq_type='LCB', C=categories, rand_seed=seed)
oracle.runOracle_Trials(budget=50, trials=3)


#%% Plot the comparison

plt.style.use('seaborn-white')
    
   
indx = np.arange(10, 50, 5)
indx[-1] = indx[-1] -1

plt.figure(figsize=(9,5))
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

plt.errorbar(indx, myexp3.mean_bestVals_batch[indx], 
                         myexp3.mean_errVals_batch[indx], 
                         fmt='s-', ms=8, linewidth=1,
                         label="EXP3 UCB")


plt.errorbar(indx, rr.mean_bestVals_batch[indx], 
                         rr.mean_errVals_batch[indx], 
                         fmt='o--', ms=8, linewidth=1, 
                         label="RoundRobin BO")

plt.errorbar(indx,random.mean_bestVals_batch[indx], 
                         random.mean_errVals_batch[indx], 
                         fmt='v--', ms=9, linewidth=1,
                         label="Random BO")

plt.errorbar(indx, oracle.mean_best_vals[indx], 
                         oracle.err_best_vals[indx], 
                         fmt='p--', ms=9, linewidth=1,
                         label="Oracle")

plt.xlabel("Iteration", fontsize=20)
plt.ylabel("Best Value so far", fontsize=20)
plt.title("Synthetic Function - D=2 C=3", fontsize=25 )
plt.legend(prop={'size': 18})
#plt.savefig("SynthFns_C3_D2.pdf", bbox_inches='tight')
