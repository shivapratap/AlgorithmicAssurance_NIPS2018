# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:14:08 2018

@author: shivap
"""

import numpy as np
from MAB.MultiPlayMAB import MultiPlayMAB
from utils.probability import distr, draw
import pandas as pd
import GPy
from tqdm import tqdm
import math


class GP_EXP3(MultiPlayMAB):  
        
    def __init__(self, objfn, initN, bounds, acq_type, C, rand_seed=108):        
        super(GP_EXP3, self).__init__(objfn, initN, bounds, acq_type, C, rand_seed=rand_seed)
        self.policy = "Single play - EXP3"  
        self.best_val_list = []
    

    def runOptim(self, budget, b, initData=None, initResult=None):
        
        if (initData and initResult):
            self.data   = initData[:]
            self.result = initResult[:]
        else:
            self.data, self.result = self.initialise()     
        
        # Initialize wts and probs
        print("Running ", self.policy, " with budget ", budget)
         # -- From Jeremy Kun site
        bestUpperBoundEstimate = 2 * budget / 3
        gamma       = math.sqrt(self.C * math.log(self.C) / ((math.e - 1) * bestUpperBoundEstimate))
        Wc          = np.ones(self.C)
        Gt          = np.zeros(self.C) # Reward Function
        nDim        = len(self.bounds)
        result_list = []       
        my_kernel = GPy.kern.RBF(input_dim = nDim, lengthscale=0.1, ARD=False)  
        
        for t in tqdm(range(budget)):
            # Compute the probability for each category
#            probabilityDistribution = distr(Wc, gamma)            
            probabilityDistribution = distr(Wc, gamma)            
            # Choose a categorical variable at random
            ht = draw(probabilityDistribution)
#            ht = 0 # for testing
            
            # Update the reward
            Gt[ht] = self.getRewardperCategory(self.f, ht, my_kernel, self.bounds, self.acq_type, b)           
            estimatedReward = 1.0 * Gt[ht] / probabilityDistribution[ht]
            
            # Update the weight
            Wc[ht] *= math.exp(estimatedReward * gamma / self.C)
            
            # Get the best value till now
#            besty = self.getBestVal(self.result)
            besty, li, vi = self.getBestVal2(self.result)
            
            
              
            #Store the results of this iteration
            result_list.append([t, ht, Gt[ht], besty])
            self.ht_recommedations.append(ht)
            
        print("Finished ", self.policy, " ... ")
        df = pd.DataFrame(result_list, columns=["iter", "ht", "Reward", "best_value"])
        bestx = self.data[li][vi]  
        self.best_val_list.append([self.batch_num, self.trial_num, li, besty, bestx])
        return df