# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:00:18 2018

@author: shivap
"""

import numpy as np
from MAB.MultiPlayMAB import MultiPlayMAB
import pandas as pd
from tqdm import tqdm

import GPy


class GPHedge(MultiPlayMAB):    
    
    def __init__(self, objfn, initN, bounds, acq_type, C, rand_seed=108):        
        super(GPHedge, self).__init__(objfn, initN, bounds, acq_type, C, rand_seed=rand_seed)
        self.policy = "GP_Hedge"  
        self.best_val_list = []
  
# =============================================================================
#     Over-riding for hedge
# =============================================================================
    def runOptim(self, budget, b, initData=None, initResult=None):
        
        if (initData and initResult):
            self.data   = initData[:]
            self.result = initResult[:]
        else:
            self.data, self.result = self.initialise()            
        
        # Initialize wts and probs
        print("Running with budget ", budget)
        eta         = np.sqrt(np.log(self.C)/budget)
        Wc          = np.ones(self.C)
        Gt          = np.zeros(self.C) # Reward Function
        nDim        = len(self.bounds)
        result_list     = []       
        my_kernel = GPy.kern.RBF(input_dim = nDim, lengthscale=0.1, ARD=False)  
        
        for t in tqdm(range(budget)):
            # Compute the probability for each category
            Ptc = Wc/np.sum(Wc)
            
            # Choose a categorical variable at random
            ht_array = np.random.multinomial(1, Ptc)
            ht = np.array(np.where(ht_array == 1))[0][0] # Convert tuple to int   
#            ht = 0 # for testing
            
            # Update the reward
            Gt[ht] = self.getRewardperCategory(self.f, ht, my_kernel, self.bounds, self.acq_type, 1)           
            
            # Get the best value till now
#            besty = self.getBestVal(self.result)
            besty, li, vi = self.getBestVal2(self.result)

#            
            # Update the weight
            Wc[ht] = Wc[ht] * (1 + eta)**Gt[ht]            
            #Store the results of this iteration
            result_list.append([t, ht, Gt[ht], besty])
            self.ht_recommedations.append(ht)
            
        print("Finished ", self.policy, " ... ")
        df = pd.DataFrame(result_list, columns=["iter", "ht", "Reward", "best_value"])
        bestx = self.data[li][vi]        
        self.best_val_list.append([self.batch_num, self.trial_num, li, besty, bestx])
        return df


