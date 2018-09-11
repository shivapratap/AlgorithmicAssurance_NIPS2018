# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 12:09:52 2018

@author: shivap
"""

import numpy as np
from tqdm import tqdm
import GPy
import pandas as pd

from MAB.MultiPlayMAB import MultiPlayMAB

class RandomArm_MAB(MultiPlayMAB):   
    
    def __init__(self, objfn, initN, bounds, acq_type, C, rand_seed=108):        
        super(RandomArm_MAB, self).__init__(objfn, initN, bounds, acq_type, C, rand_seed=rand_seed)
        self.policy = "Random MAB"
    

    def runOptim(self, budget, b):
#        self.data   = self.init_data[:]
#        self.result = self.init_result[:]
        self.data, self.result = self.initialise()
        # Initialize wts and probs
        print("Running ", self.policy, " with budget ", budget)        
        result_list     = []       
        my_kernel = GPy.kern.RBF(input_dim = self.nDim, lengthscale=0.1, ARD=False)  
        
        ht_list = np.array([np.arange(self.C)] * budget)
        ht_array = ht_list.ravel()
        
        for t in tqdm(range(budget)):     
            ht_array = np.random.randint(0, self.C, b)
            for ht in ht_array:
                self.getRewardperCategory(self.f, ht, my_kernel, self.bounds, self.acq_type, b)
                self.ht_recommedations.append(ht)
            # Get the best value till now
            besty = self.getBestVal(self.result)
            result_list.append([t, ht, besty])
            
        print("Finished ", self.policy, " ... ")
        df = pd.DataFrame(result_list, columns=["iter", "ht", "best_value"])
        return df
