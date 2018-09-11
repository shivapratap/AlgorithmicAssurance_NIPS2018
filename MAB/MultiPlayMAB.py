# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:10:48 2018

@author: shivap
"""

import numpy as np
from tqdm import tqdm
import GPyOpt
import pandas as pd
import collections


class MultiPlayMAB:
    
    def __init__(self, objfn, initN, bounds, acq_type, C, rand_seed=108):
        self.f           = objfn        # function to optimise
        self.bounds      = bounds   # function bounds
        self.C           = C        # no of categories
        self.acq_type    = acq_type
        self.initN       = initN    # no: of initial points        
        self.nDim        = len(self.bounds) # dimension
        self.rand_seed   = rand_seed
        
        # Store the ht recommendations for each iteration
        self.ht_recommedations  = []
        self.ht_hist_batch      = []
        
        # Store the name of the algorithm
        self.policy = None
    
        # Set the random number
        np.random.seed(self.rand_seed)        
        self.X = []
        self.Y = []
        
        # To check the best vals
        self.gp_bestvals = []


    def initialise(self):        
        data       = []
        result     = []
        for i in range(self.C):
            Xinit = self.generateInitialPoints(self.initN, self.bounds)
            yinit = self.f(i, Xinit).reshape(-1,1)
            data.append(Xinit)
            result.append(yinit)
        return data, result
    

    def generateInitialPoints(self, initN, bounds):
        nDim = len(bounds)
        Xinit = np.zeros((initN, len(bounds)))
        for i in range(initN):
            Xinit[i, :] = np.array([np.random.uniform(bounds[b]['domain'][0], bounds[b]['domain'][1], 1)[0] for b in range(nDim)])
        return Xinit
    
    
    def runoptimBatchList(self, trials, budget, batch_list=[1]):
        self.batch_list = batch_list
        self.budget = budget
        numBatches = len(batch_list)
        self.mean_bestVals_batch = np.zeros((budget, numBatches))
        self.mean_errVals_batch  = np.zeros((budget, numBatches))        
        
        for index in range(numBatches):
            b = batch_list[index]
            self.batch_num = b
#            print("####### Processing Batch: ", b , " ###########")            
            self.mean_bestVals_batch[:,index], self.mean_errVals_batch[:,index], hist_batch = self.runTrials(trials, budget, b)
            self.ht_hist_batch.append(hist_batch)
            
            
    def runTrials(self, trials, budget, b):
        # Initialize mean_bestvals, stderr, hist
        best_vals = np.zeros((budget, trials))
        
        for i in range(trials):
            print("Running trial: " ,i)
            self.trial_num = i
            done = False
            while not done:
                try:
                    df = self.runOptim(budget, b)
                    best_vals[:,i] = df['best_value']
                    done = True
                except Exception as exception:
                    print("Got exception: ", exception.__class__.__name__)
                    print(self.ht_recommedations)
                    
        # Runoptim updates the ht_recommendation histogram        
        ht_hist = collections.Counter(np.array(self.ht_recommedations).ravel())
        self.ht_recommedations = []
        self.mean_best_vals = np.mean(best_vals, axis=1)
        self.err_best_vals  = np.std(best_vals, axis=1)/np.sqrt(trials)
        
        # For debugging
        self.gp_bestvals = best_vals
        
        return self.mean_best_vals, self.err_best_vals, ht_hist

             
# =============================================================================
#     Over-ride this!
# =============================================================================
    def runOptim(self, budget, b):
        print("Over-ride me!")
    
# =============================================================================
#   Function returns the best value so far from a given category ht  
# =============================================================================
    def getRewardperCategory(self, objfn, ht, kernel, bounds, acq, b):
#        print("Running BO for phase: ", ht)
        Xt = self.data[ht]
        yt = self.result[ht]
        
        myBopt = GPyOpt.methods.BayesianOptimization(f=None, domain=bounds,
                                             kernel     = kernel,
                                             model_type = 'GP',
                                             X = Xt, Y =  yt,
                                             acquisition_type = acq,  
                                             evaluator_type = 'local_penalization',
                                             batch_size = b)
        
#        print("Func: ", ht)
        x_next = myBopt.suggest_next_locations()
        y_next = objfn(ht, x_next)
        
        # Append recommeded data 
        self.data[ht]   = np.row_stack((self.data[ht],   x_next))
        self.result[ht] = np.row_stack((self.result[ht], y_next))
        
        #Update the best value
        bestval_ht = np.max(self.result[ht] * -1)
#        bestval_ht = np.max(1  - self.result[ht])
#        bestval_ht = np.min(self.result[ht])
        return bestval_ht 

    def getBestVal(self, my_list):    
        temp = [np.max(i * -1) for i in my_list]
#        temp = [np.min(i) for i in my_list]
#        return np.min(temp)
        return np.max(temp)
    

# =============================================================================
# Get best value from nested list along with the index    
# =============================================================================
    def getBestVal2(self, my_list):    
        temp = [np.max(i * -1) for i in my_list]
        indx1 = [np.argmax(i * -1) for i in my_list]
        indx2 = np.argmax(temp)
        val = np.max(temp)
        list_indx = indx2
        val_indx = indx1[indx2]
        return val, list_indx, val_indx
    
    
    
# =============================================================================
#     Get the best vals of y and x
#        a = input data
#        b = result
# =============================================================================
    def get_xopt_yopt(a, b):
        indx_per_list = [np.argmax(i * -1) for i in b]
        temp_b = [b[index][indx_per_list[index]] for index in range(len(b))]
        temp_a = [a[index][indx_per_list[index]] for index in range(len(b))]
        x_opt  = temp_a[np.argmin(temp_b)]
        y_opt  = np.min(temp_b)
        index  = np.argmin(temp_b)
        return x_opt, y_opt, index
    
    def get_ht_hist(self, df):
        n = len(np.unique(df.ht.value_counts()))
        vals = np.zeros(n)
        for i in range(n):
            vals[i] = df.ht.value_counts()[i]
        return vals
    
    
    def plotResults(self):
        import matplotlib.pyplot as plt
        plt.style.use('fivethirtyeight')        
        for i in range(len(self.batch_list)):
            plt.errorbar(np.arange(self.budget), 
                         self.mean_bestVals_batch[:,i], 
                         self.mean_errVals_batch[:,i], 
                         linewidth=1, 
                         label="Batch size = " + str(self.batch_list[i]))
            plt.xlabel("Iterations")
            plt.ylabel("Best Value so far")
            plt.title(self.policy)
        plt.show()
        
#        Plot grouped bar plots for histograms of batch size
        self.df = pd.DataFrame(self.ht_hist_batch).transpose()
        colnames = []
        for i in self.batch_list:
            colnames.append("batch " + str(i))
        
        self.df.columns = colnames
        self.df.plot(kind='bar', alpha=0.7, title=self.policy, legend=False)
        
        
#        for i in range(len(self.ht_hist_batch)):
#            plt.bar(self.ht_hist_batch[i].keys(),self.ht_hist_batch[i].values())                 
#            plt.show()
  


  