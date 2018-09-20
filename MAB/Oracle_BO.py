# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 12:04:51 2018

@author: shivap
"""

import numpy as np
from tqdm import tqdm

import pandas as pd
from GPyOpt.methods import BayesianOptimization

class Oracle_BO():   
    
    def __init__(self, objfn, initN, bounds, acq_type, C, rand_seed=108):                
        self.policy  = "Oracle"    
        self.f           = objfn        # function to optimise
        self.bounds      = bounds   # function bounds
        self.C           = C        # no of categories
        self.acq_type    = acq_type
        self.initN       = initN    # no: of initial points        
        self.nDim        = len(self.bounds) # dimension
        self.rand_seed   = rand_seed
        self.opt_list    = []


    def generateInitialPoints(self, initN, bounds):
        nDim = len(bounds)
        Xinit = np.zeros((initN, len(bounds)))
        for i in range(initN):
            Xinit[i, :] = np.array([np.random.uniform(bounds[b]['domain'][0], bounds[b]['domain'][1], 1)[0] for b in range(nDim)])
        return Xinit

        
    def initialise(self):        
        data       = []
        result     = []
        for i in range(self.C):
            Xinit = self.generateInitialPoints(self.initN, self.bounds)
            yinit = self.f(i, Xinit).reshape(-1,1)
            data.append(Xinit)
            result.append(yinit)
        return data, result
    

    def my_func(self, X):
        return self.f(self.cat, X)    
    
    def runOracle_Trials(self, budget, trials):
        self.result_perTrial_list = []
        self.opt_ht_trial = []
#        self.best_vals = np.zeros((budget+5, trials))
        self.best_vals = []
#        print("Size of best vals: ", self.best_vals.shape)
        for t in range(trials):
#            df, self.best_vals[:, t] = self.runOracle(budget)
            #self.df holds the result for each category
            #best_vals is a list of best values during each trial
            self.df, bestvals = self.runOracle(budget)
            self.best_vals.append(bestvals)
            self.result_perTrial_list.append(self.df)
            self.opt_ht_trial.append(self.opt_ht)
        self.mean_best_vals = np.mean(self.best_vals, axis=0)
        self.err_best_vals  = np.std(self.best_vals, axis=0)/np.sqrt(trials)
#        
            
    def runOracle(self, budget):
        self.data, self.result = self.initialise()
        self.result_list = []
        # Initialize wts and probs
        print("Running ", self.policy, " with budget ", budget) 
        for i in tqdm(range(self.C)):
            self.cat = i
            print("Processing arm: ", self.cat)
            myBopt = BayesianOptimization(f=self.my_func, domain=self.bounds)
            myBopt.run_optimization(max_iter=budget)
            self.result_list.append(myBopt.Y_best)        
        
        self.opt_vals = [self.result_list[ii][-1] for ii in range(self.C)]
        self.opt_ht = np.argmin(self.opt_vals)
        
        maxlen = 0
        for i in range(self.C):
            if len(self.result_list[i]) > maxlen:
                maxlen = len(self.result_list[i])
        
        
        vals = np.zeros((maxlen, self.C ))
        for i in range(self.C):
            l = len(self.result_list[i])
            vals[:,i][:l] = self.result_list[i][:l]     
            
#        for i in range(self.C):
#            vals[:,i] = self.result_list[i]
     
        bv = self.result_list[self.opt_ht] * -1 
        df = pd.DataFrame(data = vals, columns=np.arange(self.C))
#        print("Size of best vals in loop: ", bv.shape)
        return df, bv
    
    
    
    def plotResults(self, stepsize=1):
        import matplotlib.pyplot as plt
        indx = np.arange(0, len(self.mean_best_vals), np.round(len(self.mean_best_vals)/stepsize))        
        
        indx[-1] = len(self.mean_best_vals) - 1 # Make sure last index matches the len
        indx = indx.astype(int)
        plt.errorbar(indx, self.mean_best_vals[indx], self.err_best_vals[indx], 
                         linewidth=1, label="Oracle")

        plt.xlabel("Iterations")
        plt.ylabel("Best Value so far")
        plt.legend()

        
        

        