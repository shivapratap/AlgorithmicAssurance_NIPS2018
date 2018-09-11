#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 16:27:03 2018

@author: shivap
"""

import numpy as np
        
# =============================================================================
# Rosenbrock Function
# https://www.sfu.ca/~ssurjano/rosen.html
# =============================================================================
def myrosenbrock(X):
    X=np.asarray(X)
    X = X.reshape((-1,2))    
    if len(X.shape)==1:# one observation
        x1=X[0]
        x2=X[1]
    else:# multiple observations
        x1=X[:,0]
        x2=X[:,1]    
    fx = 100*(x2-x1**2)**2 + (x1-1)**2    
    return fx.reshape(-1,1)

# =============================================================================
#  Six-hump Camel Function
#  https://www.sfu.ca/~ssurjano/camel6.html       
# =============================================================================
def mysixhumpcamp(X):
    X=np.asarray(X)
    X = np.reshape(X,(-1,2))
    if len(X.shape)==1:
        x1=X[0]
        x2=X[1]
    else:
        x1=X[:,0]
        x2=X[:,1]
    term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
    term2 = x1*x2
    term3 = (-4+4*x2**2) * x2**2
    fval = term1 + term2 + term3    
    return fval.reshape(-1,1)

# =============================================================================
# Beale function
# https://www.sfu.ca/~ssurjano/beale.html
# =============================================================================
def mybeale(X):
    X=np.asarray(X)
    X = X.reshape((-1,2))
    if len(X.shape)==1:
        x1=X[0]
        x2=X[1]
    else:
        x1=X[:,0]
        x2=X[:,1]	
    fval = (1.5-x1+x1*x2)**2+(2.25-x1+x1*x2**2)**2+(2.625-x1+x1*x2**3)**2    
    return fval.reshape(-1,1)


def func(ht,X):
    # ht is a categorical index
    # X is a continuous variable
    
    if ht==0: #rosenbrock
        return myrosenbrock(X)
    elif ht==1: # six hump
        return mysixhumpcamp(X)
    elif ht==2: # beale
        return mybeale(X)

    