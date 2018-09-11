# Algorithmic Assurance

Python implementation of Multitask Assurance for the paper:
 "Algorithmic Assurance: An Active Approach to Algorithmic Testing using Bayesian Optimisation".

This code is supplement to NIPS 2018 submission

Installation
============

### Dependencies
* Numpy
* Scipy
* Scikit-learn
* keras  (for lenet-5 model MNIST)
* Gpyopt (for Bayesian optimisation)
* tqdm   (to get runtime of loops)


### Usage:
* demo_syntheticFns.py runs the experiment for Synthetic functions (see testFunctions/syntheticFunctions.py for function definition)
* demo_mnist.py runs the experiment for MNIST dataset ()

* Initialise exp3 as: myexp3 = GP_EXP3(objfn=f, initN=15, bounds=bounds, acq_type='LCB', C=categories, rand_seed=seed, where f is the function to optimise, initN = number of initial points for BO, bounds = bounds for input in BO, acq_type = Acquisition function (set as in GPyopt. LCB = Lower Confidence Bound), C = number of categories


BO is coded as a minimization problem, hence please set your objective function accordingly


### Contact:
Dr Shivapratap Gopakumar, shivapratap@gmail.com

### Reference:
    Shivapratap Gopakumar, Vu Nguyen,  Sunil Gupta, Santu Rana, and Svetha Venkatesh. "Algorithmic Assurance: An Active Approach to Algorithmic Testing using Bayesian Optimisation" In NIPS 2016.
