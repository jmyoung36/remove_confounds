#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:48:28 2019

@author: jonyoung
"""

# Function to set hyperparameters for GP regression with d sets of
# parallel regression targets all corresponding to the same, single set of
# predictors. Assume a zero mean GP (for now...), exact inference and 
# Gaussian likelihood function
# Based on method presented in supplementary material of 'Correction of 
# inter-scanner and within-subject variance in structural MRI based 
# automated diagnosing', by Kostro et al, Neuroimage, 2014

# Arguments: 
#   hyp      vector of cov/lik hyperparameters
#   cov      prior covariance function
#   X        n by f matrix of training inputs
#   Y        n by d matrix of training targets
#   n_iterations    number of iterations of the optimization to do
import numpy as np
from pyGPs.Core import cov
from scipy.optimize import minimize


# function to calculate log-likelihood and gradients of log-likelihood wrt
# the hyperparameters d is number of target vectors. As we pass matrix Q = YY' rather than Y
# itself to avoid potentially expensive repeated computation of Q, d cannot
# be derived from data size so pass it as a separate parameter
def calculate_log_likelihood_grads(hyps, covfunc, X, Q, d) :
    
    # get number of subjects
    n = np.shape(Q)[0]
    
    # create vector of gradients
    grads = np.zeros_like(hyps)
    
    # get noise variance
    var = np.exp(2 * hyps[-1])
    
    # calculate kernel
    covfunc.hyp = hyps[:-1]
    K = covfunc.getCovMatrix(x=X, mode='train')
    K_sigma = K + (var * np.eye(n))
    K_sigma_inv = np.linalg.inv(K_sigma)
    
    # calculate log likelihood
    log_lik = -1 * ((-d * 0.5 * np.log(np.linalg.det(K_sigma))) - (0.5 * np.sum(np.sum(K_sigma_inv * Q))) - (d * n * 0.5 * np.log(2 * np.pi)))
   
    # calculate gradients
    # loop through hyperparameters
    for h in range(len(hyps)) :
        
        if h < len(hyps)-1 :
        
            # get gradient of kernel wrt covariance hyperparameter
            # do this with usual pyGPs covariance functionality
            K_grad = covfunc.getDerMatrix(x=X, der=h, mode='train')
              
        elif h == len(hyps)-1 :
            
            # get gradient of kernel wrt likelihood hyperparameter
            K_grad =  var * hyps[-1] * np.eye(n)
            
        else :
            
            # indexes unknown hyperparameter
            print('unknown hyperparameter!')
            return
        
        # calculate gradient
        # according to second equation in section 11.3 of supplementary
        # material of original Kostro paper
        M1 = np.matmul(K_sigma_inv, K_grad)
        M2 = np.linalg.multi_dot([K_sigma_inv, K_grad, K_sigma_inv, Q])
        grads[h] = -1 * ((-d * 0.5 * np.trace(M1)) + (0.5 * np.trace(M2)))
        
    return log_lik, grads

def minimize_Kostro(model, X, Y, n_iterations) :
    
    # get covariance function and hyps from model
    covfunc = model.covfunc
    hyps_cov = covfunc.hyp
    hyps_lik = model.likfunc.hyp
    hyps = np.array(hyps_cov + hyps_lik)

    # get data size
    n_X, f = np.shape(X)
    n_Y, d = np.shape(Y)
    
    # get cov function and hyps

    # check if sizes match as they should
    if not n_X == n_Y :
    
        print('Matrix of predictors and matrix of targets must represent the same number of subjects!')
        return

    # Create matrix Q = YY';
    Q = np.dot(Y, np.transpose(Y))

    # do optimisation with minFunc using L-BFGS method
    result = minimize(calculate_log_likelihood_grads, hyps, args=(covfunc, X, Q, d), jac=True, options={'maxiter':n_iterations})
    return result.x

