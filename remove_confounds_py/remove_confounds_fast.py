#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 11:51:40 2019

@author: jonyoung
"""
# function to remove demographic confounding variables and site effects from data
# based on GP using covariance function from 'Correction of inter-scanner and 
# within-subject variance in structural MRI based automated diagnosing'
# by Kostro et al, Neuroimage, 2014

# input variables are
# training_predictors:  n_training_subjects by n_predictors array of
#                       predictor variables. Variables can be continous
#                       (e.g. age) or binary (e.g. sex). Categorical
#                       variables (e.g. site, if there are more than two) 
#                       must be one-hot encoded to be used with this script
# testing_predictors:   n_testing_subjects by n_predictors array of
#                       predictor variables, formatted as above.
# training_data:        n_training_subjects by n_variables array of data
#                       to train confound removal
# testing_data:         n_testing_subjects by n_variables array of data
#                       to remove confounds from
# training_label:       if we want to train confound removal on only a
#                       subset of subjects (e.g. only controls) then set
#                       this label to a non-zero value and label the
#                       relevant subjects in training_group_labels. If all
#                       subjects are to be used for training, set this to 0
#                       and training_group_labels will be ignored
# training_group_labels:vector of length n_training_subjects, where
#                       subjects belonging to the group used for training
#                       the confound removal are designated with
#                       training_label and all other subjects are given
#                       another label of any value. Ignored if
#                       training_label = 0
# normalisation:        Boolean - if true, normalise all predictor
#                       variables to [0, 1] range 

import numpy as np
import pyGPs
from scipy.spatial.distance import pdist, squareform
import minimize_Kostro

def remove_confounds_fast(training_predictors, testing_predictors, training_data, testing_data, training_label, training_group_labels, normalisation, verbose) :
    
     # start by checking all inputs
    assert (np.shape(training_predictors)[0] == np.shape(training_data)[0]), 'Training predictors and training data must have same number of subjects'
    assert (np.shape(testing_predictors)[0] == np.shape(testing_data)[0]), 'Testing predictors and testing data must have same number of subjects' 
    assert (np.shape(training_predictors)[1] == np.shape(testing_predictors)[1]), 'Training and testing predictors must have same number of variables'
    assert (np.shape(training_data)[1] == np.shape(testing_data)[1]), 'Training and testing data must have same number of variables'
    assert (len(training_group_labels) == np.shape(training_data)[0]), 'Training group labels must have length equal to number of training subjects'
  
    # initialise corrected data
    corrected_testing_data = np.zeros_like(testing_data)
    corrected_training_data = np.zeros_like(training_data)

    # normalise the training and testing predictors
    if normalisation :
    
        testing_predictors = testing_predictors - np.min(training_predictors, axis=0)
        training_predictors = training_predictors - np.min(training_predictors, axis=0)
        testing_predictors = testing_predictors.astype(float) / np.max(training_predictors, axis=0)
        training_predictors = training_predictors.astype(float) / np.max(training_predictors, axis=0)

    # do regression
    n_variables = np.shape(training_data)[1]

    # if training group label not equal to 0, just train on subjects with
    # the given label
    # first copy original training predictors and data for predictions
    testing_training_predictors = training_predictors
    testing_training_data = training_data


    if not training_label == 0 :
        
        training_predictors = training_predictors[training_group_labels == training_label, :]
        training_data = training_data[training_group_labels == training_label, :]
        
    # set up GP
    
    # calculate distance matrix of training predictors to initialise RBF
    dists = squareform(pdist(training_predictors))
    
    # covariance is linear + RBF + noise
    # these all have built-in scale so no need to introduce extra hyps
    # set scale hyps to unity
    # set RBF length hyp to log of median dist
    k = pyGPs.cov.Linear(log_sigma=np.log(1.0)) + pyGPs.cov.RBF(log_ell=np.log(np.median(dists[:])), log_sigma=np.log(1.0)) + pyGPs.cov.Noise(log_sigma=np.log(1.0))
    
    # zero mean
    m = pyGPs.mean.Zero()
    
    model = pyGPs.GPR()
    model.setPrior(mean=m, kernel=k)
    model.setNoise(log_sigma=np.log(np.std(training_data[:])))
    
    # optimize the hyperparameters by maximizing log-likelihood over all
    # variables
    if verbose :
    
        print('Optimizing hyperparameters...')
    
    hyps_opt = minimize_Kostro.minimize_Kostro(model, training_predictors, training_data, 200)
    
    if verbose :
    
        print('Hyperparameters optimized!')
    
    # set GP with optimized hyperparameters
    # must convert arrays to list
    model.covfunc.hyp = list(hyps_opt[:-1])
    model.setNoise(log_sigma=np.log(np.std(training_data[:])))

    # loop through variables, removing the effects of confounds on each one
    for i in range(n_variables) :
    
        if (i % 1000) == 0 and verbose :
       
            print '%i features processed' % i
                   
        # targets are the i'th column of features
        training_targets = training_data[:, i]
        
        # set training data
        model.setData(training_predictors, training_targets)
            
        # make predictions on training data
        ym, ys2, fm, fs2, lp = model.predict(testing_training_predictors)
        
        # store residuals
        corrected_training_data[:, i] = testing_training_data[:, i] - np.squeeze(ym)
        
        # make predictions on testing data
        ym, ys2, fm, fs2, lp = model.predict(testing_predictors)
        
        # store residuals
        corrected_testing_data[:, i] =  testing_data[:, i] - np.squeeze(ym)
        
    return corrected_training_data, corrected_testing_data
               
 