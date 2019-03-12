#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:41:50 2018

@author: jonyoung
"""

# function to remove demographic confounding variables and site effects from data
# based on GP using covariance function from 'Correction of inter-scanner and 
# within-subject variance in structural MRI based automated diagnosing'
# by Kostro et al, Neuroimage, 2014
# Unlike the original paper, this function learns an individual set of kernel
# hyperparameters for each variable being corrected, making it more
# accurate but slower.
#
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
from scipy.spatial.distance import pdist
from pyGPs import mean, cov, GPR

def remove_confounds(training_predictors, testing_predictors, training_data, testing_data, training_label, training_group_labels, normalisation, verbose) :

    # start by checking all inputs
    assert (np.shape(training_predictors)[0] == np.shape(training_data)[0]), 'Training predictors and training data must have same number of subjects'
    assert (np.shape(testing_predictors)[0] == np.shape(testing_data)[0]), 'Testing predictors and testing data must have same number of subjects' 
    assert (np.shape(training_predictors)[1] == np.shape(testing_predictors)[1]), 'Training and testing predictors must have same number of variables'
    assert (np.shape(training_data)[1] == np.shape(testing_data)[1]), 'Training and testing data must have same number of variables'
    assert (len(training_group_labels) == np.shape(training_data)[0]), 'Training group labels must have length equal to number of training subjects'
    
    # initialise corrected data
    # initialise corrected data
    corrected_testing_data = np.zeros_like(testing_data)
    corrected_training_data = np.zeros_like(training_data)

    # normalise the training and testing predictors, exploiting boradcasting
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

    # if training group label not equal to 0, just train on subjects with the given label
    if training_label != 0 :
        
        training_predictors = training_predictors[training_group_labels == training_label, :]
        training_data = training_data[training_group_labels == training_label, :];
    
    dists = pdist(training_predictors)

    for i in range(n_variables) :
        
        if (i % 100) == 0 and verbose :
       
            print '%i features processed' % i
         
        # targets are the i'th column of features
        training_targets = training_data[:, i];
        
        # initialise GP for regression
        model = GPR()
        
        # use constant mean, hyp initialised to 0
        m = mean.Const(c=0)
        
        # initial hyperparameter vals:
        # log of median pairwise (training) distance for covSE
        # 0 for covConst
        # 0 (= log(1) ) for all covariance scaling weights
        # NB in pyGPs many covariance functions have built in scaling
        # pay close attention to how it works.
        # hyps: scaling weights are all 0 (=log (1) )
        # hyps are: scale of linear kernel
        # scale of RBF kernel
        # length scale of RBF kernel (= median distance)
        # scale of const kernel
        # scale of noise kernel
        k = cov.Linear(log_sigma = 0) + cov.RBF(log_ell = np.log(np.median(dists)), log_sigma = 0) + cov.Const(log_sigma = 0) + cov.Noise(log_sigma = 0)
        
        # add priors
        model.setPrior(kernel=k, mean=m)
    
        # set likelhood sd
        model.setNoise( log_sigma = np.log(np.std(training_targets)))
    
        # optimise the hyperparameters
        model.optimize(training_predictors, training_targets, numIterations=200)
    
        # make predictions on training data
        ym, ys2, fm, fs2, lp = model.predict(testing_training_predictors)
        
        # store residuals
        corrected_training_data[:, i] = testing_training_data[:, i] - np.squeeze(ym)
        
        # make predictions on testing data
        ym, ys2, fm, fs2, lp = model.predict(testing_predictors)
        
        # store residuals
        corrected_testing_data[:, i] =  testing_data[:, i] - np.squeeze(ym)
           
    return corrected_training_data, corrected_testing_data
        
