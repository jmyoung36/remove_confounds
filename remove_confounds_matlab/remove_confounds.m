function corrected_data = remove_confounds(training_predictors, testing_predictors, training_data, testing_data, training_label, training_group_labels, normalisation)

% function to remove demographic confounding variables and site effects from data
% based on GP using covariance function from 'Correction of inter-scanner and 
% within-subject variance in structural MRI based automated diagnosing'
% by Kostro et al, Neuroimage, 2014
% Unlike the original paper, this script learns an individual set of kernel
% hyperparameters for each variable being corrected, making it more
% accurate but slower.

% input variables are
% training_predictors:  n_training_subjects by n_predictors array of
%                       predictor variables. Variables can be continous
%                       (e.g. age) or binary (e.g. sex). Categorical
%                       variables (e.g. site, if there are more than two) 
%                       must be one-hot encoded to be used with this script
% testing_predictors:   n_testing_subjects by n_predictors array of
%                       predictor variables, formatted as above.
% training_data:        n_training_subjects by n_variables array of data
%                       to train confound removal
% testing_data:         n_testing_subjects by n_variables array of data
%                       to remove confounds from
% training_label:       if we want to train confound removal on only a
%                       subset of subjects (e.g. only controls) then set
%                       this label to a non-zero value and label the
%                       relevant subjects in training_group_labels. If all
%                       subjects are to be used for training, set this to 0
%                       and training_group_labels will be ignored
% training_group_labels:vector of length n_training_subjects, where
%                       subjects belonging to the group used for training
%                       the confound removal are designated with
%                       training_label and all other subjects are given
%                       another label of any value. Ignored if
%                       training_label = 0
% normalisation:        Boolean - if true, normalise all predictor
%                       variables to [0, 1] range 


% check inputs
if size(training_predictors, 1) ~= size(training_data, 1)
    
    error('Training predictors and training data must have same number of subjects');
    
end

if size(testing_predictors, 1) ~= size(testing_data, 1)
    
    error('Testing predictors and testing data must have same number of subjects');
    
end

if size(training_predictors, 2) ~= size(testing_predictors, 2)
    
    error('Training and testing predictors must have same number of variables');
    
end

if size(training_data, 2) ~= size(testing_data, 2)
    
    error('Training and testing data must have same number of variables');
    
end

if length(training_group_labels) ~= size(training_data, 1)
    
    error('Training group labels must have length equal to number of training subjects');
    
end

% initialise corrected data
corrected_data = zeros(size(testing_data));

% normalise the training and testing predictors
if normalisation
    
    testing_predictors = bsxfun(@minus, testing_predictors, min(training_predictors));
    training_predictors = bsxfun(@minus, training_predictors, min(training_predictors));
    testing_predictors = bsxfun(@rdivide, testing_predictors, max(training_predictors));
    training_predictors = bsxfun(@rdivide, training_predictors, max(training_predictors));
         
end

% do regression
n_variables = size(training_data, 2);

% if training group label not equal to 0, just train on subjects with
% the given label
if training_label ~= 0
        
        training_predictors = training_predictors(training_group_labels == training_label, :);
        training_data = training_data(training_labels == -1, :);
        
end

for i = 1:n_variables
         
    % targets are the j'th column of features
    training_targets = training_data(:, i);
    testing_targets = testing_data(:, i);
        
   
    % initial hyperparameter vals:
    % log of median pairwise (training) distance for covSE
    % 0 for covConst
    % 0 (= log(1) ) for all covariance scaling weights
    dists = pdist(training_predictors);
    meanfunc = @meanConst; hyp.mean = 0;
    likfunc = @likGauss; hyp.lik = log(std(training_targets));
    cov_lin={@covLIN};
    cov_SE={@covSEisoU}; hyp_cov_SE = log(median(dists(:)));
    cov_const={@covConst}; hyp_cov_const = 0;
    cov_eye={@covEye};
    cov_lin_scaled={@covScale,{cov_lin{:}}}; hyp_cov_lin_scaled = 0;
    cov_SE_scaled={@covScale,{cov_SE{:}}}; hyp_cov_SE_scaled = [0 hyp_cov_SE];
    cov_eye_scaled={@covScale,{cov_eye{:}}}; hyp_cov_eye_scaled = 0;
    cov_sum_scaled={@covSum,{cov_lin_scaled,cov_SE_scaled,cov_const, cov_eye_scaled}}; hyp_cov_sum_scaled = [hyp_cov_lin_scaled hyp_cov_SE_scaled hyp_cov_const hyp_cov_eye_scaled];
    hyp.cov = hyp_cov_sum_scaled;
    covfunc = cov_sum_scaled;
    
    % optimise the hyperparameters
    hyp_opt = minimize(hyp, @gp, -200, @infExact, meanfunc, covfunc, likfunc, training_predictors, training_targets);

    % make predictions on testing data
    [mu s2] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc, training_predictors, training_targets, testing_predictors);
        
    % store residuals
    corrected_data(:, i) =  testing_targets - mu;
           
end
        
end
