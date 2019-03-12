function hyp_opt = minimize_Kostro(hyp, covfunc, X, Y, n_iterations)
% Function to set hyperparameters for GP regression with d sets of
% parallel regression targets all corresponding to the same, single set of
% predictors. Assume a zero mean GP (for now...), exact inference and 
% Gaussian likelihood function
% Based on method presented in supplementary material of 'Correction of 
% inter-scanner and within-subject variance in structural MRI based 
% automated diagnosing', by Kostro et al, Neuroimage, 2014
%
% Arguments:
% 
%   hyp      struct of vectors of cov/lik hyperparameters
%   cov      prior covariance function
%   X        n by f matrix of training inputs
%   Y        n by d matrix of training targets
%   n_iterations    number of iterations of the optimization to do

% function to calculate log-likelihood and gradients of log-likelihood wrt
% the hyperparameters
% d is number of target vectors. As we pass matrix Q = YY' rather than Y
% itself to avoid potentially expensive repeated computation of Q, d cannot
% be derived from data size so pass it as a separate parameter
function [log_lik, grads] = calculate_log_likelihood(hyps, covfunc, X, Q, d)
    
    % get number of subjects
    n = size(Q, 1);
    
    % create vector of gradients
    grads = zeros(size(hyps));
    
    % get noise variance
    var = exp(2 * hyps(end));
    
    % calculate kernel
    K = feval(covfunc{:}, hyps(1:end-1), X);
    K_sigma = K + (var * eye(n));
    K_sigma_inv = inv(K_sigma);
    
    % calculate log likelihood
    log_lik = -1 * ((-d * 0.5 * log(det(K_sigma)))  - (0.5 * sum(sum(K_sigma_inv .* Q))) - (d * n * 0.5 * log(2 * pi)));
   
    % calculate gradients
    % loop through hyperparameters
    for h = 1:length(hyps)
        
        if h < length(hyps)
        
            % get gradient of kernel wrt covariance hyperparameter
            % do this with usual GPML covariance functionality
            K_grad = feval(covfunc{:}, hyps(1:end-1), X, [], h);
              
        else if h == length(hyps)
            
            % get gradient of kernel wrt likelihood hyperparamete
            K_grad =  var * hyps(end) * eye(n);
            
        else
            
            % indexes unknown hyperparameter
            error('unknown hyperparameter!');
            
        end
        
        % calculate gradient
        % according to second equation in section 11.3 of supplementary
        % material of original Kostro paper
        M1 = K_sigma_inv * K_grad;
        M2 = (K_sigma_inv * K_grad * K_sigma_inv) * Q;
        grads(h) = -1 * ((-d * 0.5 * trace(M1)) + (0.5 * trace(M2)));
        
    end
    
end  

mode = 'local_test';

if strcmp(mode, 'cluster')
    
    % add GPML cov directories
    p_old = path;
    addpath('/home/k1511004/Matlab/gpml-matlab-v3.6-2015-07-07/');
    addpath('/home/k1511004/Matlab/gpml-matlab-v3.6-2015-07-07/cov/');
    
    % add directories for minFunc optimization
    addpath('/home/k1511004/Matlab/minFunc_2012/');
    addpath('/home/k1511004/Matlab/minFunc_2012/autoDif/');
    addpath('/home/k1511004/Matlab/minFunc_2012/logisticExample/');
    addpath('/home/k1511004/Matlab/minFunc_2012/logisticExample/minFunc/');
    addpath('/home/k1511004/Matlab/minFunc_2012/logisticExample/minFunc/compiled/');
    addpath('/home/k1511004/Matlab/minFunc_2012/logisticExample/minFunc/mex/');
    
end

% rearrange hyperparameters into single vector
hyps = [hyp.cov hyp.lik]';

% get data size
[n_X f] = size(X);
[n_Y d] = size(Y);

% check if sizes match as they should
if n_X ~= n_Y
    
   error('Matrix of predictors and matrix of targets must represent the same number of subjects!');
    
end

% Create matrix Q = YY';
Q = Y * Y';

% derivative check
% derivativeCheck(@calculate_log_likelihood, hyps, 1, 1, covfunc, X, Q, d)

% do optimisation with minFunc using L-BFGS method
options.Method='lbfgs';
options.MaxIter=n_iterations;
hyps_opt = minFunc(@calculate_log_likelihood, hyps, options, covfunc, X, Q, d);
    
% put optimised hyperparameters back in original structure
hyp_opt.cov = hyps_opt(1:end-1);
hyp_opt.lik = hyps_opt(end);

if strcmp(mode, 'cluster')

    path(p_old);

end

end