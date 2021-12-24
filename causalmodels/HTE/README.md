# causalmodels - HTE

# HTE Library
* Each of these models use the same input API, so I will go through the inputs once here. The outputs will vary across models. I will only indicate when a function may take a specialized input based on 'aux_dictionary'

*Inputs:*     
    data_est        (obj) name of the PanDas dataframe 
    feature_name    (list) list of features used for predictiong treatment and outcome
    outcome_name    (str) name of outcome feature
    treatment_name  (str) name of treatment feature
    het_feature     (list) list of features used for driving heterogeneity
    ymodel          (obj) model used for predicting outcome
    tmodel          (obj) model used for predicting treatment
    n_data_splits   (int) number of splits in the data
    aux_dictionary  (dict) contains various elements, specialized for each function

*Outputs:*
A list with the following elements:
- heterogeneous treatment effect estimates
- heterogeneous treatment effect standard errors
- predicted counterfactual control outcome
- other output, which varies per model.


    
## het_dml_approaches.HR()
The Heterogeneous Residuals model builds on Semenova, Goldman, Chernozhukov, and Taddy (2021) - https://arxiv.org/abs/1712.09988 - by residualzing the entire matrix of heterogeneous treatments, rather than residualizing only the treatment variable. This means it reframe heterogeneous treatment effects as a multiple treatments problem.

* aux_dictionary['force_second_stage']    (str) {None, 'OLS','CVLasso', 'Lasso'}
	- None   -  provides the pseudo-outcomes, no standard errors provided!
	- 'OLS'  -  does the second stage OLS regression 
	- 'CVLasso' - does feature selection with CV Lasso and OLS regression with a two-way split
	- 'Lasso' - does feature selection with a Lasso and OLS regression with a two-way split
* other_output['coefficients']	Dataframe of OLS coefficients from regression the proxy HTE on *het_feature*
* other_output['Treatment outcome metric']		$R^2$ of predicting the treatment

## het_dml_approaches.SGCT()
An implementation of Semenova, Goldman, Chernozhukov, and Taddy (2021) - https://arxiv.org/abs/1712.09988. 

* aux_dictionary['force_second_stage']    (str) {None, 'OLS','CVLasso', 'Lasso'}
	- None   -  provides the pseudo-outcomes, no standard errors provided!
	- 'OLS'  -  does the second stage OLS regression 
	- 'CVLasso' - does feature selection with CV Lasso and OLS regression with a two-way split
	- 'Lasso' - does feature selection with a Lasso and OLS regression with a two-way split

* other_output['coefficients']	Dataframe of OLS coefficients from regression the proxy HTE on *het_feature*
* other_output['Treatment outcome metric']		$R^2$ of predicting the treatment
    
## trees.grf
I am basically doing a wrapper for _econml_'s implementation of GRF.
Check out it's documentation here: https://econml.azurewebsites.net/_autosummary/econml.grf.CausalForest.html
* aux_dictionary['criterion']       'mse','het', default='mse'
* aux_dictionary['honest']           (default=True) whether trees should be trained in an honest manner.
* aux_dictionary['inference']        (default=True) whether inference should be ienabled via out-of-bag bootstrap
* aux_dictionary['subforest_size']   (default=4) The number of trees in each sub-forest that is used in the bootstrap-of-little-bags calculation. The parameter n_estimators must be divisible by subforest_size. Should typically be a small constant. 	

* other_output['alpha']	the alpha value used for inference
* other_output['jac']		the conditional jacobian values
* other_output['hte_feature_importance']		The feature importances based on the amount of parameter heterogeneity they create. The higher, the more important the feature. The importance of a feature is computed as the (normalized) total heterogeneity that the feature creates. 



## gml.blp_estimate()
_under construction_    
## gml.gates()
_under construction_    

## other.DR
Based on "Optimal doubly robust estimation of heterogeneous causal effects" from https://arxiv.org/abs/2004.14497 Note that this does not do the kernel weighting version needed. This version can either output the pseudo_outcome, for which standard errors are not provided.

* aux_dictionary['force_second_stage']    (str) {None, 'OLS','CVLasso', 'Lasso'}
	- None   -  provides the pseudo-outcomes, no standard errors provided!
	- 'OLS'  -  does the second stage OLS regression 
	- 'CVLasso' - does feature selection with CV Lasso and OLS regression with a two-way split
	- 'Lasso' - does feature selection with a Lasso and OLS regression with a two-way split
* other_output['coefficients']		 Dataframe of OLS coefficients from regression the proxy HTE on *het_feature*

## other.het_ols
This is just heterogeneous OLS, where each feature in *het_feature* is interacted with *treatment_name*               