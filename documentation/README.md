# Documentation

## stnomics.ate
**Average Treatment Effects models**, with examples in the [ATE example](https://github.com/shoepaladin/statanomics/blob/main/examples/ATE\%20Functions.ipynb). Each of these models use the same inputs, so I will go through the inputs once here. The outputs will vary across models, and I will detail each of these per model.

*Inputs:*              
* data_est (pandas dataframe) - pandas dataframe name that contains the outcome, treatment indicator, and other features
* split_name (string) - the name of the feature created that will be used for splitting the dataset for cross-fitting. (not used for *ols_vanilla*)
* feature_name (list) - names of conditioning features
* outcome_name (string) - name of the outcome feature, assumed to be continuous
* treatment_name (string) - name of the treatment feature, assumed to be binary
* ymodel (object) - a ML model used to predict the continuous outcome feature. 
* tmodel (object) - a ML model used to predict the binary treatment feature. 
* n_data_splits (int) - how many cross-fitting splits you want to do
* aux_dictionary (dict) - a dictionary that may contain the following entries: 
  * 'n_bins': (number of bins used for the propensity binning model),
  * 'upper': (the lower limit of trained propensity scores),
  * 'lower': (the upper limit of trained propensity scores),
  * 'bootstrapreps': (the number of bootstraps used to generate standard errors for specific models)

*Outputs:*
A dictionary with the following elements:
* 'ATE TE': the ATE treatment estimate
* 'ATE SE': the ATE standard error
* 'ATT TE': the ATT treatment estimate
* 'ATT SE': the ATT treatment estimate
* 'PScore': the propensity score, only available if the function explicitly calculates one

### stnomics.ate.ols_vanilla( ) 
This is your regular ordinary least squares (OLS) model. Note that ATE is the same as ATT here because of the constant treatment effects assumption.

### stnomics.ate.propbinning( )
This uses a combination of regression adjustment and propensity binning models to estimate the average treatment effect. The high-level design is to first estimate propensity scores, divide them into a set number of equal sized bins (aux_dictionary['n_bins']), then use different regression models for each bin. 

### stnomics.ate.ipw( )
This implements inverse propensity weighting where it directly requires treatment and control outcomes to estimate the average treatment effect. 
               
### stnomics.ate.ipw_wls( ) 
This uses inverse propensity weighting and estimate the average treatment effect using a weighted least squares regression. 
            
### stnomics.ate.dml.dml_plm( ) 
This implements the Partial Linear Model from the Chernozhukov et al. (2018) _Double/Debiased Machine Learning for Treatmentand Structural Parameters_. Note that ATE is the same as ATT here because of the constant treatment effects assumption.


### stnomics.ate.dml.dml_irm( ) 
This implements the Interactive Regression Model from the Chernozhukov et al. (2018) _Double/Debiased Machine Learning for Treatmentand Structural Parameters_ to estimate the average treatment effect.



## stnomics.hte
**Heterogeneous Treatment Effects models**, with some demonstrations in the [HTE example notebook](https://github.com/shoepaladin/statanomics/blob/main/examples/HTE_Model_Specs.ipynb). Each of these models use the same input API, so I will go through the inputs once here. The outputs will vary across models. I will only indicate when a function may take a specialized input based on 'aux_dictionary'

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


    
### stnomics.hte.het_dml_approaches.HR()
The Heterogeneous Residuals model builds on Semenova, Goldman, Chernozhukov, and Taddy (2021) - https://arxiv.org/abs/1712.09988 - by residualzing the entire matrix of heterogeneous treatments, rather than residualizing only the treatment variable. This means it reframe heterogeneous treatment effects as a multiple treatments problem.

* aux_dictionary['force_second_stage']    (str) {None, 'OLS','CVLasso', 'Lasso'}
	- None   -  provides the pseudo-outcomes, no standard errors provided!
	- 'OLS'  -  does the second stage OLS regression 
	- 'CVLasso' - does feature selection with CV Lasso and OLS regression with a two-way split
	- 'Lasso' - does feature selection with a Lasso and OLS regression with a two-way split
* other_output['coefficients']	Dataframe of OLS coefficients from regression the proxy HTE on *het_feature*
* other_output['Treatment outcome metric']		$R^2$ of predicting the treatment

### stnomics.hte.het_dml_approaches.SGCT()
An implementation of Semenova, Goldman, Chernozhukov, and Taddy (2021) - https://arxiv.org/abs/1712.09988. 

* aux_dictionary['force_second_stage']    (str) {None, 'OLS','CVLasso', 'Lasso'}
	- None   -  provides the pseudo-outcomes, no standard errors provided!
	- 'OLS'  -  does the second stage OLS regression 
	- 'CVLasso' - does feature selection with CV Lasso and OLS regression with a two-way split
	- 'Lasso' - does feature selection with a Lasso and OLS regression with a two-way split

* other_output['coefficients']	Dataframe of OLS coefficients from regression the proxy HTE on *het_feature*
* other_output['Treatment outcome metric']		$R^2$ of predicting the treatment
    
###  stnomics.hte.trees.grf
This implements  [econml](https://github.com/microsoft/EconML)'s GRF.
Check out it's documentation here: https://econml.azurewebsites.net/_autosummary/econml.grf.CausalForest.html
* aux_dictionary['criterion']       'mse','het', default='mse'
* aux_dictionary['honest']           (default=True) whether trees should be trained in an honest manner.
* aux_dictionary['inference']        (default=True) whether inference should be ienabled via out-of-bag bootstrap
* aux_dictionary['subforest_size']   (default=4) The number of trees in each sub-forest that is used in the bootstrap-of-little-bags calculation. The parameter n_estimators must be divisible by subforest_size. Should typically be a small constant. 	

* other_output['alpha']	the alpha value used for inference
* other_output['jac']		the conditional jacobian values
* other_output['hte_feature_importance']		The feature importances based on the amount of parameter heterogeneity they create. The higher, the more important the feature. The importance of a feature is computed as the (normalized) total heterogeneity that the feature creates. 



### stnomics.hte.gml.blp_estimate()
_under construction_    
### gml.gates()
_under construction_    

###  stnomics.hte.other.DR
Based on "Optimal doubly robust estimation of heterogeneous causal effects" from https://arxiv.org/abs/2004.14497 Note that this does not do the kernel weighting version needed. This version can either output the pseudo_outcome, for which standard errors are not provided.

* aux_dictionary['force_second_stage']    (str) {None, 'OLS','CVLasso', 'Lasso'}
	- None   -  provides the pseudo-outcomes, no standard errors provided!
	- 'OLS'  -  does the second stage OLS regression 
	- 'CVLasso' - does feature selection with CV Lasso and OLS regression with a two-way split
	- 'Lasso' - does feature selection with a Lasso and OLS regression with a two-way split
* other_output['coefficients']		 Dataframe of OLS coefficients from regression the proxy HTE on *het_feature*

### stnomics.hte.other.het_ols
This is just heterogeneous OLS, where each feature in *het_feature* is interacted with *treatment_name*
	





## stnomics.diagnostics
A series of functions that can be used to diagnose the unconfoundedness/conditional independence assumption/exogeneity assumptions for causal inference, following the potential outcomes mental model. Some examples can be found in [this example notebook](https://github.com/shoepaladin/statanomics/blob/main/examples/Testing\%20Overlap\%20Feature\%20Balance.ipynb).

### stnomics.diagnostics.selection.hdm_selection(*data_est, 
                        feature_name, outcome_name, treatment_name*)
This function implements Belloni, Chernozhukov, and Hansen (2014) to use LASSO regressions for feature selection. It uses cross-validated linear LASSO and logistic LASSO regressions to choose features that jointly predict the outcome and treatment, respectively.

| Input | Type | Description |
| --- | --- | ---|
|**data_est**  | (DataFrame) | dataframe name
|**feature_name**  | (list)    | list of candiate feature names to choose from
|**outcome_name**  | (String)    | name of outcome in dataframe
|**treatment_name**  | (String)    | name of treatment indicator in dataframe

| Output | Type | Description |
| --- | --- | ---|
|**unique_entries**  | (list) | list of features that predict either treatment or control
|**selected_lasso_features**  | (dict)    | a dictionary with an 'outcome' and 'treatment' entry that records the features that individual predict the outcome or treatment.


### stnomics.diagnostics.balance.propensity_overlap(*df,
                    pscore, treatment*)

This simply plots:
1. propensity scores for treatment and control groups.
2. inverse propensity weights for treatment and control groups.

| Input | Type | Description |
| --- | --- | ---|
|**df**  | (DataFrame) | dataframe name
|**pscore**  | (String)    | name of propensity score in dataframe
|**treatment**  | (String)    | name of treatment indicator in dataframe



### stnomics.diagnostics.balance.feature_balance(*df,
                    distri_check, ipw_diff,
                    feature_list, pscore, treatment*)

This implement a propensity controlling/weighting way of assess unconfoundedness, where we control for the propensity score in an OLS, or do a weighting scheme.

| Input | Type | Description |
| --- | --- | ---|
|**df**  | (DataFrame) | dataframe name
|**feature_list**  | (list)      | list of feature names to check balance for.
|**pscore**  | (String)    | name of propensity score in dataframe
|**treatment**  | (String)    | name of treatment indicator in dataframe
|**dml_ols**  | (String)    | determines whether an OLS or DML Partial Linear Model is used to assess balance. Takes on values of: {'DML','OLS'}
|**dml_model**  | (obj)    | Model to predict the feature, only used of **dml_ols**='DML'

| Output | Type | Description |
| --- | --- | ---|
|**result_df**  | (DataFrame) | dataframe that lists for each row a feature. Columns indicate the difference, pvalues, and t-statistic for different ways of comparing treatment and control: (1) 'Raw': no controls; (2) 'OLS-PScore': from the OLS or DML model. ; (3)   'Raw SD': no controls but with the standardized outcome; (4) 'OLS-PScore SD': from the OLS or DML model but with the standardized outcome.



### stnomics.diagnostics.balance.plot_difference(*results_df, figure_size*)

This uses the outputed dataframe from **feature_balance** to create a figure similar to the _cobalt_ R-package. 

| Input | Type | Description |
| --- | --- | ---|
|**df**  | (DataFrame) | dataframe name
|**figure_size**  | (tuple)   | two-element tuple that tells the figure size




### stnomics.diagnostics.balance.summarize_result_df(*results_df*)

This uses the outputted dataframe from the function __feature_balance__ and shows the distribution of raw differences, OLS differences, and distributions of statistically significant differences.

| Input | Type | Description |
| --- | --- | ---|
|**df**  | (DataFrame) | dataframe name





