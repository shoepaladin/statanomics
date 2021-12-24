# causalmodels-ATE


# ATE Library
Each of these models use the same inputs, so I will go through the inputs once here. The outputs will vary across models, and I will detail each of these per model.

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

## ols_vanilla( ) 
This is your regular ordinary least squares (OLS) model. Note that ATE is the same as ATT here because of the constant treatment effects assumption.

## propbinning( )
This uses a combination of regression adjustment and propensity binning models to estimate the average treatment effect. The high-level design is to first estimate propensity scores, divide them into a set number of equal sized bins (aux_dictionary['n_bins']), then use different regression models for each bin. 


## ipw( )
This implements inverse propensity weighting where it directly requires treatment and control outcomes to estimate the average treatment effect. 
               

## ipw_wls( ) 
This uses inverse propensity weighting and estimate the average treatment effect using a weighted least squares regression. 

                              
## dml_plm( ) 
This implements the Partial Linear Model from the Chernozhukov et al. (2018) _Double/Debiased Machine Learning for Treatmentand Structural Parameters_. Note that ATE is the same as ATT here because of the constant treatment effects assumption.


## dml_irm( ) 
This implements the Interactive Regression Model from the Chernozhukov et al. (2018) _Double/Debiased Machine Learning for Treatmentand Structural Parameters_ to estimate the average treatment effect.

               