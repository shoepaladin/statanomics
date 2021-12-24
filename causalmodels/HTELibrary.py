#!/usr/bin/env python
# coding: utf-8

# # Heterogenous treatment effect models
# Julian Hsu
# 30 Aug 2021
# 
# ### Table of Contents with Navigation Links
# * [Write ML Models](#Section1)
# * [Simulator Functions](#Section2)
# * [Many Simulations](#Section3)

# In[15]:


import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np
import os as os 
import scipy.stats 

from matplotlib import gridspec
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import statsmodels.api as sm
from IPython.display import display    

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression, Lasso, LassoCV, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error

import warnings


# In[16]:


## 


# <a id='Section1'></a>
# ## <font color='red'> Write ML Models </font>
# Write several functions here for estimate HTE. Each model _must_ do datasplitting.
# These functions will do a lot of predictions, so try to standardize the prediction models.
# 
# Each of the model takes this standardized input, so I will only indicate when a function may take a specialized input based on 'aux_dictionary'
# 
#     data_est        (obj) name of the PanDas dataframe 
#     feature_name    (list) list of features used for predictiong treatment and outcome
#     outcome_name    (str) name of outcome feature
#     treatment_name  (str) name of treatment feature
#     het_feature     (list) list of features used for driving heterogeneity
#     ymodel          (obj) model used for predicting outcome
#     tmodel          (obj) model used for predicting treatment
#     n_data_splits   (int) number of splits in the data
#     aux_dictionary  (dict) contains various elements, specialized for each function

# #### Here are some standardized functions that will be used across models.

# In[17]:


def rmse(truth, estimate):
    return     np.sqrt(np.sum( (truth-estimate)**2) / (len(truth)))

def mae(truth, estimate):
    return np.sum( np.abs(truth-estimate)) / (len(truth))

def mape(truth,estimate):
    return np.average( np.abs(truth - estimate)/truth ) 

def r2(truth,estimate):
    return np.sqrt(  np.corrcoef( truth, estimate)[0,1]  )


lasso_max_iter = 1000
def second_stage(approach, test_data, train_data, covar_list, het_feature ):
    '''
    Outputs treatment estimates, standard errors, and OLS coefficient results    
    
    For simplification, it runs the ols of 'y' ~ 'covar_list', where 'covar_list' is the heterogeneous features that includes interactions
    
    where 'het_feature' is heterogeneous features. no interactions.
    ''' 

    if approach=='OLS':
        ## Now run OLS regression on the test set.
        ## Predict the treatment assume all observations in the test set are treated.
        X = sm.add_constant(train_data[covar_list])
        finalmodel = sm.OLS(train_data['y'], X)
        finalmodel_fit = finalmodel.fit()
        ## To estimate the individual treatment, assume that all observations are treated.
        X_test = pd.concat([test_data[['cons','ones']], test_data[het_feature] ], axis=1)
        treatment_estimate = finalmodel_fit.predict( X_test )

    elif approach=='CVLasso':
        ## Train lasso on the training dataset, and recover the selected features.
        ## As a default, keep the main treatment residual in the selected features.
        X = sm.add_constant(train_data[covar_list])            

        lasso_selection = LassoCV(cv=5, random_state=27, n_jobs=-1).fit(X, train_data['y'])
        lasso_fit = Lasso(alpha=lasso_selection.alpha_, max_iter=lasso_max_iter).fit(X, train_data['y']) 
        selected_lasso_features = []
        for x,b in zip(X.columns, lasso_fit.coef_):
            if (b != 0) & (x!='const') & (x !='t'):
                selected_lasso_features.append(x)
            else:
                pass

        ## Now run OLS regression on the test set.
        ## Predict the treatment assume all observations in the test set are treated.
        X = sm.add_constant(test_data[['t']+selected_lasso_features])                    
        finalmodel = sm.OLS(test_data['y'], X)
        finalmodel_fit = finalmodel.fit()
        ## To estimate the individual treatment, assume that all observations are treated.   
        X_test = pd.concat([test_data[['cons','ones']], test_data[[h[3:] for h in selected_lasso_features]] ], axis=1)
        treatment_estimate = finalmodel_fit.predict( X_test )
    elif approach=='Lasso':
        X = sm.add_constant(train_data[covar_list])            
        lasso_fit = Lasso(alpha=lasso_alpha, max_iter=lasso_max_iter).fit(X, train_data['y'])            
        selected_lasso_features = []
        for x,b in zip(X.columns, lasso_fit.coef_):
            if (b != 0) & (x!='const') & (x !='t'):
                selected_lasso_features.append(x)
            else:
                pass

        ## Now run OLS regression on the test set.
        ## Predict the treatment assume all observations in the test set are treated.
        X = sm.add_constant(test_data[['t']+selected_lasso_features])                    
        finalmodel = sm.OLS(test_data['y'], X)
        finalmodel_fit = finalmodel.fit()
        ## To estimate the individual treatment, assume that all observations are treated.     
        X_test = pd.concat([test_data[['cons','ones']], test_data[[h[3:] for h in selected_lasso_features]] ], axis=1)
        treatment_estimate = finalmodel_fit.predict( X_test )
        
    else:
        print('Did not choose an option!')

        
    ## Estimate standard errors on the test dataset
    ## Want all of the variance-covariance matrix except for the intercept term.
    var_cov = np.array(  finalmodel_fit.cov_params()  )[ 1:, 1:]

    X1 = np.dot( np.array(X_test)[:,1:], var_cov)
    output_se = np.sqrt( np.abs(np.dot( X1, np.array(finalmodel_fit.params[1:]).astype(float) ) ))    

    ## Output dataframe of final stage OLS results
    finalmodel_fit_coef = pd.DataFrame(index = finalmodel_fit.model.exog_names, data={'coef':finalmodel_fit.params, 
                                                                                       'se':finalmodel_fit.bse,
                                                                                       'pvalue':finalmodel_fit.pvalues,
                                                                                       'N':finalmodel_fit.model.nobs
                                                                                      })    
    return list(treatment_estimate), list(output_se), finalmodel_fit_coef

def second_stage_no_interactions(approach, test_data, train_data, het_feature ):
    '''just like second_stage, but there are not interactions involved'''
    if approach=='OLS':
        ## Now run OLS regression on the test set.
        ## Predict the treatment assume all observations in the test set are treated.
        X = sm.add_constant(train_data[het_feature])
        finalmodel = sm.OLS(train_data['y'], X)
        finalmodel_fit = finalmodel.fit()
        ## To estimate the individual treatment, assume that all observations are treated.
        X_test = sm.add_constant(test_data[het_feature])
        treatment_estimate = finalmodel_fit.predict( X_test )
        
    elif approach=='CVLasso':
        ## Train lasso on the training dataset, and recover the selected features.
        ## As a default, keep the main treatment residual in the selected features.
        X = sm.add_constant(train_data[het_feature])            

        lasso_selection = LassoCV(cv=5, random_state=27, n_jobs=-1).fit(X, train_data['y'])
        lasso_fit = Lasso(alpha=lasso_selection.alpha_, max_iter=lasso_max_iter).fit(X, train_data['y']) 
        selected_lasso_features = []
        for x,b in zip(X.columns, lasso_fit.coef_):
            if (b != 0) & (x!='const'):
                selected_lasso_features.append(x)
            else:
                pass

        ## Now run OLS regression on the test set.
        ## Predict the treatment assume all observations in the test set are treated.
        X_test = sm.add_constant(test_data[selected_lasso_features])                    
        finalmodel = sm.OLS(test_data['y'], X_test)
        finalmodel_fit = finalmodel.fit()
        treatment_estimate = finalmodel_fit.predict( X_test )
    elif approach=='Lasso':
        X = sm.add_constant(train_data[het_feature])            
        lasso_fit = Lasso(alpha=lasso_alpha, max_iter=lasso_max_iter).fit(X, train_data['y'])            
        selected_lasso_features = []
        for x,b in zip(X.columns, lasso_fit.coef_):
            if (b != 0) & (x!='const'):
                selected_lasso_features.append(x)
            else:
                pass

        ## Now run OLS regression on the test set.
        ## Predict the treatment assume all observations in the test set are treated.
        X_test = sm.add_constant(test_data[selected_lasso_features])                    
        finalmodel = sm.OLS(test_data['y'], X_test)
        finalmodel_fit = finalmodel.fit()
        treatment_estimate = finalmodel_fit.predict( X_test )        
    else:
        print('Did not choose an option!')
    ## Estimate standard errors on the test dataset
    ## Want all of the variance-covariance matrix except for the intercept term.
    var_cov = np.array(  finalmodel_fit.cov_params()  )[ 1:, 1:]

    X1 = np.dot( np.array(X_test)[:,1:], var_cov)
    output_se = np.sqrt( np.abs(np.dot( X1, np.array(finalmodel_fit.params[1:]).astype(float) ) ))    

    ## Output dataframe of final stage OLS results
    finalmodel_fit_coef = pd.DataFrame(index = finalmodel_fit.model.exog_names, data={'coef':finalmodel_fit.params, 
                                                                                       'se':finalmodel_fit.bse,
                                                                                       'pvalue':finalmodel_fit.pvalues,
                                                                                       'N':finalmodel_fit.model.nobs
                                                                                      })    
    return list(treatment_estimate), list(output_se), finalmodel_fit_coef
        


# In[18]:


## Standardized function for assigning data splits WITHOUT reordering the data
def block_splits(data_est=pd.DataFrame(), split_name='splits', n_data_splits=4):
    ## assign data-splitting based on a sequence.
    ## This assumes the data is already randomly sorted, but it accommodates cases where the
    ## propensity score or predicted outcome is already estimated
    data_est[split_name] = np.zeros(len(data_est))
    interval = int( len(data_est) / n_data_splits)
    for p in range(n_data_splits):
        lower = interval*p
        upper = interval*(p+1)
        if p==n_data_splits-1:
            upper = len(data_est)
        mask2 = np.zeros(len(data_est))
        mask2[lower:upper] = 1
        mask2 = pd.Series(mask2.astype(bool))
        data_est.loc[mask2.to_list(), split_name] = p
    
        


# In[19]:


## Standardized Function for Predicting the Treatment Indicator.
def predict_treatment_indicator(dataset, split_name, n_data_splits, feature,treatment, model):
    treatment_hat = []
    for r in np.arange(n_data_splits):
        train = (dataset[split_name] != r)
        test = (dataset[split_name]==r)
        lg = model.fit(dataset[feature][train==True],dataset[treatment][train==True])
        prediction = lg.predict_proba(dataset[feature][test==True])[:,1]
        treatment_hat.extend(prediction)
    return np.array(treatment_hat)


## Standardized Function for Predicting Counterfactual Outcomes.
def predict_counterfactual_outcomes(dataset, split_name, n_data_splits, feature, treatment, outcome,model):
    yhat_treat = []
    yhat_control = []
    for r in np.arange(n_data_splits):
        train = (dataset[split_name] != r)
        test = (dataset[split_name]==r)            
        bin_control = (dataset[treatment]==0)
        bin_treat = (dataset[treatment]==1)        

        ## Predict counterfactual outcomes for treatment
        ols_treat=model.fit(dataset[feature][(bin_treat==True) & (train==True)], dataset[outcome][(bin_treat==True) & (train==True)]) 
        prediction = ols_treat.predict(dataset[feature][(test==True)])
        yhat_treat.extend(prediction)
        
        ## Predict counterfactual outcomes for control
        ols_control=model.fit(dataset[feature][(bin_control==True) & (train==True)], dataset[outcome][(bin_control==True) & (train==True)]) 
        prediction = ols_control.predict(dataset[feature][(test==True)])
        yhat_control.extend(prediction)
    return np.array(yhat_treat), np.array(yhat_control)


# #### DML based approaches

# In[20]:


class other:
    '''
    Based on "Optimal doubly robust estimation of heterogeneous causal effects" 
    from https://arxiv.org/abs/2004.14497
    note that this does not do the kernel weighting version needed.
    This version can either output the pseudo_outcome, for which standard errors are not provided.
    
    
    aux_dictionary['force_second_stage']    (str) {None, 'OLS','CVLasso', 'Lasso'}
        None   -  provides the pseudo-outcomes, no standard errors provided!
        'OLS'  -  does the second stage OLS regression 
        'CVLasso' - does feature selection with CV Lasso and OLS regression with a two-way split
        'Lasso' - does feature selection with a Lasso and OLS regression with a two-way split
    '''
    
    def DR(data_est, 
                    feature_name, outcome_name, treatment_name,
                    het_feature,
                    ymodel,tmodel,
                   n_data_splits, 
                     aux_dictionary):
        block_splits(data_est, 'splits', n_data_splits)
        
        ## Calculate propensity score
        that = predict_treatment_indicator(data_est, 'splits', n_data_splits, feature_name,treatment_name, tmodel)
        
        ## Calculcate the counterfactual outcomes
        yhat_treat, yhat_control = predict_counterfactual_outcomes(data_est, 'splits', n_data_splits, feature_name, treatment_name, outcome_name,ymodel)
        
        ra_portion = yhat_treat - yhat_control
        adj_treatment = (data_est[treatment_name]==1)*(data_est[outcome_name] - yhat_treat)/that
        adj_control = (data_est[treatment_name]==0)*(data_est[outcome_name] - yhat_control)/(1-that)
        
        pseudo_outcome = ra_portion - adj_treatment + adj_control

        output_baseline_hat = yhat_control[:]
        
        if (aux_dictionary['force_second_stage']==None):
            output_treatment_estimate = pseudo_outcome[:]
            output_se_hat = np.ones(len(pseudo_outcome))*(-1)
            
            other_output = {}
        else:
            approach = aux_dictionary['force_second_stage']
            
            ols_coef_pd = {}

            data_for_2nd_stage = data_est.copy()
            data_for_2nd_stage['y'] = pseudo_outcome[:]
            data_for_2nd_stage['t'] = data_est[treatment_name].copy()
        
            data_for_2nd_stage['half'] = 0
            half = np.int(len(data_for_2nd_stage) / 2)
            data_for_2nd_stage.loc[data_for_2nd_stage.iloc[-half:].index, 'half' ] = 1

            data_for_2nd_stage['cons'] = 1
            data_for_2nd_stage['ones'] = 1

            data_est_half = {'0': data_for_2nd_stage.loc[data_for_2nd_stage['half']==0], 
                           '1': data_for_2nd_stage.loc[data_for_2nd_stage['half']==1]}            
            output_treatment_estimate = []
            output_se_hat = []        
            for test_i,train_i in zip(['0','1'], ['1','0']):            
                test_data = data_est_half[test_i]
                train_data = data_est_half[train_i]
                treatment_estimate,se_estimate, coef_pd = second_stage_no_interactions(approach, test_data, train_data, het_feature )                    
                output_treatment_estimate.extend(list(treatment_estimate))            
                output_se_hat.extend(list(se_estimate))            
                ols_coef_pd[test_i] = coef_pd.copy()

            other_output = {'coefficients':ols_coef_pd}
#                             'Treatment outcome metric':t_r2, 
#                             'Outcome prediction metric':y_r2}
            
        ## Output the treatment estimate and propensity scores
        return output_treatment_estimate, output_se_hat, that, output_baseline_hat, other_output    
        
        
    def het_ols(data_est, 
                        feature_name, outcome_name, treatment_name,
                        het_feature,
                        ymodel,tmodel,
                       n_data_splits, 
                         force_second_stage):
        block_splits(data_est, 'splits', n_data_splits)        
        output_treatment_estimate = []
        output_se_hat = []
        that = []
        output_baseline_hat = []

        ## Create interactions of each feature with treatment.
        het_interactions = []
        for x in het_feature:
            data_est[x+'_x'] = data_est[treatment_name]*data_est[x]
            het_interactions.append(x + '_x')


        ## Run fully interacted regression
        for r in np.arange(n_data_splits):
            train = data_est.loc[ data_est['splits'] != r ]
            test = data_est.loc[ data_est['splits'] == r ] 

            ## Estimate regression on the training dataset
            X = sm.add_constant(train[feature_name + [treatment_name] + het_interactions])
            finalmodel = sm.OLS(train[outcome_name], X)
            finalmodel_fit = finalmodel.fit()

            ## Estimate treatment effects on the test dataset
            treatment_effects = dict(finalmodel_fit.params)    
            treatment_estimates = np.zeros(len(test))
            het_treatment_effects = []
            for x in treatment_effects.keys():
                if x=='T':
                    treatment_estimates += treatment_effects[x]
                    het_treatment_effects.append(treatment_effects[x])
                elif x in het_interactions:
                    treatment_estimates += treatment_effects[x]*test[x.replace('_x','')]
                    het_treatment_effects.append(treatment_effects[x])
                else:
                    pass

            ## Estimate standard errors on the test dataset
            var_cov = np.array(  finalmodel_fit.cov_params()  )[ -1 * len(het_interactions) - 1:, -1 * len(het_interactions) - 1:]
            output_se = np.sqrt( np.abs(np.dot( np.dot( test[[treatment_name] + het_interactions], var_cov), np.array(het_treatment_effects).astype(float) ) ))

            ## Estimate baseline
            output_baseline = test[outcome_name] - test[treatment_name]*treatment_estimates
            output_treatment_estimate.extend(treatment_estimates.tolist())
            output_se_hat.extend(output_se.tolist())
            output_baseline_hat.extend(output_baseline.tolist())

        return output_treatment_estimate, output_se_hat, that, output_baseline_hat
        


# #### Generalized Random Forest Models.
# I am basically doing a wrapper for _econml_'s implementation of GRF.
# Check out it's documentation here: https://econml.azurewebsites.net/_autosummary/econml.grf.CausalForest.html
# ***
# **inputs specific to GRF**
# 
#     criterion       'mse','het', default='mse'
#     honest           (default=True) whether trees should be trained in an honest manner.
#     inference        (default=True) whether inference should be ienabled via out-of-bag bootstrap
#     subforest_size   (default=4) The number of trees in each sub-forest that is used in the bootstrap-of-little-bags calculation. The parameter n_estimators must be divisible by subforest_size. Should typically be a small constant. 
# 
# **these are standard forest inputs, so I won't define them here**
# 
#     n_estimators     number of trees
#     max_depth
#     min_samples_split
#     min_samples_leaf
#     min_weight_fraction_leaf
#     min_var_fraction_leaf
#     min_var_leaf_on_val
#     max_features
#     min_impurity_decrease
#     max_samples
#     min_balancedness_tol
#     fit_intercept
#     n_jobs
#     random_state
#     verbose
#     warm_start
# 
# 

# In[ ]:





# In[21]:


from econml.grf import CausalForest, RegressionForest

class trees:
## predict() >> predict_full() or predict_tree_average >> predict_tree_average_full
## predict(X) returns the prefix of relevant fitted local parameters
##.   theta(X)[1,...,n_relevant_outputs], [lb(x),ub(x)]
## predict_full(X, interval=False, alpha=0.05) - returns fitted local parameters for each x in X: 
##.   theta(x), [lb(x), ub(x)]
## predict_interval(X, alpha=0.05)
##.   lb(x), ub(x)
## predict_tree_average(X) returns the prefix of relevant fitted parameters for each X. The average of parameters estimates by each tree.
##.   theta(X)[1,...,n_relevant_outputs]
## predict_tree_average_full(X) returns the fitted local parameter for each X
##.   theta(x), [lb(x), ub(x)]
##
## predict_var()  returns the covariate matrix
## prediction_stderr() returns the standard error of each coordiate of the prefix
##
## predict_alpha_and_jac(X, slice=None, parallel=True)  Return the value of the conditional jacobian E[J | X=x] and the conditional alpha E[A | X=x] using the forest as kernel weight.
##    alpha, jac

    def grf(data_est, 
                    feature_name, outcome_name, treatment_name,
                    het_feature,
                    ymodel,tmodel,
                   n_data_splits, 
                     aux_dictionary):
            
        for x,y in zip( ['criterion','n_estimators','n_jobs'], ['mse',100, 10] ):
            if x in aux_dictionary.keys():
                pass
            else:
                aux_dictionary[x] = y

        ## Estimate a propensity score model using RegressionForest
        grf_rf = RegressionForest(n_estimators=aux_dictionary['n_estimators'], 
                        honest=True, 
                        inference=True, 
                        n_jobs=aux_dictionary['n_jobs'])
        grf_rf_fit = grf_rf.fit(data_est[feature_name], data_est[treatment_name])
        that = grf_rf_fit.predict(data_est[feature_name]).flatten()
        
        grf_rf_fit = grf_rf.fit(data_est.loc[data_est[treatment_name]==0][feature_name], 
                                data_est.loc[data_est[treatment_name]==0][outcome_name])        
        output_baseline_hat = grf_rf_fit.predict(data_est[feature_name]) .flatten()
        
        
        ## Estimate the causal forest to get treatment effect estimates
        cf=CausalForest(n_estimators=aux_dictionary['n_estimators'], 
                        criterion=aux_dictionary['criterion'],
                        honest=True, 
                        inference=True, 
                        fit_intercept=True, n_jobs=aux_dictionary['n_jobs'])
        cf_fit = cf.fit(data_est[feature_name], data_est[treatment_name], data_est[outcome_name])

        output_treatment_estimate = cf_fit.predict(data_est[feature_name], interval=True, alpha=0.05)[0]
        output_se_hat = cf_fit.prediction_stderr(data_est[feature_name])
        alpha,jac = cf_fit.predict_alpha_and_jac(data_est[feature_name],parallel=True)
        other_output = {'alpha':alpha, 'jac':jac, 'hte_feature_importance':cf_fit.feature_importances_}

        return output_treatment_estimate, output_se_hat, that, output_baseline_hat,other_output


# #### Implement Generic ML (GML) Inference metrics
# let's look at the best linear predictor (BLP) metric based on the Horvitz-Thompson transformations of the estimates. Code up a version that intakes estimated HTEs, rather than trying to estimate different ones each time.

# In[22]:


'''
Estimate metrics from GML.
'''
def horvitz_thompson_transform(df, x,p,t):    
    return (df[t]==1)*df[x]/df[p] + (df[t]==0)*df[x]/(1-df[p])

class gml:
    ## This function is UNDER CONSTRUCTION because I need to figure out a way to export the model instead of just estimates.
    def blp_estimate(data_est, 
                    feature_name, outcome_name, treatment_name,
                    het_feature,
                    ymodel,tmodel,
                   n_data_splits, 
                     aux_dictionary, data_splits, hte_model):
        block_splits(data_est, 'splits', n_data_splits)
        
        that = predict_treatment_indicator(data_est, 'splits', n_data_splits, feature_name,treatment_name, tmodel)

        data_est['prob'] = that
        reported_beta_list = []
        reported_het_var_list = []
        for r in range(data_splits):
            ## 1. Split the data in half.
            split_data = data_est.loc[ np.random.uniform(0,1, len(data_est)) < 0.5 ]

            ## 2. Calculate the HT transformed outcome and each feature.
            split_data['yt'] = horvitz_thompson_transform(split_data, outcome_name, 'prob', treatment_name)
            for x in feature_name:
                split_data[x+'_t'] = horvitz_thompson_transform(split_data, x , 'prob', treatment_name)
            ## 3a. Calculate HTE for this one split.            
            split_data['het'] = hte_model(split_data, 
                                feature_name, outcome_name, treatment_name,
                                het_feature,
                                ymodel,tmodel,
                               n_data_splits, 
                                 aux_dictionary)[0]

            ## 3b. Calculate the deviation away from the average 
            split_data['het_t'] = split_data['het'] - np.average(split_data['het'])

            ## 4. Run the regression on the transformed outcome.
            X = sm.add_constant( split_data[['het_t'] + [x+'_t' for x in feature_name  ] ] ) 
            model = sm.OLS(split_data['yt'], X).fit()
    #         print(model.summary())

            ## 5. Gather the beta values.
            reported_beta = model.params[1] 
            reported_beta_list.append(reported_beta)
            
            ## 6. Gather variance in HTE in this split
            het_var = np.var(split_data['het'])            
            reported_het_var_list.append(het_var)
        return het_var, reported_beta_list


    def blp(data_est, 
                    feature_name, outcome_name, treatment_name,
                    prob, het,  baseline,               
                   n_data_splits):
        block_splits(data_est, 'splits', n_data_splits)        
        reported_beta2_list = []
        reported_pvalue2_list = []
        
        reported_beta1_list = []
        reported_pvalue1_list = []
        
        for r in range(n_data_splits):
            ## 1. Denote the data split based on a bootstrapping procedure                                  
            bt_index = np.random.choice(len(data_est), len(data_est), replace=True)                                  
            split_data = data_est.iloc[bt_index]
            
#             ## 2. Calculate the HT transformed outcome and each feature.
#             split_data['yt'] = horvitz_thompson_transform(split_data, outcome_name, prob, treatment_name)
#             split_data['baselinet'] = horvitz_thompson_transform(split_data, baseline , prob, treatment_name)
#             ## 3. Calculate the deviation away from the average 
#             split_data['het_t'] = split_data[het] - np.average(split_data[het])

#             ## 4. Run the regression on the transformed outcome.
#             X = sm.add_constant( split_data[['het_t','baselinet'] ] ) 
#             model = sm.OLS(split_data['yt'], X).fit()
# #             print(model.summary())

#             ## 5. Gather the beta values.
#             reported_beta2_list.append(model.params['het_t'] )
#             reported_pvalue2_list.append(model.pvalues['het_t'] )
            
#             reported_beta1_list.append(model.params['const'])
#             reported_pvalue1_list.append(model.pvalues['const'])
            
            
            ######
            ## Another way to do this is to do WLS
            split_data['res_t'] = split_data[treatment_name] - split_data[prob]
            split_data['res_t_x_res_hte'] = split_data['res_t']*(split_data[het] - np.average(split_data[het]))
            X = sm.add_constant(split_data[[baseline] + ['res_t','res_t_x_res_hte']] )
#             model = sm.WLS.fit(split_data[outcome_name], X , weights = (split_data[prob]*(1-split_data[prob]))**(-2) )
#             print(X.dtypes )
            model = sm.WLS(split_data[outcome_name], X ).fit()            
            
            ## 5. Gather the beta values.
            reported_beta2_list.append(model.params['res_t_x_res_hte'] )
            reported_pvalue2_list.append(model.pvalues['res_t_x_res_hte'] )
            
            reported_beta1_list.append(model.params['res_t'])
            reported_pvalue1_list.append(model.pvalues['res_t'])
            
        het_var = np.var(data_est[het])
        reported_beta2_list = np.array(reported_beta2_list)
        reported_pvalue2_list = np.array(reported_pvalue2_list)
        
        reported_beta1_list = np.array(reported_beta1_list)
        reported_pvalue1_list = np.array(reported_pvalue1_list)
        
        output_dict = {'beta1_list': reported_beta1_list,
                       'beta1': np.median(reported_beta1_list[~np.isnan(reported_beta1_list)]),
                      'pvalue1_list':reported_pvalue1_list,
                      'pvalue1_adj': np.median( reported_pvalue1_list[~np.isnan(reported_pvalue1_list)]  )*2,
                      'beta2_list': reported_beta2_list,
                      'beta2': np.median(reported_beta2_list[~np.isnan(reported_beta2_list)]),
                      'pvalue2_list':reported_pvalue2_list,
                      'pvalue2_adj': np.median( reported_pvalue2_list[~np.isnan(reported_pvalue2_list)]  )*2 }
        return output_dict

    def gates(data_est, 
                    feature_name, outcome_name, treatment_name,
                    prob, het,  
                    n_bins,
                   n_data_splits):
        reported_gamma_list = []
        data_est['yt'] = horvitz_thompson_transform(data_est, outcome_name, prob, treatment_name)
        for x in feature_name:
            data_est[x+'_t'] = horvitz_thompson_transform(data_est, x , prob, treatment_name)

        ## Sorted treatment effects in the het.     
        het_in_bins = np.array_split(np.sort(data_est[het]),n_bins)
        for g in range(n_bins):
            het_bin = het_in_bins[g]
            data_est['gk_'+str(g)] = 0
            data_est.loc[data_est[het].between(het_bin[0],het_bin[-1]) , 'gk_'+str(g)] = 1

        X = sm.add_constant( data_est[[x+'_t' for x in feature_name] + ['gk_'+str(g) for g in range(n_bins)] ] ) 
        model = sm.OLS(data_est['yt'], X).fit()
    #     print(data_est[[x+'_t' for x in feature_name] + ['gk_'+str(g) for g in range(n_bins)] ].describe())
    #     print(model.summary())
        reported_gamma_list = model.params[-n_bins:]

        gates = 0
        for gk,gi in zip(reported_gamma_list, range(n_bins)):
            gates+=gk**2*np.sum(data_est['gk_'+str(gi)])

        return reported_gamma_list,gates


# In[23]:


class het_dml_approaches:
    ## Approach 1
    def HR(data_est, 
                    feature_name, outcome_name, treatment_name,
                    het_feature,
                    ymodel,tmodel,
                   n_data_splits, 
                     aux_dictionary):
        '''    
        aux_dictionary['force_second_stage']    (str) {None, 'OLS','CVLasso', 'Lasso'}
        '''    
        block_splits(data_est, 'splits', n_data_splits)
        
        ## 1st Stage: Predict treatment indicator
        that = predict_treatment_indicator(data_est, 'splits', n_data_splits, feature_name,treatment_name, tmodel)
        ## 1st Stage: Predict treatment indicator interacted with each feature
        ## remember to remove the feature interacted with the treatment as a predictor
        residual_treatment_hat = [data_est[treatment_name] - that]
        for x in het_feature:
            data_est[x+'_x'] = data_est[treatment_name]*data_est[x]
            treatment_hat = []
            for r in np.arange(n_data_splits):
                train = (data_est['splits'] != r)
                test = (data_est['splits']==r)
                ols_pred = ymodel.fit(data_est[feature_name][train==True],
                                                data_est[x+'_x'][train==True])
                prediction = ymodel.predict(data_est[feature_name][test==True])
                treatment_hat.extend(prediction)            

            diff = data_est[x+'_x'] - treatment_hat 
            residual_treatment_hat.append(diff)        

        ## 1st Stage: Predict the outcome
        ## Train a model on a training set, and predict onto the test set.
        outcome_hat = []
        output_baseline_hat = []
        for r in np.arange(n_data_splits):
            train = (data_est['splits'] != r)
            test = (data_est['splits']==r)
            ols_control=ymodel.fit(data_est[feature_name ][(train==True)],
                                   data_est[outcome_name][(train==True)]) 
            prediction = ols_control.predict(data_est[feature_name ][(test==True)])
            outcome_hat.extend(prediction)

            ols_control=ymodel.fit(data_est[feature_name ][( (train==True) & (data_est[treatment_name]==0) )],
                                   data_est[outcome_name][( (train==True) & (data_est[treatment_name]==0) )]) 
            prediction = ols_control.predict(data_est[feature_name ][(test==True)])
            output_baseline_hat.extend(prediction)
            
            
        y_r2 = r2(data_est[outcome_name], outcome_hat)
        ## 2nd Stage: Estimate OLS of residualized treatments on outcome
        ## Interact the residualized treatment with the heterogeneous outcomes    
        residual_outcome = data_est[outcome_name] - outcome_hat    

        data_for_2nd_stage = pd.DataFrame(data={'y': residual_outcome,
                                                treatment_name:data_est[treatment_name],
                                                'outcome_hat':outcome_hat,
                                                't':residual_treatment_hat[0]})
        covar_list = ['t']
        h_i = 1
        for x in het_feature:
            data_for_2nd_stage['t_x'+x] = residual_treatment_hat[h_i]
            data_for_2nd_stage[x] = data_est[x]
            h_i+=1
            covar_list.append('t_x'+x)    


        data_for_2nd_stage['half'] = 0
        half = np.int(len(data_for_2nd_stage) / 2)
        data_for_2nd_stage.loc[data_for_2nd_stage.iloc[-half:].index, 'half' ] = 1

        data_for_2nd_stage['cons'] = 1
        data_for_2nd_stage['ones'] = 1
        
        data_est_half = {'0': data_for_2nd_stage.loc[data_for_2nd_stage['half']==0], 
                       '1': data_for_2nd_stage.loc[data_for_2nd_stage['half']==1]}

        output_treatment_estimate = []
        output_se_hat = []
        
        '''
        If the dimensionality of teratment array relative to sample size is less than 20%, then just do an OLS.
        But if user is forcing OLS, CVLasso, or Lasso, then let them.
        '''        
        if aux_dictionary['force_second_stage'] == None:
            dim_D = len(het_feature)+1
            if (float(dim_D/len(data_est)) <= 0.20):
                approach='OLS'
            else:
                approach='CVLasso'
        else:
            approach = aux_dictionary['force_second_stage']

        ols_coef_pd = {}
        for test_i,train_i in zip(['0','1'], ['1','0']):            
            test_data = data_est_half[test_i]
            train_data = data_est_half[train_i]
            treatment_estimate,se_estimate, coef_pd = second_stage(approach, test_data, train_data, covar_list, het_feature )
            output_treatment_estimate.extend(list(treatment_estimate))            
            output_se_hat.extend(list(se_estimate))            
            ols_coef_pd[test_i] = coef_pd.copy()
            

        other_output = {'coefficients':ols_coef_pd, 
                        'Treatment outcome metric':t_r2}
        ## Output the treatment estimate and propensity scores
        return output_treatment_estimate, output_se_hat, that, output_baseline_hat, other_output



    ## Approach 2
    def SGCT(data_est, 
                    feature_name, outcome_name, treatment_name,
                    het_feature,
                    ymodel,tmodel,
                   n_data_splits, 
                 aux_dictionary):                
        '''    
        aux_dictionary['force_second_stage']    (str) {None, 'OLS','CVLasso', 'Lasso'}
        '''
        block_splits(data_est, 'splits', n_data_splits)
        
        ## 1st Stage: Predict treatment indicator
        that = predict_treatment_indicator(data_est, 'splits', n_data_splits, feature_name,treatment_name, tmodel)

        ## 1st Stage: Residualize Treatment Indicator
        residual_treatment_hat = data_est[treatment_name] - that 

        ## 1st Stage: Predict the outcome
        outcome_hat = []
        output_baseline_hat = []
        for r in np.arange(n_data_splits):
            train = (data_est['splits'] != r)
            test = (data_est['splits']==r)
            ols_control=ymodel.fit(data_est[feature_name ][(train==True)],
                                   data_est[outcome_name][(train==True)]) 
            prediction = ols_control.predict(data_est[feature_name ][(test==True)])
            outcome_hat.extend(prediction)

            ols_control=ymodel.fit(data_est[feature_name ][( (train==True) & (data_est[treatment_name]==0) )],
                                   data_est[outcome_name][( (train==True) & (data_est[treatment_name]==0) )]) 
            prediction = ols_control.predict(data_est[feature_name ][(test==True)])
            output_baseline_hat.extend(prediction)

        ## 2nd Stage: Estimate OLS of residualized treatments on outcome
        ## Interact the residualized treatment with the heterogeneous outcomes    
        residual_outcome = data_est[outcome_name] - outcome_hat  

        data_for_2nd_stage = pd.DataFrame(data={'y': residual_outcome,
                                                treatment_name:data_est[treatment_name],
                                                'outcome_hat':outcome_hat,
                                                't':residual_treatment_hat})
        covar_list = ['t']
        h_i = 1
        for x in het_feature:
            data_for_2nd_stage['t_x'+x] = data_est[x] * residual_treatment_hat
            data_for_2nd_stage[x] = data_est[x]
            h_i+=1
            covar_list.append('t_x'+x)    

        ## We want to do ad-hoc feature selection with a Lasso. However, we cannot directly use a Lasso for 
        ## causal inference because there is feature selection bias (#thatsnothowitworks). Therefore, we 
        ## run Lasso on one half of the dataset, and use the selected features in an OLS on the other half.
        data_for_2nd_stage['half'] = 0
        half = np.int(len(data_for_2nd_stage) / 2)
        data_for_2nd_stage.loc[data_for_2nd_stage.iloc[-half:].index, 'half' ] = 1
        
        data_for_2nd_stage['cons'] = 1
        data_for_2nd_stage['ones'] = 1
        
        data_est_half = {'0': data_for_2nd_stage.loc[data_for_2nd_stage['half']==0], 
                       '1': data_for_2nd_stage.loc[data_for_2nd_stage['half']==1]}

        output_treatment_estimate = []
        output_se_hat = []
        '''
        If the dimensionality of teratment array relative to sample size is less than 20%, then just do an OLS.
        But if user is forcing OLS, CVLasso, or Lasso, then let them.
        '''        
        if aux_dictionary['force_second_stage'] == None:
            dim_D = len(het_feature)+1
            if (float(dim_D/len(data_est)) <= 0.20):
                approach='OLS'
            else:
                approach='CVLasso'
        else:
            approach = aux_dictionary['force_second_stage']
        ols_coef_pd = {}

        for test_i,train_i in zip(['0','1'], ['1','0']):            
            test_data = data_est_half[test_i]
            train_data = data_est_half[train_i]
            treatment_estimate,se_estimate, coef_pd = second_stage(approach, test_data, train_data, covar_list, het_feature )
            output_treatment_estimate.extend(list(treatment_estimate))            
            output_se_hat.extend(list(se_estimate))            
            ols_coef_pd[test_i] = coef_pd.copy()
            
        other_output = {'coefficients':ols_coef_pd, 
                        'Treatment outcome metric':t_r2}
        ## Output the treatment estimate and propensity scores
        return output_treatment_estimate, output_se_hat, that, output_baseline_hat, other_output


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0
    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

def generate_data(N, 
                  dependence,
                             size_non_sparse,
                             size_sparse,
                             treatment_function,
                             outcome_function,
                             hte_function, 
                             known_propensity_score,
                             functional_form):
    '''
    N: : (int) sample size
    dependence : (boolean) for whether generated features are independent (False) or not (True)
    size_non_sparse : (int) number of features that are non-sparse. Half will be binary.
    size_sparse : (int) number of features that are sparse. Half will be binary.
    treatment_function : function for determining treatment
    outcome_function : function for determining the baseline outcome
    hte_function : function for determining the HTE 
    known_propensity_score : (array) whether propensity score is known (not None) or not and needs to calculated 
                                by the function treatment_function (is None)
    functional_form : function that is either {linear, power, polynomial, log, cosine} for HTE only.
    '''
    number_of_binary_non_sparse = int(size_non_sparse/2)
    number_of_binary_sparse = int(size_sparse/2)

    df = pd.DataFrame()
    if dependence==True:        
        cov =  np.zeros((size_non_sparse + size_sparse,size_non_sparse + size_sparse))+0.5
        cov[np.diag_indices(size_non_sparse + size_sparse)] = 1
        x = np.random.multivariate_normal(np.zeros( size_non_sparse + size_sparse ) , cov, N).T

    else:
        x = np.random.multivariate_normal(np.zeros( size_non_sparse + size_sparse) , np.eye(size_non_sparse + size_sparse), N).T        
    ## Iteratively go through features. 
    for f in range(size_non_sparse+size_sparse):
        base = x[f]
        ## 1. Generate a non-sparse continuous feature
        if f < number_of_binary_non_sparse:
            pass
        ## 2. Generate a non-sparse binary feature
        elif (f >= number_of_binary_non_sparse) & (f < size_non_sparse) :
            base = np.array(base > np.percentile(base, 50)).astype(np.int32)

        ## 3. Generate a sparse continuous feature    
        elif (f >= size_non_sparse ) & (f < size_non_sparse+number_of_binary_sparse):
            base[(np.random.uniform(0,1,N)<0.99)] = 0            
        ## 4. Generate a sparse binary feature    
        else:
            base = np.array(base > np.percentile(base, 99)).astype(np.int32)

        df['x_'+str(f)] = base
            
    ## Treatment Assignment
    if known_propensity_score is not None:
        treatment = (np.random.uniform(0,1, N) > known_propensity_score).astype(np.int32)
    else:
        treatment_raw,propensity_score,exp_treatment = treatment_function(df, functional_form)
        treatment = (treatment_raw).astype(np.int32)
        treatment_sd = np.std(treatment)


    ## Baseline outcome
    df['baseline_Y'],exp_outcome = outcome_function(df, functional_form) 

    ## HTE
    df['GT'], exp_hte = hte_function(df, functional_form)

    df['T'] = treatment
    df['T_propensity'] = propensity_score
    df['Y'] = df['baseline_Y'] + df['T']*df['GT']

# Assign squared terms
    for f in range(size_non_sparse+size_sparse):
        if f < number_of_binary_non_sparse:
            df['x_'+str(f)+'__2'] = df['x_'+str(f)].pow(2)
            df['x_'+str(f)+'__3'] = df['x_'+str(f)].pow(3)            
        elif (f >= size_non_sparse ) & (f < size_non_sparse+number_of_binary_sparse):
            df['x_'+str(f)+'__2'] = df['x_'+str(f)].pow(2)
            df['x_'+str(f)+'__3'] = df['x_'+str(f)].pow(3)            

    return df, [exp_treatment, exp_outcome,exp_hte]


# In[25]:


'''
functions below for treatment, baseline outcomes, and hte, which are linear 
combinations of functions of features. There are no interactions between features.
'''
def linear_treatment(df,func):
    N = len(df)
    x = [c for c in df.columns if 'x' in c]
    latent_treatment = np.zeros(N)
    exp_full = ''
    for f in x:
        a,exp = func(df[f])
        latent_treatment += a
        exp_full += exp.replace('x','('+f+')')+' + '
    latent_treatment /= 5
    propensity_score = np.exp(latent_treatment) / (1 + np.exp(latent_treatment))
    treatment = (propensity_score) > np.random.uniform(0,1, N)
    return treatment,propensity_score, exp_full

def linear_outcome(df,func):
    N = len(df)
    x = [c for c in df.columns if 'x' in c]    
    y = np.random.uniform(0,1, N)
    exp_full = ''
    for f in x:
        a,exp = func(df[f])
        y += a
        exp_full += exp.replace('x','('+f+')')+' + '
    return y, exp_full

def linear_hte(df,func):
    N = len(df)
    x = [c for c in df.columns if 'x' in c]    
    hte = np.zeros(N)
    exp_full = ''
    for f in x:
        a,exp = func(df[f])
        hte += a
        exp_full += exp.replace('x','('+f+')')+' + '
    return hte, exp_full

'''
Functional form assumptions of how to transform features.
'''
def power(x):
    a =np.random.uniform(0,3)
    b = np.random.uniform(1,2)
    y =a**((x/100)*b)
    exp = str(np.round(a,2))+'^{ x/100 ' + str(np.round(b,2)) +'}'
    return y, exp

def log(x):
    a = np.random.uniform(0,2)
    y = np.log( np.abs(x) + 1)*a
    exp = str(np.round(a,2))+'log(|x|+a)'
    return y, exp

def linear(x):
    a = np.random.uniform(1,2)
    y = a*x 
    exp = str(np.round(a,2))+'x'
    return y, exp

def polynomial(x):
    a =np.random.uniform(0,3)
    b = np.random.uniform(1,2)    
    y =a*x**(b)
    exp = str(np.round(a,2))+'x^{' + str(np.round(b,2)) +'}'    
    return y, exp


# In[26]:




