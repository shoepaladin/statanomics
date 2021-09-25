#!/usr/bin/env python
# coding: utf-8

# # Causal Inference Examples
# Julian Hsu
# Date Made: 5 Aug 2021 
# 
# ### Table of Contents with Navigation Links
# * [Write Causal Models](#Section1)
# * [Simulate Data](#Section2)
# * [Bootstrapping Examples](#Section3)
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import os as os 

from matplotlib import gridspec
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit

from IPython.display import display    

import scipy.stats 

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error


# <a id='Section1'></a>
# 
# ## Write Causal Models
# Write several functions here for estimate HTE. Each model _must_ do datasplitting.
# These functions will do a lot of predictions, so try to standardize the prediction models.
# 

# In[2]:


from sklearn import metrics
class predQC:
    def treatment(metric= metrics.recall_score, 
                  t_true=np.ones(5), t_hat=np.zeros(5)):
        return metric(t_true, t_hat)
    def outcome(metric= metrics.r2_score, 
                ytrue=np.ones(5), yhat=np.zeros(5), treatment = np.array([0,1,1,0,1]) ):
        overall = metric(ytrue, yhat)
        t = metric(ytrue[(treatment==1)], yhat[(treatment==1)])
        c = metric(ytrue[(treatment==0)], yhat[(treatment==0)])
        return overall, t, c
    def battery(ytrue,yhat,treatment,
                t_true, t_hat,
               tmetric,ymetric):
        ymetrics=outcome(ymetric, ytrue, yhat, treatment)
        return {'Treatment Status Metric Name':tmetric.__name__,
                'Treatment Status Metric': treatment(tmetric, t_true, t_hat),
                'Treatment Status N': len(t_true),                
                'Outcome Metric':ymetric.__name__,
                'Outcome Metric Overall': ymetrics[0],
                'Outcome Metric Treatment': ymetrics[1],
                'Outcome Metric Control': ymetrics[2] ,               
                'Outcome N': len(ytrue),                
                'Outcome Treatment N': (treatment==1).sum(),
                'Outcome Control N': (treatment==0).sum()
               }


# In[3]:


class bootstrap:
    def reps(bootstrapreps, est_model, *args):
        ate_est = []
        ate_se = []
        att_est = []
        att_se = []        
        for b in range(bootstrapreps):
            bt_index = np.random.choice(len(args[0]), 
                                        len(args[0]),
                                        replace=True)
            df_bt = args[0].iloc[bt_index]        
            est = est_model(df_bt, args[1], args[2],
                     args[3], args[4],args[5],args[6], args[7],args[8])
            ate_est.append(est['ATE TE'])
            ate_se.append(est['ATE SE'])
            att_est.append(est['ATT TE'])
            att_se.append(est['ATT SE'])            
        return ate_est, ate_se, att_est, att_se

    def results(bt):
        return np.average(bt), np.std(bt)
    
    def go_reps(bootstrapreps, est_model, *args):
        reps_output = bootstrap.reps(bootstrapreps, est_model, *args)
        ate_te = bootstrap.results(reps_output[0])
        ate_se = bootstrap.results(reps_output[1])
        att_te = bootstrap.results(reps_output[2])
        att_se = bootstrap.results(reps_output[3])        
        return {'model':est_model.__name__, 
                'ATE mean':ate_te[0], 'ATE std':ate_te[1],
                'ATE SE mean':ate_se[0], 'ATE SE std':ate_se[1],
                'ATT mean':ate_te[0], 'ATT std':ate_te[1],
                'ATT SE mean':ate_se[0], 'ATT SE std':ate_se[1]               }
# bootstrap.go_reps(bootstrapreps=50, propbinning, data_est, 
#                 split_name, feature_name, outcome_name, treatment_name,
#                 ymodel,tmodel,
#                n_data_splits,
#                aux_dictionary)


# In[4]:


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

def predict_continuous(dataset, split_name, n_data_splits, feature,outcome, model):
    x_hat = []
    for r in np.arange(n_data_splits):
        train = (dataset[split_name] != r)
        test = (dataset[split_name]==r)
        lg = model.fit(dataset[feature][train==True],dataset[outcome][train==True])
        prediction = lg.predict(dataset[feature][test==True])
        x_hat.extend(prediction)
    return np.array(x_hat)
 


# In[5]:


def ols_vanilla(data_est, 
                split_name, feature_name, outcome_name, treatment_name,
                ymodel,tmodel,
               n_data_splits, aux_dictionary ):
    ols = sm.OLS(data_est[outcome_name], sm.add_constant(data_est[[treatment_name]+feature_name]) ).fit()
    return {'ATE TE':ols.params[1], 'ATE SE': ols.bse[1],'ATT TE':ols.params[1], 'ATT SE': ols.bse[1]}


# In[30]:


def propbinning(data_est, 
                split_name, feature_name, outcome_name, treatment_name,
                ymodel,tmodel,
               n_data_splits,
               aux_dictionary):
    main_result = propbinning_main(data_est, 
                split_name, feature_name, outcome_name, treatment_name,
                ymodel,tmodel,
               n_data_splits,
               aux_dictionary)
    pbin_bt_results = bootstrap.go_reps(aux_dictionary['bootstrapreps'], propbinning_main, data_est, 
                    split_name, feature_name, outcome_name, treatment_name,
                    ymodel,tmodel,
                   n_data_splits,
                   aux_dictionary)
    return {'ATE TE':pbin_bt_results['ATE mean'], 'ATE SE': pbin_bt_results['ATE std'],            'ATT TE':pbin_bt_results['ATT mean'], 'ATT SE': pbin_bt_results['ATT std'], 'PScore':main_result['PScore']}

def propbinning_main(data_est, 
                split_name, feature_name, outcome_name, treatment_name,
                ymodel,tmodel,
               n_data_splits,
               aux_dictionary):
        
    data_est[split_name] = np.random.choice(n_data_splits, len(data_est), replace=True)
    data_est = data_est.sort_values(by=split_name)
        
    ## Predict Treatment
    that = predict_treatment_indicator(data_est, split_name, n_data_splits, feature_name,treatment_name,tmodel)
    data_est['that'] = that
    
    ## Sort by the probabilities and split into bins.
    ## For each bin, estimate counterfactuals for the treatment and control groups.
    data_est['that_bin'] = pd.qcut(data_est['that'], q=aux_dictionary['n_bins'], labels=False)    
    min_size = data_est['that_bin'].value_counts().min()
    
    data_est = data_est.sort_values(by=['that_bin',split_name])
    
    yhat_treat = []
    yhat_control = []
    for b in range(aux_dictionary['n_bins']):
        ## Use the first and last entry of each bin as cutpoints.    
        bin_of_interest = (data_est['that_bin']==b)
        for r in np.arange(n_data_splits):
            train = (bin_of_interest==True) & (data_est[split_name] != r)
            test = (bin_of_interest==True) & (data_est[split_name]==r)            
            bin_control = (bin_of_interest==True) & (data_est[treatment_name]==0)
            bin_treat = (bin_of_interest==True) & (data_est[treatment_name]==1)        

            ## Predict counterfactual outcomes for treatment
            ols_treat=ymodel.fit(data_est[feature_name][(bin_treat==True) & (train==True)], data_est[outcome_name][(bin_treat==True) & (train==True)]) 
            tpred = ols_treat.predict(data_est[feature_name][(test==True) ])
            yhat_treat.extend(tpred)

            ## Predict counterfactual outcomes for control
            ols_control=ymodel.fit(data_est[feature_name][(bin_control==True) & (train==True)], data_est[outcome_name][(bin_control==True) & (train==True)]) 
            cpred = ols_control.predict(data_est[feature_name][(test==True)])
            yhat_control.extend(cpred)

    ## Take the difference between the counterfactuals
    treatment_estimate = np.array(yhat_treat) - np.array(yhat_control)
    
    ## Output the treatment estimate and propensity scores
    return {'ATE TE':np.average(treatment_estimate), 'ATE SE': np.std(treatment_estimate),            'ATT TE':np.average(treatment_estimate[(data_est[treatment_name]==1)]), 'ATT SE': np.std(treatment_estimate[(data_est[treatment_name]==1)]),            'PScore':that}


# In[36]:


'''
Inverse-propensity weighting:
estimator, where asympotics come from Hirano, Imbens, Ridder (2004) Econometrica: 
https://scholar.harvard.edu/imbens/files/efficient_estimation_of_average_treatment_effects_using_the_estimated_propensity_score.pdf

'''

def ipw_main(data_est, 
                split_name, feature_name, outcome_name, treatment_name,
                ymodel,tmodel,
               n_data_splits,
               aux_dictionary):
    ## Create and sort by splits
    data_est[split_name] = np.random.choice(n_data_splits, len(data_est), replace=True)
    data_est.sort_values(by=split_name, inplace=True)

    ## 1st Stage: Predict treatment indicator
    that = predict_treatment_indicator(data_est, split_name, n_data_splits, feature_name,treatment_name,tmodel)

    keep_these = (that >= aux_dictionary['lower']) & (that <= aux_dictionary['upper'])    
    
    ipw_a = (data_est[outcome_name] / that)*(data_est[treatment_name]==1)
    ipw_b = (data_est[outcome_name] / (1-that))*(data_est[treatment_name]==0)
    
    ipw_a_att = ipw_a * that

    results = np.average(ipw_a[keep_these] - ipw_b[keep_these]), np.std(ipw_a[keep_these] - ipw_b[keep_these] ),        np.average( (ipw_a[keep_these] - ipw_b[keep_these])*that ),  np.std( (ipw_a[keep_these] - ipw_b[keep_these])*that ),        that
        
    return {'ATE TE':results[0], 'ATE SE': results[1], 'ATT TE':results[2], 'ATT SE': results[3], 'PScore':results[4]}


def ipw(data_est, 
                split_name, feature_name, outcome_name, treatment_name,
                ymodel,tmodel,
               n_data_splits,
               aux_dictionary):
    main_result = ipw_main(data_est, 
                split_name, feature_name, outcome_name, treatment_name,
                ymodel,tmodel,
               n_data_splits,
               aux_dictionary)
    ipw_bt_results = bootstrap.go_reps(aux_dictionary['bootstrapreps'], ipw_main, data_est, 
                    split_name, feature_name, outcome_name, treatment_name,
                    ymodel,tmodel,
                   n_data_splits,
                   aux_dictionary)
    return {'ATE TE':ipw_bt_results['ATE mean'], 'ATE SE': ipw_bt_results['ATE std'], 'ATT TE':ipw_bt_results['ATT mean'], 'ATT SE': ipw_bt_results['ATT std'], 'PScore': main_result['PScore']}


# In[10]:


'''
https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.WLS.html
weightsarray_like, optional

    A 1d array of weights. If you supply 1/W then the variables are pre- multiplied by 1/sqrt(W). 
    If no weights are supplied the default value is 1 and WLS results are the same as OLS.

'''
def ipw_wls(data_est, 
                split_name, feature_name, outcome_name, treatment_name,
                ymodel,tmodel,
               n_data_splits,
               aux_dictionary):
    ## Create and sort by splits
    data_est[split_name] = np.random.choice(n_data_splits, len(data_est), replace=True)
    data_est.sort_values(by=split_name, inplace=True)

    ## 1st Stage: Predict treatment indicator
    that = predict_treatment_indicator(data_est, split_name, n_data_splits, feature_name,treatment_name,tmodel)

    keep_these = (that >= aux_dictionary['lower']) & (that <= aux_dictionary['upper'])    
    
    ipw_weights = data_est[treatment_name] / that + (1-data_est[treatment_name]) / (1-that)
    
    wls = sm.WLS(data_est[outcome_name][keep_these],
           sm.add_constant(data_est[treatment_name])[keep_these],
           weights = ipw_weights[keep_these]
          ).fit()
    
    ipw_att_weights = data_est[treatment_name]  + (1-data_est[treatment_name])*that / (1-that)
    
    wls_att = sm.WLS(data_est[outcome_name][keep_these],
           sm.add_constant(data_est[treatment_name])[keep_these],
           weights = ipw_att_weights[keep_these]
          ).fit()
    

    return {'ATE TE':wls.params[1], 'ATE SE': wls.bse[1], 'ATT TE':wls_att.params[1], 'ATT SE': wls_att.bse[1], 'PScore':that}


# In[11]:


def dml_plm(data_est, 
                split_name, feature_name, outcome_name, treatment_name,
                ymodel,tmodel,
               n_data_splits,
               aux_dictionary):

    ## Create and sort by splits
    data_est[split_name] = np.random.choice(n_data_splits, len(data_est), replace=True)
    data_est.sort_values(by=split_name, inplace=True)

    ## 1st Stage: Predict treatment indicator
    that = predict_treatment_indicator(data_est, split_name, n_data_splits, feature_name,treatment_name,tmodel)
        
    ## Residualize treatment and outcome    
    outcome_hat = []        
    for r in np.arange(n_data_splits):
        train = (data_est[split_name] != r)
        test = (data_est[split_name]==r)
        ols = ymodel.fit(data_est[feature_name][train==True],data_est[outcome_name][train==True])
        prediction = ols.predict(data_est[feature_name][test==True])
        outcome_hat.extend(prediction)

    outcome_hat = np.array(outcome_hat)
    
    treatment_residual = data_est[treatment_name].to_numpy() - that
    outcome_residual = data_est[outcome_name].to_numpy() - outcome_hat
    
    ## shave off propensity scores below and under certain thresholds
    keep_these = (that >= aux_dictionary['lower']) & (that <= aux_dictionary['upper'])    
    ## Second stage OLS, for a partial linear model
    X = sm.add_constant(treatment_residual[keep_these])
    finalmodel_fit = sm.OLS( list(outcome_residual[keep_these]), X).fit()
    
    return {'ATE TE':finalmodel_fit.params[-1], 'ATE SE': finalmodel_fit.bse[-1],         'ATT TE':finalmodel_fit.params[-1], 'ATT SE': finalmodel_fit.bse[-1], 'PScore':that}


# In[18]:


def dml_irm(data_est, 
                split_name, feature_name, outcome_name, treatment_name,
                ymodel,tmodel,
               n_data_splits, aux_dictionary ):
    data_est[split_name] = np.random.choice(n_data_splits, len(data_est), replace=True)
    data_est = data_est.sort_values(by=split_name)

    ## 1st Stage: Predict treatment indicator
    that = predict_treatment_indicator(data_est, split_name, n_data_splits, feature_name,treatment_name,tmodel)
    
    ## 2nd Stage: Predict counterfactual outcomes
    yhat_treat, yhat_control = predict_counterfactual_outcomes(data_est, split_name, n_data_splits, feature_name, treatment_name, outcome_name,ymodel)
    
    ## Residualize:
    y_control_residual = data_est[outcome_name]- yhat_control
    y_treat_residual = data_est[outcome_name]- yhat_treat    
    
    #####
    ## ATE Estimator on the residuals
    #####
    ra_term = yhat_treat - yhat_control    

    first_fraction = (data_est[treatment_name]==1)*(y_treat_residual) / that
    second_fraction = (data_est[treatment_name]==0)*(y_control_residual) / (1-that)
    ipw_term = first_fraction - second_fraction
    
    treatment_estimate = ra_term + ipw_term
    keep_these = (that >= aux_dictionary['lower']) & (that <= aux_dictionary['upper'])    
        
    treatment_estimate = treatment_estimate[keep_these]
    
    ## standard error
    score = treatment_estimate - np.mean(treatment_estimate)
    j0 = 1 - 2*np.mean(score)+ np.mean(score**2)
    var_hat = j0**(-1) * np.mean(score**2)
    var_hat = np.std(score)
    
    ## ATE Standard error.
    score = treatment_estimate - np.mean(treatment_estimate)
    var_hat = 0   
    for r in np.arange(n_data_splits):
        here = score[data_est[split_name]==r]
        add_to = np.mean(here**2) * (1 / n_data_splits)
        var_hat += add_to
    SE = np.sqrt(var_hat) /np.sqrt(len(data_est))
    
    #####
    ## ATT Estimator on the residuals
    #####      
    prob_unconditional = (data_est[treatment_name].mean())
    att_first_fraction = (data_est[treatment_name]==1)*(y_control_residual)/prob_unconditional
    att_second_fraction = (that)*(data_est[treatment_name]==0)*(y_control_residual)/( that* (prob_unconditional) )

    treatment_estimate_att = att_first_fraction - att_second_fraction

    ## ATE Standard error.
    score = treatment_estimate_att - np.mean(treatment_estimate_att)
    var_hat = 0   
    for r in np.arange(n_data_splits):
        here = score[data_est[split_name]==r]
        add_to = np.mean(here**2) * (1 / n_data_splits)
        var_hat += add_to
    SE_ATT = np.sqrt(var_hat) /np.sqrt(len(data_est))
    
    return {'ATE TE':np.mean(treatment_estimate), 'ATE SE': SE,         'ATT TE':np.mean(treatment_estimate_att), 'ATT SE': SE_ATT, 'PScore':that}
    


