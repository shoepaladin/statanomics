#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os as os
print(os.getcwd())
print(os.listdir(os.getcwd()))

# # HR Model
# 

# In[407]:


import pandas as pd

import numpy as np
import os as os 

import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LassoCV

import warnings


# In[ ]:


def r2(truth,estimate):
    return np.sqrt(  np.corrcoef( truth, estimate)[0,1]  )


# In[410]:


def predict_treatment_indicator(dataset, split_name, n_data_splits, feature,treatment, model):
    '''
    This function estimates the treatment indicator in a data-splitting fashion.

    dataset         (dataframe)   dataframe
    split_name      (str)         name of feature that denotes split categories
    n_data_splits   (int)         number of unique values in dataset[split_name]
    feature.        (list)        list of feature names to predict treatment
    treatment       (str)         name of treatment indicator 
    model           (function)    model used to predict treatment
    '''
    treatment_hat = []
    for r in np.arange(n_data_splits):
        train = (dataset[split_name] != r)
        test = (dataset[split_name]==r)
        lg = model.fit(dataset[feature][train==True],dataset[treatment][train==True])
        prediction = lg.predict_proba(dataset[feature][test==True])[:,1]
        treatment_hat.extend(prediction)
    return np.array(treatment_hat)


# In[413]:


def second_stage(approach, test_data, train_data, covar_list, het_feature ):
    '''
    Outputs treatment estimates, standard errors, and OLS coefficient results    
    
    approach      (str)         denotes whether the second stage is an OLS, CVLasso, or Lasso
    test_data     (dataframe)   dataframe of test data we get estimates for
    train_data    (dataframe)   dataframe of training data we use the CVLasso/Lasso on for selection. In the case of OLS, we use the train data to estimate the OLS coefficients.
    covar_list    (list)        list of features used for prediction/control
    het_feature   (list)        list of features we use to estimate HTE.
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
        lasso_fit = Lasso(alpha=lasso_selection.alpha_, max_iter=20000).fit(X, train_data['y']) 
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
        lasso_fit = Lasso(alpha=5, max_iter=200000).fit(X, train_data['y'])            
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


# In[1]:


def het_ols(data_est, 
                    number_of_splits,
                    feature_name, outcome_name, treatment_name,
                    het_feature):
    '''
    Just run an OLS with data-splitting.
    
    data_est        (obj) name of the PanDas dataframe 
    number_of_splits      (int) number of splits
    feature_name    (list) list of features used for predictiong treatment and outcome
    outcome_name    (str) name of outcome feature
    treatment_name  (str) name of treatment feature
    het_feature     (list) list of features used for driving heterogeneity
    '''    
    output_treatment_estimate = []
    output_se_hat = []
    that = []
    output_baseline_hat = []
    
    ## Create and sort by splits
    data_est['splits'] = np.random.choice(number_of_splits, len(data_est), replace=True)
    data_est.sort_values(by='splits', inplace=True)

    ## Create interactions of each feature with treatment.
    het_interactions = []
    for x in het_feature:
        data_est[x+'_x'] = data_est[treatment_name]*data_est[x]
        het_interactions.append(x + '_x')


    ## Run fully interacted regression
    for r in np.arange(number_of_splits):
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
            if x==treatment_name:
                treatment_estimates += treatment_effects[x]
                het_treatment_effects.append(treatment_effects[x])
            elif x in het_interactions:
                treatment_estimates += finalmodel_fit.params[x]*test[x.replace('_x','')]
                het_treatment_effects.append(treatment_effects[x])
            else:
                pass

        ## Estimate standard errors on the test dataset
        var_cov = np.array(  finalmodel_fit.cov_params()  )[ -1 * len(het_interactions) - 1:, -1 * len(het_interactions) - 1:]
        XX = np.dot( test[[treatment_name] + het_interactions], var_cov)
        output_se = np.sqrt( np.abs(np.dot( XX, np.array(het_treatment_effects).astype(float) ) ))
                
        ## Estimate baseline
        output_baseline = test[outcome_name] - test[treatment_name]*treatment_estimates
        output_treatment_estimate.extend(treatment_estimates.tolist())
        output_se_hat.extend(output_se.tolist())
        output_baseline_hat.extend(output_baseline.tolist())
    
    data_est.drop(labels=['splits'], axis=1, inplace=True)            
        
    return output_treatment_estimate, output_se_hat, that, output_baseline_hat


# In[1]:


class het_dml_approaches:
    ## Approach 1
    def residualize_D(data_est, 
                    number_of_splits,
                      feature_name, outcome_name, treatment_name,
                    het_feature,
                    ymodel,tmodel,
                     force_second_stage):
        '''    
        data_est        (obj) name of the PanDas dataframe 
        number_of_splits      (int) number of splits
        feature_name    (list) list of features used for predictiong treatment and outcome
        outcome_name    (str) name of outcome feature
        treatment_name  (str) name of treatment feature
        het_feature     (list) list of features used for driving heterogeneity
        ymodel          (obj) model used for predicting outcome
        tmodel          (obj) model used for predicting treatment
        force_second_stage    (str) {None, 'OLS','CVLasso', 'Lasso'}
        '''            
        ## Create and sort by splits
        data_est['splits'] = np.random.choice(number_of_splits, len(data_est), replace=True)
        data_est.sort_values(by='splits', inplace=True)
        
        ## 1st Stage: Predict treatment indicator
        that = predict_treatment_indicator(data_est, 'splits', number_of_splits, feature_name,treatment_name, tmodel)


        ## 1st Stage: Predict treatment indicator interacted with each feature
        ## remember to remove the feature interacted with the treatment as a predictor
        residual_treatment_hat = [data_est[treatment_name] - that]
        for x in het_feature:
            data_est[x+'_x'] = data_est[treatment_name]*data_est[x]
            treatment_hat = []
            for r in np.arange(number_of_splits):
                train = (data_est['splits'] != r)
                test = (data_est['splits']==r)
                ols_pred = ymodel.fit(data_est[feature_name][train==True],
                                                data_est[x+'_x'][train==True])
                prediction = ymodel.predict(data_est[feature_name][test==True])
                treatment_hat.extend(prediction)            

            diff = data_est[x+'_x'] - treatment_hat 
            residual_treatment_hat.append(diff)        

        ## 1st Stage: Predict the outcome
        ## Train an ols model on a training set, and predict onto the test set.
        outcome_hat = []
        baseline_hat = []
        for r in np.arange(number_of_splits):
            train = (data_est['splits'] != r)
            test = (data_est['splits']==r)
            ols_control=ymodel.fit(data_est[feature_name ][(train==True)],
                                   data_est[outcome_name][(train==True)]) 
            prediction = ols_control.predict(data_est[feature_name ][(test==True)])
            outcome_hat.extend(prediction)

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

        output_baseline_hat = []
        output_treatment_estimate = []
        output_se_hat = []
        
        '''
        If the dimensionality of teratment array relative to sample size is less than 20%, then just do an OLS.
        But if user is forcing OLS, CVLasso, or Lasso, then let them.
        '''        
        if force_second_stage == None:
            dim_D = len(het_feature)+1
            if (float(dim_D/len(data_est)) <= 0.20):
                approach='OLS'
            else:
                approach='CVLasso'
        else:
            approach = force_second_stage

        ols_coef_pd = {}
        for test_i,train_i in zip(['0','1'], ['1','0']):            
            test_data = data_est_half[test_i]
            train_data = data_est_half[train_i]
            treatment_estimate,se_estimate, coef_pd = second_stage(approach, test_data, train_data, covar_list, het_feature )
            output_treatment_estimate.extend(list(treatment_estimate))            
            output_se_hat.extend(list(se_estimate))            
            ols_coef_pd[test_i] = coef_pd.copy()
            
        data_est.drop(labels=['splits'], axis=1, inplace=True)            
        ## Output the treatment estimate and propensity scores
        return output_treatment_estimate, output_se_hat, ols_coef_pd, that, output_baseline_hat


    ## Approach 2
    def residualize_P(data_est, 
                    number_of_splits,
                      feature_name, outcome_name, treatment_name,
                    het_feature,
                    ymodel,tmodel,
                     force_second_stage):        
        
        '''    
        data_est        (obj) name of the PanDas dataframe 
        number_of_splits      (int) number of splits
        feature_name    (list) list of features used for predictiong treatment and outcome
        outcome_name    (str) name of outcome feature
        treatment_name  (str) name of treatment feature
        het_feature     (list) list of features used for driving heterogeneity
        ymodel          (obj) model used for predicting outcome
        tmodel          (obj) model used for predicting treatment
        force_second_stage    (str) {None, 'OLS','CVLasso', 'Lasso'}
        '''
        ## Create and sort by splits
        data_est['splits'] = np.random.choice(number_of_splits, len(data_est), replace=True)
        data_est.sort_values(by='splits', inplace=True)
        
        ## 1st Stage: Predict treatment indicator
        that = predict_treatment_indicator(data_est, 'splits', number_of_splits, feature_name,treatment_name, tmodel)


        ## 1st Stage: Residualize Treatment Indicator
        residual_treatment_hat = data_est[treatment_name] - that 

        ## 1st Stage: Predict the outcome
        outcome_hat = []
        baseline_hat = []
        for r in np.arange(number_of_splits):
            train = (data_est['splits'] != r)
            test = (data_est['splits']==r)
            ols_control=ymodel.fit(data_est[feature_name ][(train==True)],
                                   data_est[outcome_name][(train==True)]) 
            prediction = ols_control.predict(data_est[feature_name ][(test==True)])
            outcome_hat.extend(prediction)

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

        output_baseline_hat = []
        output_treatment_estimate = []
        output_se_hat = []
        '''
        If the dimensionality of teratment array relative to sample size is less than 20%, then just do an OLS.
        But if user is forcing OLS, CVLasso, or Lasso, then let them.
        '''        
        if force_second_stage == None:
            dim_D = len(het_feature)+1
            if (float(dim_D/len(data_est)) <= 0.20):
                approach='OLS'
            else:
                approach='CVLasso'
        else:
            approach = force_second_stage
        ols_coef_pd = {}

        for test_i,train_i in zip(['0','1'], ['1','0']):            
            test_data = data_est_half[test_i]
            train_data = data_est_half[train_i]
            treatment_estimate,se_estimate, coef_pd = second_stage(approach, test_data, train_data, covar_list, het_feature )
            output_treatment_estimate.extend(list(treatment_estimate))            
            output_se_hat.extend(list(se_estimate))            
            ols_coef_pd[test_i] = coef_pd.copy()
            
            
        data_est.drop(labels=['splits'], axis=1, inplace=True)
        ## Output the treatment estimate and propensity scores
        return output_treatment_estimate, output_se_hat, ols_coef_pd,that, output_baseline_hat


# In[417]:


def hdm_selection(data_est, 
                    feature_name, outcome_name, treatment_name):
    '''    
    data_set         (obj)    dataframe
    feature_name     (list)   list of features to choose from
    outcome_name     (str)    name of outcome in data_set
    treatment_name   (str)    name of treatment in data_set
    
    Run a cross-validated lasso regressions on the outcome and treatment to 
    select features that predict either or both.
    '''    
    selected_lasso_features = {}
    selected_lasso_features['treatment'] = []
    selected_lasso_features['outcome'] = []    
    selected_lasso_df = {}
    for n,y in zip( ['outcome','treatment'], [outcome_name, treatment_name]):        
        lasso_selection = LassoCV(cv=5, random_state=27, normalize=True, n_jobs=-1).fit(data_est[feature_name], data_est[y]) 
        lasso_fit = Lasso(alpha=lasso_selection.alpha_, max_iter=200000).fit(data_est[feature_name], data_est[y]) 
        
        output_df = pd.DataFrame()
        for x,b in zip(data_est[feature_name].columns, lasso_fit.coef_):
            row = pd.DataFrame(index=[x], data={'coef_'+n:b})
            output_df = output_df.append(row)
            if (b != 0) & (x!='const') & (x !='t'):
                selected_lasso_features[n].append(x)
            else:
                pass
        selected_lasso_df[n] = output_df
    selected_lasso_features['df'] = selected_lasso_df['outcome'].merge(selected_lasso_df['treatment'])
    unique_entries = list(selected_lasso_features.values())
    return unique_entries, selected_lasso_features
        

    


# In[1]:


import scipy.stats 
import statsmodels.api as sm
def predict_cv(dataset, split_name, n_data_splits, feature, x, model):
    y_hat = []
    for r in np.arange(n_data_splits):
        train = (dataset[split_name] != r)
        test = (dataset[split_name]==r)
        lg = model.fit(dataset[feature][train==True],dataset[x][train==True])    
        prediction = lg.predict(dataset[feature][test==True])
        y_hat.extend(prediction)
    return np.array(y_hat)

class hist_valid:
    def feature_balance(df,
                        feature_list, pscore, treatment,
                       dml_ols, dml_model):
        '''
        df            (DataFrame) dataframe name
        feature_list  (list)      list of feature names to check balance for.
        pscore        (String)    name of propensity score in dataframe
        treatment.    (String)    name of treatment indicator in dataframe
        dml_ols.      (String)    {'DML','OLS'}
        dml_model     (model)     Model to fit the feature.
        '''    
        control_data = df.loc[df[treatment]==0]
        treatment_data = df.loc[df[treatment]==1]
        ## Initialize the result dataframe
        result_df = pd.DataFrame()
        if dml_ols.upper()=='DML':
            df['splits'] = np.random.choice(5,len(df), replace=True)
        else:
            pass

        ## Loop through each of the features
        for x,i in zip(feature_list, range(len(feature_list))):
            t_raw = treatment_data[x]
            c_raw = control_data[x]

            ## Remove the feature iterating over from the full list
            feature_list_no_x = feature_list[:]
            feature_list_no_x.remove(x)

            ## Calculate simple differences between treatment and control
            diff = np.average(t_raw) - np.average(c_raw)
            diff_tstat, diff_pvalue = scipy.stats.ttest_ind(t_raw, c_raw)

            ## Calculate the normalized difference
            normalized_diff = diff / np.sqrt( np.var(t_raw) + np.var(c_raw)  )

            ## Difference controlling for propensity score              

            if dml_ols.upper()=='OLS':
                X = sm.add_constant( df[ [treatment] + [pscore]] ).astype(float)            
                OLS_pscore = sm.OLS( df[x] , X ).fit()
                ols_diff, ols_pvalue, ols_tstat = OLS_pscore.params[treatment], OLS_pscore.pvalues[treatment], OLS_pscore.tvalues[treatment]
            elif dml_ols.upper()=='DML':
                X = sm.add_constant( df[treatment] - df[pscore] ).astype(float)            
                x_hat =  predict_cv(df, 'splits', 5, feature_list_no_x, x, dml_model)
                OLS_pscore = sm.OLS( df[x] - x_hat , X ).fit()
                ols_diff, ols_pvalue, ols_tstat = OLS_pscore.params.values[-1], OLS_pscore.pvalues.values[-1], OLS_pscore.tvalues.values[-1]


            ## Calculate raw standardized difference
            diff_sd = diff / df[x].std()

            diff_tstat_sd, diff_pvalue_sd = scipy.stats.ttest_ind(t_raw/df[x].std(), c_raw/df[x].std())


            ## SD Difference controlling for propensity score              
            df['x_std'] = (df[x] - df[x].mean() )/df[x].std()
            if dml_ols.upper()=='OLS':
                X = sm.add_constant( df[ [treatment] + [pscore]] ).astype(float)            
                OLS_pscore = sm.OLS( df['x_std'] , X ).fit()
                ols_diff_sd, ols_pvalue_sd, ols_tstat_sd = OLS_pscore.params[treatment], OLS_pscore.pvalues[treatment], OLS_pscore.tvalues[treatment]  
            elif dml_ols.upper()=='DML':
                X = sm.add_constant( df[treatment] - df[pscore] ).astype(float)
                x_hat =  predict_cv(df, 'splits', 5, feature_list_no_x, 'x_std', dml_model)
                OLS_pscore = sm.OLS( df['x_std'] - x_hat , X ).fit()
                ols_diff_sd, ols_pvalue_sd, ols_tstat_sd = OLS_pscore.params.values[-1], OLS_pscore.pvalues.values[-1], OLS_pscore.tvalues.values[-1]  



            row = pd.DataFrame(index=[i], data={'feature':x, 
                                                'Raw Difference':diff, 
                                                'Raw PValue': diff_pvalue, 
                                                'Raw TStat': diff_tstat,
                                                'Normalized Diff': normalized_diff, 
                                                'OLS-PScore Difference': ols_diff, 
                                                'OLS-PScore PValue': ols_pvalue, 
                                                'OLS-PScore TStat': ols_tstat,
                                                'Raw Difference SD':diff_sd, 
                                                'Raw PValue SD': diff_pvalue_sd, 
                                                'Raw TStat SD': diff_tstat_sd,
                                                'OLS-PScore Difference SD': ols_diff_sd, 
                                                'OLS-PScore PValue SD': ols_pvalue_sd, 
                                                'OLS-PScore TStat SD': ols_tstat_sd                                            
                                               })
            result_df = result_df.append(row)
        result_df.sort_values(by='feature', inplace=True)

        try:
            df.drop(columns=['x_std'], inplace=True)        
            df.drop(columns=['x_std'], inplace=True)
        except:
            pass

        return result_df

    def summarize_result_df(result_df):

        all_differences = result_df[['Raw Difference SD','OLS-PScore PValue SD']].abs().describe()
        stat_sig_raw_differences = result_df.loc[ result_df['Raw PValue SD'] < 0.05]['Raw Difference SD'].abs().describe()
        a = stat_sig_raw_differences.to_frame()
        a.rename(columns={"Raw Difference SD": "Stat Sig Raw Difference SD"}, inplace=True)

        stat_sig_raw_differences = result_df.loc[ result_df['OLS-PScore PValue SD'] < 0.05]['OLS-PScore Difference SD'].abs().describe()
        b = stat_sig_raw_differences.to_frame()
        b.rename(columns={"OLS-PScore Difference SD": "Stat Sig OLS-PScore Difference SD"}, inplace=True)

        return pd.concat([all_differences, a, b], axis=1)      
    
    def plot_difference(results_df, figuresize):
        fig, ax = plt.subplots(figsize=figuresize)  # Create the figure    
        y_pos = range( len(results_df) )
        values = results_df['feature']
        diff_sd =  results_df['Raw Difference SD'].abs() 

        diff_ols_sd =  results_df['OLS-PScore Difference SD'].abs() 

        diff_sd_se = results_df['Raw Difference SD'].abs() / results_df['Raw TStat SD'].abs()
        diff_ols_se = results_df['OLS-PScore Difference SD'].abs() / results_df['OLS-PScore TStat SD'].abs()    

        diff_sd_ci95_upper,diff_sd_ci95_lower = diff_sd + 1.96*diff_sd_se , diff_sd - 1.96*diff_sd_se
        diff_ols_ci95_upper,diff_ols_ci95_lower = diff_ols_sd + 1.96*diff_ols_se , diff_ols_sd - 1.96*diff_ols_se    

        rects = ax.barh( values, diff_sd , color='blue',alpha=0.25, label='Raw')    
        hlines= ax.hlines( values, diff_sd_ci95_lower, diff_sd_ci95_upper,
                          colors='blue', linestyles='solid', label='95% CI')


        rects = ax.barh( values, diff_ols_sd , color='red', alpha=0.25, label='Controlled')
        hlines= ax.hlines( values, diff_ols_ci95_lower, diff_ols_ci95_upper,
                          colors='red', linestyles='solid', label='95% CI')

        ax.legend()
        ax.set_xlabel('Abs of Standardized Difference')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(values)
        ax.invert_yaxis()  # labels read top-to-bottom


# In[ ]:





# In[ ]:




