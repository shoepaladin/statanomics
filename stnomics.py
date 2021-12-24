#!/usr/bin/env python
# coding: utf-8

# In[20]:


'''
This library has the following classes:

- diagnostics.selection
- diagnostics.balance
- predQC
- bootstrap
- ate
- ate.dml

'''
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import scipy.stats 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import metrics


from matplotlib import gridspec
import matplotlib.pyplot as plt

import os as os 


# In[21]:


## Standard functions

def predict_cv(dataset, split_name, n_data_splits, feature, x, model):
    y_hat = []
    for r in np.arange(n_data_splits):
        train = (dataset[split_name] != r)
        test = (dataset[split_name]==r)
        lg = model.fit(dataset[feature][train==True],dataset[x][train==True])    
        prediction = lg.predict(dataset[feature][test==True])
        y_hat.extend(prediction)
    return np.array(y_hat)



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
 
    
    
    


# In[ ]:




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

    def rmse(truth, estimate):
        return     np.sqrt(np.sum( (truth-estimate)**2) / (len(truth)))

    def mae(truth, estimate):
        return np.sum( np.abs(truth-estimate)) / (len(truth))

    def mape(truth,estimate):
        return np.average( np.abs(truth - estimate)/truth ) 

    def r2(truth,estimate):
        return np.sqrt(  np.corrcoef( truth, estimate)[0,1]  )


# In[ ]:



class secondstage:
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

    


# In[ ]:




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


# In[22]:


class diagnostics:
    
    class selection:

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
            selection_coeff_names = pd.DataFrame()
            for n,y in zip( ['outcome','treatment'], [outcome_name, treatment_name]):        
                if n=='outcome':
                    lasso_selection = LassoCV(cv=5, random_state=27, normalize=True, n_jobs=-1).fit(data_est[feature_name], data_est[y]) 
                    lasso_fit = Lasso(alpha=lasso_selection.alpha_, max_iter=200000).fit(data_est[feature_name], data_est[y]) 
                else:
                    lasso_selection = LogisticRegressionCV(cv=5, random_state=27, penalty='l2', n_jobs=-1).fit(data_est[feature_name], data_est[y]) 
                    lasso_fit = LogisticRegression(C=lasso_selection.C_[0], penalty='l2', max_iter=200000).fit(data_est[feature_name], data_est[y])             
                entry = pd.DataFrame(data={'type':n, 'features':feature_name, 'coef':lasso_fit.coef_.flatten()})
                selection_coeff_names = selection_coeff_names.append(entry)
                for x,b in zip(data_est[feature_name].columns, lasso_fit.coef_.flatten()):
                    if (b != 0) & (x!='const') & (x !='t'):
                        selected_lasso_features[n].append(x)
                    else:
                        pass
            unique_entries = list(selected_lasso_features.values())
            return unique_entries, selected_lasso_features, selection_coeff_names


    class balance:

        def propensity_overlap(df, pscore, treatment):
            '''
            df         (DataFrame) dataframe name
            pscore     (String)    name of propensity score in dataframe
            treatment. (String)    name of treatment indicator in dataframe
            '''
            control = df.loc[df[treatment]==0][pscore]
            treatment = df.loc[df[treatment]==1][pscore]
            ## Plot propensity score
            fig, ax = plt.subplots(nrows=2,ncols=1, figsize=(10,6), sharey=False, sharex=False)
            t = ax[0].hist(treatment, density=True, bins=100, alpha=0.25, color='orange', label='T')
            c = ax[0].hist(control, density=True, bins=100, alpha=0.25, color='blue', label='C')
            ax[0].set_xlabel('Propensity Score')
            ax[0].set_ylabel('Density')
            ax[0].grid(True)
            ax[0].legend()
            ax[0].set_title("Distribution of Propensity Scores")

            t = ax[1].hist(1/ treatment, density=True, bins=100, alpha=0.25, color='coral', label='T')
            c = ax[1].hist(1/ (1-control), density=True, bins=100, alpha=0.25, color='royalblue', label='C')
            ax[1].set_xlabel('Inverse Propensity Weights')
            ax[1].set_ylabel('Density')
            ax[1].grid(True)
            ax[1].legend()
            ax[1].set_title("Distribution of Inverse Propensity Weights")

            plt.show()

            for below in [0.01,0.05,0.10]:        
                control_below =  np.sum( (control < below) )
                treatment_below =  np.sum( (treatment < below) )    
                print('   Control N below {0:5.2f}: {1}'.format(below, control_below))
                print(' Treatment N below {0:5.2f}: {1}\n'.format(below, treatment_below))        

            for above in [0.90, 0.95, 0.99]:
                control_below =  np.sum( (control > above) )
                treatment_below =  np.sum( (treatment > above) )    
                print('   Control N above {0:5.2f}: {1}'.format(above, control_below))
                print(' Treatment N above {0:5.2f}: {1}\n'.format(above, treatment_below))        


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


# In[23]:



    
class ate:
    

    def propbinning(data_est, 
                    split_name, feature_name, outcome_name, treatment_name,
                    ymodel,tmodel,
                   n_data_splits,
                   aux_dictionary):
        main_result = ate.propbinning_main(data_est, 
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

        block_splits(data_est, split_name, n_data_splits)

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
        block_splits(data_est, split_name, n_data_splits)

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
        main_result = ate.ipw_main(data_est, 
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


    # In[36]:


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
        block_splits(data_est, split_name, n_data_splits)

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
    
    
    class dml:

        def dml_plm(data_est, 
                        split_name, feature_name, outcome_name, treatment_name,
                        ymodel,tmodel,
                       n_data_splits,
                       aux_dictionary):

            block_splits(data_est, split_name, n_data_splits)

            ## 1st Stage: Predict treatment indicator
            if ('that' in aux_dictionary.keys() ):
                if (aux_dictionary['that'] != None).any():
                    that = aux_dictionary['that'][:]
            else:
                that = predict_treatment_indicator(data_est, split_name, n_data_splits, feature_name,treatment_name,tmodel)

            ## Residualize outcome
            if ('yhat' in aux_dictionary.keys() ):
                if (aux_dictionary['yhat'] != None).any():
                    outcome_hat = aux_dictionary['yhat'][:]

            else:
                ## Create and sort by splits
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


        # In[39]:


        def dml_irm(data_est, 
                        split_name, feature_name, outcome_name, treatment_name,
                        ymodel,tmodel,
                       n_data_splits, aux_dictionary ):
            block_splits(data_est, split_name, n_data_splits)


            ## 1st Stage: Predict treatment indicator
            if ('that' in aux_dictionary.keys() ):
                if (aux_dictionary['that'] != None):
                    that = aux_dictionary['that'][:]
            else:
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


# In[24]:



class hte:
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
            that, t_r2 = predict_treatment_indicator(data_est, 'splits', n_data_splits, feature_name,treatment_name, tmodel)

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
            that, t_r2 = predict_treatment_indicator(data_est, 'splits', n_data_splits, feature_name,treatment_name, tmodel)
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


            y_r2 = predQC.r2(data_est[outcome_name], outcome_hat)
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
                treatment_estimate,se_estimate, coef_pd = second_stage.second_stage(approach, test_data, train_data, covar_list, het_feature )
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
            that, t_r2 = predict_treatment_indicator(data_est, 'splits', n_data_splits, feature_name,treatment_name, tmodel)

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
                treatment_estimate,se_estimate, coef_pd = second_stage.second_stage(approach, test_data, train_data, covar_list, het_feature )
                output_treatment_estimate.extend(list(treatment_estimate))            
                output_se_hat.extend(list(se_estimate))            
                ols_coef_pd[test_i] = coef_pd.copy()

            other_output = {'coefficients':ols_coef_pd, 
                            'Treatment outcome metric':t_r2}
            ## Output the treatment estimate and propensity scores
            return output_treatment_estimate, output_se_hat, that, output_baseline_hat, other_output

