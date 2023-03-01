


import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import scipy.stats 
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, Lasso, LogisticRegressionCV, LogisticRegression


from matplotlib import gridspec
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os as os 
from IPython.display import display    

class coefstab:
    '''
    See CoefficientStability.ipynb for more details.
    
    inputs are:
    df             a Pandas dataframe
    W              string name of the treatment indicator
    y              string name of the outcome
    obs_list       list of the strings for the features we are controlling for.
    beta_hat       a float that represents the true treatment effect we are benchmarking against. The closer this number is to the 
                   estimated treatment effect, the more room you allow for bias. The further away, the higher the bar you give for bias.
    r_max          a float on range of (0,1] that represents the MAXIMUM r2 score you could attain. The higher this number is, the
                   more room you are giving yourself to allow for bias.
    te_model               [delta_ml only] the model used to estimate the propensity score
    te_model_dict_inputs   [delta ml only] dictionary of arguments used for the causal model
    y_model                [delta ml only] the model used to estimate the outcome variable
                    
    How do we use beta_max and r_max? These are user determines and represent how likely we think it is that we have a certain amount of bias. A sample statement follows. Suppose that we set beta_max=2 and r_max= 0.99, and calculated a delta of 6.
    Given the covariates we control for in obs_list, the unobserved covariates needed to increase the R-squared to 0.99 would have to be 6 times more important to eliminating bias so that our estimated treatment becomes 2.
    
    '''
    
    
    def delta_ols(df, W, y, obs_list, beta_hat, R_max):    
        random_partitions = 2
        df['splits'] = np.random.choice(random_partitions, len(df), replace=True)
        df.sort_values(by=['splits'], inplace=True)
        beta_dot = 0
        R_dot = 0
        beta_tilde = 0
        R_tilde= 0

        W_hat = []
        for r in range(random_partitions):
            train = df.loc[df['splits']!=r]
            test = df.loc[df['splits']==r]
            ## 1. Estimate regression on just W. Remember to use cross-fitting to get the correct R2
            model_on_train = sm.OLS(train[y],sm.add_constant(train[W]) ).fit()        
            model_predict_test = model_on_train.predict(sm.add_constant(test[W]) )        
            beta_dot += model_on_train.params[1] / random_partitions
            R_dot += np.sqrt(np.corrcoef( model_predict_test, test[y])[0,1]) / random_partitions

            ## 2. Estimate regression on W and observed. Remember to use cross-fitting to get the correct R2
            model_on_train = sm.OLS(train[y],sm.add_constant(train[[W] + obs_list]) ).fit()        
            model_predict_test = model_on_train.predict(sm.add_constant(test[[W] + obs_list]) )        
            beta_tilde += model_on_train.params[1] / random_partitions
            R_tilde += np.sqrt(np.corrcoef( model_predict_test, test[y])[0,1]) / random_partitions

            ## 3. calculate the expectation of W given the observed features
            model_on_train = sm.OLS(train[W],sm.add_constant(train[obs_list]) ).fit()        
            model_predict_test = model_on_train.predict(sm.add_constant(test[obs_list]) )        
            W_hat.extend(model_predict_test.to_list())

        tau_x = np.var(np.array(df[W]) - W_hat)
        sigma_x = np.var(df[W])
        sigma_y = np.var(df[y])
        A = (beta_tilde - beta_hat)**2 * (tau_x *(beta_dot - beta_tilde)**2 * sigma_x) +\
            (beta_tilde-beta_hat)**3*(tau_x*sigma_x - tau_x**2)

        numerator = (beta_tilde-beta_hat)*(R_tilde - R_dot)*sigma_y*tau_x + (beta_tilde - beta_hat)*sigma_x*tau_x*(beta_dot - beta_tilde)**2
        numerator += 2*A

        denominator = (R_max-R_tilde)*sigma_y*(beta_dot - beta_tilde)*sigma_x + (beta_tilde-beta_hat)*(R_max - R_tilde)*sigma_y*(sigma_x - tau_x)
        denominator +=A
        return numerator / denominator

    def delta_ml(df, W, y, 
                       beta_hat, R_max,
                      te_model, te_model_dict_inputs,
                    y_model):    

        beta_dot = 0
        R_dot = 0
        beta_tilde = 0
        R_tilde= 0

        ## 1. Estimate regression on just W. 
        step1 = sm.OLS(df[y],sm.add_constant(df[W]) ).fit()        
        beta_dot = step1.params[1] 
        R_dot = step1.rsquared

        ## 2. Estimate regression on W and observed. Remember to use cross-fitting to get the correct R2    
        step2 = te_r2_output(te_model, te_model_dict_inputs,
                    y_model)    
        beta_tilde = step2[0]
        R_tilde = step2[2]

        ## 3. calculate the expectation of W given the observed features
        random_partitions = 2
        df['splits'] = np.random.choice(random_partitions, len(df), replace=True)
        df.sort_values(by=['splits'], inplace=True)    
        W_hat = []
        for r in range(random_partitions):
            train = df.loc[df['splits']!=r]
            test = df.loc[df['splits']==r]

            model_on_train = te_model_dict_inputs['tmodel'].fit(sm.add_constant(train[te_model_dict_inputs['feature_name']]), 
                                                                train[W] )
            model_predict_test = model_on_train.predict_proba(sm.add_constant(test[te_model_dict_inputs['feature_name']]) )[:,1]
            W_hat.extend(model_predict_test)


        tau_x = np.var(np.array(df[W]) - W_hat)
        sigma_x = np.var(df[W])
        sigma_y = np.var(df[y])
        A = (beta_tilde - beta_hat)**2 * (tau_x *(beta_dot - beta_tilde)**2 * sigma_x) +\
            (beta_tilde-beta_hat)**3*(tau_x*sigma_x - tau_x**2)

        numerator = (beta_tilde-beta_hat)*(R_tilde - R_dot)*sigma_y*tau_x + (beta_tilde - beta_hat)*sigma_x*tau_x*(beta_dot - beta_tilde)**2
        numerator += 2*A

        denominator = (R_max-R_tilde)*sigma_y*(beta_dot - beta_tilde)*sigma_x + (beta_tilde-beta_hat)*(R_max - R_tilde)*sigma_y*(sigma_x - tau_x)
        denominator +=A
        return numerator / denominator

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


# In[ ]:





# In[4]:


def predict_cv(dataset, split_name, n_data_splits, feature, x, model):
    y_hat = []
    for r in np.arange(n_data_splits):
        train = (dataset[split_name] != r)
        test = (dataset[split_name]==r)
        lg = model.fit(dataset[feature][train==True],dataset[x][train==True])    
        prediction = lg.predict(dataset[feature][test==True])
        y_hat.extend(prediction)
    return np.array(y_hat)


# In[11]:


import scipy.stats 
import statsmodels.api as sm

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
    



