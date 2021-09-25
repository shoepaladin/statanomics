# Diagnostics for Feature Selection and Assessing Balance
## Documentation
22 Aug 2021

**selection**.hdm_selection(*data_est, 
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


**balance**.propensity_overlap(*df,
                    pscore, treatment*)

This simply plots:
1. propensity scores for treatment and control groups.
2. inverse propensity weights for treatment and control groups.

| Input | Type | Description |
| --- | --- | ---|
|**df**  | (DataFrame) | dataframe name
|**pscore**  | (String)    | name of propensity score in dataframe
|**treatment**  | (String)    | name of treatment indicator in dataframe



**balance**.feature_balance(*df,
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



**balance**.plot_difference(*results_df, figure_size*)

This uses the outputed dataframe from **feature_balance** to create a figure similar to the _cobalt_ R-package. 


| Input | Type | Description |
| --- | --- | ---|
|**df**  | (DataFrame) | dataframe name
|**figure_size**  | (tuple)   | two-element tuple that tells the figure size




**balance**.summarize_result_df(*results_df*)

This uses the outputted dataframe from the function __feature_balance__ and shows the distribution of raw differences, OLS differences, and distributions of statistically significant differences.

| Input | Type | Description |
| --- | --- | ---|
|**df**  | (DataFrame) | dataframe name





