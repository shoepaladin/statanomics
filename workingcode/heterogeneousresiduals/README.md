
Version: 29 April 2021

This contains the package for using the Heterogeneous Residuals model. The Heterogeneous Residuals model builds on Semenova, Goldman, Chernozhukov, and Taddy (2021) - https://arxiv.org/abs/1712.09988 - by residualzing the entire matrix of heterogeneous treatments, rather than residualizing only the treatment variable. This means it reframe heterogeneous treatment effects as a multiple treatments problem.

 
 HRmodel.het_dml_approaches.**residualize_D**(*data_est, number_of_splits, feature_name, outcome_name, treatment_name, het_feature, ymodel, tmodel, force_second_stage*)

The Heterogeneous Residuals model that residualizes the entire treatment matrix, treating heterogeneous treatment effects as a multiple treatments problem.

| Input | Type | Description |
| --- | --- | ---|
| **data_est** | object | name of the PanDas dataframe  |
| **number_of_splits** | int | number of splits  |
| **feature_name** | list | list of features used for predictiong treatment and outcome  |
| **outcome_name** | str | name of outcome feature  |
| **treatment_name** | str | name of treatment feature  |
| **het_feature** | list | list of features used for driving heterogeneity. Note this can be different from **feature_name**. If user's objective is to estimate as fine-grained as possible, then the recommendation is to set **het_feature**==**feature_name**  |
| **ymodel** | object | model used for predicting outcome and the treatment matrix. This should be a continuous predictor, unless the treatment matrix is all indictors.  |
| **tmodel** | object | model used for predicting the treatment indicator.  |
| **force_second_stage** | str | {None, 'OLS','CVLasso', 'Lasso'} : this allows the user to set what the second stage regression is for choosing features. OLS is always used for second stage estimation and inference. If *None* is selected, it automatically uses *CVLasso* if $dim(D)/N >0.20$. This selection is done in the function *HRmodel.hdm_selection* |

| Output | Type | Description |
| --- | --- | --- |
| **output_treatment_estimate** |  array | Array of point HTE estimates for each observation in the dataset |
| **output_se_hat** | array | Array of standard errors for each observation in the dataset |
| **ols_coef_pd** | dictionary | Dictionary of OLS output from the second stage results. The dictionary has the structure: 
|   |  | { '0': indicates the OLS results are for the first randomly chosen half of the dataset 
|   |  |     {index: OLS feature names for heterogeneity,
|   |  |     'coef': OLS coefficients,
|   |  |     'se': OLS standard errors,
|   |  |     'pvalue': OLS pvalues,
|   |  |     'N': sample size
|   |  |     },
|   |  | { '1': indicates the OLS results are for the second randomly chosen half of the dataset 
|   |  |     {index: OLS feature names for heterogeneity,
|   |  |     'coef': OLS coefficients,
|   |  |     'se': OLS standard errors,
|   |  |     'pvalue': OLS pvalues,
|   |  |     'N': sample size
|   |  |     }}
| **that** | array | propensity score for the treatment indicator | 
| **output_baseline_hat** |  array | Array of predicted control outcomes for each observation in the dataset |



HRmodel.het_dml_approaches.**residualize_P**(*data_est, number_of_splits, feature_name, outcome_name, treatment_name, het_feature, ymodel, tmodel, force_second_stage*)

An implementation of Semenova, Goldman, Chernozhukov, and Taddy (2021) - https://arxiv.org/abs/1712.09988. This function exists because it was used as a base for modification to create the Heterogeneous Residuals model.

| Input | Type | Description |
| --- | --- | ---|
| **data_est** | object | name of the PanDas dataframe  |
| **number_of_splits** | int | number of splits  |
| **feature_name** | list | list of features used for predictiong treatment and outcome  |
| **outcome_name** | str | name of outcome feature  |
| **treatment_name** | str | name of treatment feature  |
| **het_feature** | list | list of features used for driving heterogeneity. Note this can be different from **feature_name**. If user's objective is to estimate as fine-grained as possible, then the recommendation is to set **het_feature**==**feature_name**  |
| **ymodel** | object | model used for predicting outcome and the treatment matrix. This should be a continuous predictor, unless the treatment matrix is all indictors.  |
| **tmodel** | object | model used for predicting the treatment indicator.  |
| **force_second_stage** | str | {None, 'OLS','CVLasso', 'Lasso'} : this allows the user to set what the second stage regression is for making inference. If *None* is selected, it automatically uses *CVLasso* if $dim(D)/N >0.20$. This selection is done in the function *HRmodel.hdm_selection* |



HRmodel.**het_ols**(*data_est, number_of_splits, feature_name, outcome_name, treatment_name, het_feature, ymodel, tmodel, force_second_stage*)

A data-splitting way of doing OLS to estimate heterogeneous treatment effects.

| Input | Type | Description |
| --- | --- | ---|
| **data_est** | object | name of the PanDas dataframe  |
| **number_of_splits** | int | number of splits  |
| **feature_name** | list | list of features used for predictiong treatment and outcome  |
| **outcome_name** | str | name of outcome feature  |
| **treatment_name** | str | name of treatment feature  |
| **het_feature** | list | list of features used for driving heterogeneity. Note this can be different from **feature_name**. If user's objective is to estimate as fine-grained as possible, then the recommendation is to set **het_feature**==**feature_name**  |






HRmodel.**hist_valid**.feature_balance(*df,
                    distri_check, ipw_diff,
                    feature_list, pscore, treatment*)

This implement a propensity controlling/weighting way of assess unconfoundedness, where we control for the propensity score in an OLS, or do a weighting scheme.

| Input | Type | Description |
| --- | --- | ---|
|**df**  | (DataFrame) | dataframe name
|**distri_check**  | (boolean)   | whether to plot weighted features 
|**ipw_diff**  | (boolean)   | whether to calculate IPW differences
|**feature_list**  | (list)      | list of feature names to check balance for.
|**pscore**  | (String)    | name of propensity score in dataframe
|**treatment**  | (String)    | name of treatment indicator in dataframe




HRmodel.**hist_valid**.plot_difference(*results_df, figure_size*)

This uses the outputed dataframe from **feature_balance** to create a figure similar to the _cobalt_ R-package. 


| Input | Type | Description |
| --- | --- | ---|
|**df**  | (DataFrame) | dataframe name
|**figure_size**  | (tuple)   | two-element tuple that tells the figure size





HRmodel.**hist_valid**.summarize_result_df(*results_df*)

This uses the outputted dataframe from the function __feature_balance__ and shows the distribution of raw differences, OLS differences, and distributions of statistically significant differences.

| Input | Type | Description |
| --- | --- | ---|
|**df**  | (DataFrame) | dataframe name





