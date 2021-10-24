# statanomics
update date 23oct2021
***
Here is a one-stop shop of:
1. teaching material for scientists interested in learning more about causal inference
2. causal inference models to estimate average treatment effects (ATE/ATET);
3. causal inference models to estimate Heterogeneous Treatment Effects (HTE); 
4. diagnostics for assess underlying assumptions needed for causal inference following the Neyman-Rubin's Potential Outcomes model.

The API and syntaxes are centralized, so you can swap one model for another just by changing the functional call!

## causal_inference_cc
Causal Inference Crash Course which is a series of slides/presentations that covers the basics of causal inference. The audience is a scientist interested in learning about causal inference. This is a WiP series. As of Oct-2021, this presentation covers the foundations, matching-based models, and inference properties of matching-based models.


## causalmodels
Propensity-score-based models to estimate the average treatment effect and average treatment effect on the treated. While models such as OLS, double robust, and inverse propensity-weighting models are supported.
* Pending functions:
	- quantile regression 
	- regressions with discrete outcomes
	- instrumental variable (IV) regression via 2SLS
	- difference-in-difference
	- heterogeneous treatment effect (HTE) / conditional average treatment effect (CATE) models
* No current plans to support:
	- propensity score matching models
	- kernel and local regression models
	- synthetic control models
	


## diagnostics
Various metrics and tests to asses the unconfoundedness and overlap assumptions - following the potential outcomes models. There are no tests for the stable unit treatment value assumption (SUTVA).
* Pending functions:
	- Coefficient stability metric via [Oster (2016)](https://www.brown.edu/research/projects/oster/sites/brown.edu.research.projects.oster/files/uploads/Unobservable_Selection_and_Coefficient_Stability_0.pdf)
	- Exogeneity sensitivity via [Imbens 2003](https://scholar.harvard.edu/files/imbens/files/sensitivity_to_exogeneity_assumptions_in_program_evaluation.pdf)

## heterogeneousresiduals
A custom implementation of a heterogeneous treatment effects version for double machine learning, based on The Heterogeneous Residuals model builds on Semenova, Goldman, Chernozhukov, and Taddy (2021).



