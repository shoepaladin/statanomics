# statanomics
Here is a one-stop shop of:
1. causal inference models to estimate average treatment effects (ATE/ATET);
2. causal inference models to estimate Heterogeneous Treatment Effects (HTE); and
3. diagnostics for assess underlying assumptions needed for causal inference following the Neyman-Rubin's Potential Outcomes model.

The API and syntaxes are centralized, so you can swap one model for another just by changing the functional call!




## 1. causalmodels
Propensity-score-based models to estimate the average treatment effect and average treatment effect on the treated. While models such as OLS, double robust, and inverse propensity-weighting models are covered, propensity score matching models are not (because they are hard). 

## 2. diagnostics
Various metrics and tests to asses the unconfoundedness and overlap assumptions - following the potential outcomes models. There are no tests for the stable unit treatment value assumption (SUTVA).
