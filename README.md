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

## 3. causal_inference_cc
Causal Inference Crash Course which is a series of slides/presentations that covers the basics of causal inference. The audience is a scientist interested in learning about causal inference. This is a WiP series. As of Oct-2021, this presentation covers the foundations, matching-based models, and inference properties of matching-based models.

## 4. heterogeneousresiduals
A custom implementation of a heterogeneous treatment effects version for double machine learning, based on The Heterogeneous Residuals model builds on Semenova, Goldman, Chernozhukov, and Taddy (2021).



