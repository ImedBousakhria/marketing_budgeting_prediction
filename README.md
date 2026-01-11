# E-Commerce Revenue Prediction

## Overview

Machine learning project predicting e-commerce revenue using ensemble methods. Analyzes marketing dynamics to optimize budget allocation in high-variance environments.

## Objectives

1. Build robust revenue prediction model using advertising metrics (Ad_Spend, Clicks, Impressions)
2. Identify priority marketing levers through feature importance analysis
3. Compare XGBoost vs Bagging Random Forest performance

## Key Results

- **XGBoost Performance**: R² = 0.52, MAE = $1,634, RMSE = $2,274
- **Accuracy**: 82.36%
- **Top Features**: Impressions (0.35) > Clicks (0.25) > Ad_Spend (0.15)
- **Insight**: Reach and engagement quality matter more than raw budget

## Dataset

**Source**: 100k synthetic e-commerce transactions

**Variables**: Revenue (target), Ad_Spend, Clicks, Impressions, Ad_CTR, Ad_CPC, Category, Region, temporal features

**Preprocessing**: Aggregated to 5,490 campaign-level observations to reduce noise

## Methodology

**Problem**: Supervised regression minimizing MSE

**Models**:
- **XGBoost** - Sequential boosting with L2 regularization (winner)
- **Bagging Random Forest** - Parallel ensemble baseline

**Train/Val/Test Split**: 64% / 16% / 20%

**Optimization**: Grid Search with 4-fold CV (27 hyperparameter combinations)

## Results

| Metric | XGBoost | Bagging RF |
|--------|---------|-----------|
| R² | **0.5244** | 0.5037 |
| MAE ($) | **1,634** | 1,679 |
| RMSE ($) | **2,274** | 2,323 |

**Strategic Takeaway**: Maximize impressions and engagement quality over budget volume. Electronics shows high engagement efficiency; Toys shows highest revenue predictability (r=0.63).

## Author

[Imed Bousakhria](https://github.com/imedBousakhria)

---

⭐ Star this repo if you find it useful!
