# E-Commerce Revenue Prediction: Ensemble Machine Learning Analysis

## Project Overview

This repository contains a comprehensive machine learning project focused on predicting e-commerce revenue using ensemble methods. The project analyzes marketing dynamics and develops a robust predictive model capable of guiding budget allocation decisions in a high-variance environment.

## Objectives

1. **Develop a robust predictive model** that estimates revenue based on advertising metrics (Ad_Spend, Clicks, Impressions) while handling heteroscedasticity
2. **Identify priority action levers** through feature importance analysis to guide strategic marketing decisions
3. **Rigorously compare** two ensemble algorithm families (Gradient Boosting vs Bagging Random Forests) in a marketing context

## Key Results

- **XGBoost Model Performance**: R² = 0.5244, MAE = $1,634.29, RMSE = $2,273.93
- **Predictive Accuracy**: 82.36% (1 - MAPE)
- **Feature Hierarchy**: Impressions (0.35) > Clicks (0.25) > Ad_Spend (0.15)
- **Key Insight**: Reach and engagement quality trump raw budget volume

## Dataset Description

**Source**: Synthetic e-commerce transaction data (100,000 observations)

**Key Variables**:
- **Target**: `Revenue` (continuous, $0-$5,000)
- **Ad Metrics**: Ad_Spend, Clicks, Impressions, Ad_CTR, Ad_CPC
- **Transaction Data**: Units_Sold, Discount_Applied, Conversion_Rate
- **Categorical**: Category (5 types: Books, Clothing, Electronics, Home Appliances, Toys), Region
- **Temporal**: Transaction_Date (aggregated to Month, DayOfWeek)

**Data Integrity**: 0 missing values, 15 features

**Transformation**: 100,000 transactions → 5,490 campaign-level aggregates (noise reduction)

## Methodology

### Problem Formulation
Supervised regression: f: ℝᵖ → ℝ minimizing MSE loss

### Algorithms

#### 1. XGBoost (Winner)
- **Why**: Sequential iterative boosting corrects residuals progressively
- **Advantages**:
  - Robust to outliers (iterative isolation vs OLS sensitivity)
  - Captures non-linearities via decision tree splits
  - Built-in L₂ regularization prevents overfitting
  - Kaggle-proven empirical performance
- **Hyperparameters**: n_estimators=300, learning_rate=0.05, max_depth=7, subsample=0.8

#### 2. Bagging Random Forest (Baseline)
- **Why**: Parallel ensemble variance reduction via averaging
- **Advantages**:
  - Intrinsically stable (vs boosting overfitting risk)
  - Feature importance interpretation
  - Fair comparison baseline
- **Configuration**: 10 Bagging estimators × 100 RF trees each

### Data Pipeline

```python
# 1. Aggregation: Reduce stochastic noise
data = df.groupby(['Transaction_Date', 'Category', 'Region', 'Month', 'DayOfWeek']).agg({
    'Ad_Spend': 'sum', 'Clicks': 'sum', 'Impressions': 'sum', 'Revenue': 'sum'
}).reset_index()

# 2. Feature Engineering
df['Spend_Efficiency'] = df['Ad_Spend'] * df['Clicks']  # Interaction term
df['Month'] = df['Transaction_Date'].dt.month           # Temporal extraction
df['DayOfWeek'] = df['Transaction_Date'].dt.dayofweek

# 3. Encoding & Scaling
X = pd.get_dummies(X, columns=['Category', 'Region'], drop_first=True)
X_scaled = StandardScaler().fit_transform(X_train)  # Avoid data leakage

# 4. Hierarchical Train/Val/Test Split
Train: 3,513 (64%) | Validation: 879 (16%) | Test: 1,098 (20%)
```

### Hyperparameter Optimization
- **Grid Search**: 27 combinations (n_estimators × learning_rate × max_depth)
- **Cross-Validation**: 4-fold CV on Train+Val
- **Metric**: R² (variance explained, business-intuitive)
- **Best Params**: Identified via maximizing validation R²

## Results & Analysis

### Performance Comparison

| Metric | XGBoost | Bagging RF |
|--------|---------|-----------|
| R² Score | **0.5244** | 0.5037 |
| MAE ($) | **1,634.29** | 1,678.93 |
| RMSE ($) | **2,273.93** | 2,322.93 |
| MAPE (%) | **17.64** | — |
| Accuracy | **82.36%** | — |

**Winner**: XGBoost (+4% relative improvement in R²)

### Feature Importance Hierarchy
1. **Impressions (0.35)** - Reach is the dominant driver
2. **Clicks (0.25)** - Engagement quality matters
3. **Ad_Spend (0.15)** - Budget volume ranks 3rd

**Strategic Implication**: Maximize reach and engagement quality, not just spending

### Key Visualizations
- **Correlation Matrix**: Moderate multicollinearity (0.68-0.76) among Ad_Spend/Clicks/Impressions
- **Boxplots**: Extreme right-skew (outliers to $5,000) justifies ensemble methods
- **Residual Analysis**: XGBoost concentrates errors around zero better
- **Learning Curves**: Convergence plateau suggests larger datasets would improve performance

## Model Strengths

✅ **Robust outlier handling** - Isolates aberrant values without distorting mainstream predictions

✅ **Non-linearity capture** - Automatic threshold detection (e.g., "If Impressions>10k AND CTR>2%, boost revenue prediction 20%")

✅ **Interpretability** - Feature importance provides actionable business insights

✅ **Rigorous validation** - Strict Train/Val/Test separation eliminates optimism bias

## Model Limitations

❌ **Unexplained variance (48%)** - Exogenous factors (promotions, competition, WOM) not captured

❌ **Synthetic data** - External validity limited; real-world validation needed

❌ **Static model** - No continuous retraining; market evolution not handled

❌ **Temporal naivety** - Uses Month/DayOfWeek but misses latency effects (ad today → purchase 2 weeks later)

❌ **No confidence intervals** - Point predictions lack uncertainty quantification

## Future Improvements

### Methodological Extensions
- Quantile regression for confidence intervals
- Nested cross-validation for robustness
- Bayesian optimization (Optuna) instead of Grid Search
- Ensemble stacking (XGBoost + RF + meta-learner)
- SHAP values for local interpretability

### Data Enrichment
- Macroeconomic variables (unemployment, consumer confidence)
- Customer RFM features (Recency, Frequency, Monetary value)
- Lagged features (impact of ads 1-4 weeks prior)
- Advanced seasonality (STL decomposition)

### Production Deployment
- MLOps pipeline with monthly retraining
- Automated budget allocation recommender
- A/B testing framework for model validation
- Interactive dashboard (Streamlit/Dash)

## Report Structure

The main report (`rapport_TAA_spaced.tex`) follows academic standards:

1. **Abstract** (150-250 words) - Problem, methodology, results, conclusion
2. **Introduction** - Context, scientific objectives, literature review
3. **Methodology** - Algorithm justification, hyperparameters, regularization strategy
4. **Experiments** - Dataset description, preprocessing, results, analysis
5. **Conclusion** - Synthesis, limitations, improvement roadmap

**Length**: 4-5 pages (excluding cover page)

**Language**: French

**Format**: LaTeX with proper spacing and figure placeholders

## Key Findings Summary

### Question
*How to predict e-commerce revenue accurately despite non-linearity and extreme outliers?*

### Answer
Ensemble methods (XGBoost + Bagging RF) achieve **82% accuracy** through three complementary strategies:

1. **Data aggregation** - Transaction→Campaign level reduces stochastic noise
2. **Feature engineering** - Efficiency metrics capture multiplicative interactions
3. **Recursive partitioning** - Isolate aberrant values in distinct conditional rules

### Strategic Takeaway
**Quality over Quantity**: Prioritize reach (impressions) + engagement (clicks) over raw budget. Electronics shows high engagement efficiency; Toys shows high revenue predictability.

## References

- Breiman, L. (2001). "Random Forests". *Machine Learning*, 45(1), 5-32
- Friedman, J. H. (2001). "Greedy function approximation: A gradient boosting machine". *Annals of Statistics*, 29(5), 1189-1232
- Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System". *KDD*, 785-794

## Author & Contact

**Academic Project** - Master's in Computer Science

Created: January 2026

---

⭐ **If this helps, consider starring the repo!**
