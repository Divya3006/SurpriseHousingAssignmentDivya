# Surprise Housing - House Price Prediction with Regularization

> Predict house prices using Linear Regression, Ridge Regression, and Lasso Regression with regularization. Identify key drivers of house prices and determine optimal lambda values for regularization techniques.

## Table of Contents
* [General Info](#general-information)
* [Business Problem](#business-problem)
* [Dataset Description](#dataset-description)
* [Technologies Used](#technologies-used)
* [Installation & Setup](#installation--setup)
* [Project Structure](#project-structure)
* [Analysis Steps](#analysis-steps)
* [Key Findings](#key-findings)
* [Model Comparison](#model-comparison)
* [Conclusions](#conclusions)
* [Recommendations](#recommendations)
* [Acknowledgements](#acknowledgements)

## General Information

Surprise Housing is a US-based company that uses data analytics to purchase houses at prices below their actual values and flip them at a higher price for profit. The company is now entering the Australian real estate market and needs a data-driven approach to:

1. **Predict actual house values** to identify underpriced properties
2. **Identify significant variables** that drive house prices
3. **Determine optimal regularization parameters** to avoid overfitting

### Business Problem

- **Objective**: Build regression models to predict house prices in the Australian market
- **Challenge**: Identify which features are most important for pricing decisions
- **Goal**: Use regularization (Ridge and Lasso) to create robust, generalizable models
- **Outcome**: Enable data-driven property acquisition decisions and pricing strategies

### Dataset Description

- **Source**: train.csv
- **Observations**: 1,460 houses with sale prices
- **Features**: 79 features including:
  - Property characteristics (bedrooms, bathrooms, lot size)
  - Quality metrics (overall quality, condition)
  - Location features (neighborhood, zoning)
  - Building features (foundation, roofing, materials)
  - Temporal features (year built, year sold)
- **Target Variable**: SalePrice (in USD)

**Feature Categories**:
- **Numeric Features**: Area measurements, quality scores, year information
- **Categorical Features**: Neighborhood, zoning, building type, material types

## Technologies Used

### Programming Language
- Python 3.8+

### Libraries & Frameworks
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - statsmodels
  - scipy

## Analysis Steps

### Step 1: Data Loading and Exploration
- Load housing data with proper handling of missing values
- Display dataset shape, data types, and basic statistics
- Identify and report missing value patterns

**Output**: Dataset overview and data quality assessment

### Step 2: Data Cleaning and Preparation
- Remove unique identifier columns (Id)
- Handle missing values:
  - **Numeric features**: Fill with median values
  - **Categorical features**: Fill with 'None' category
- Standardize data formats

### Step 3: Feature Engineering and Encoding
- One-hot encode categorical variables
- Convert categorical features to numeric format suitable for regression
- Preserve feature interpretability

### Step 4: Exploratory Data Analysis (EDA)
- Analyze target variable distribution
- Visualize correlations with SalePrice
- Identify relationships between features
- Create correlation heatmaps

### Step 5: Train-Test Split and Feature Scaling
- Split data: 80% training, 20% testing
- Apply StandardScaler for feature normalization
- Ensure consistent scaling between train and test sets

### Step 6: Model Building
Build three regression models:

#### 6A: Linear Regression (Baseline)
- Standard OLS regression without regularization
- Serves as performance baseline
- Useful for feature coefficient interpretation

#### 6B: Ridge Regression (L2 Regularization)
- Adds L2 penalty term: λ × Σ(coefficients²)
- Reduces coefficient magnitude but keeps all features
- Find optimal lambda using 5-fold cross-validation

#### 6C: Lasso Regression (L1 Regularization)
- Adds L1 penalty term: λ × Σ|coefficients|
- Can reduce some coefficients exactly to zero
- Performs automatic feature selection
- Find optimal lambda using 5-fold cross-validation

### Step 7: Feature Importance Analysis
- Extract and compare coefficients from all models
- Rank features by importance
- Visualize coefficient differences between models

### Step 8: Model Comparison
- Compare R² scores across models
- Compare RMSE and MAE metrics
- Visualize actual vs predicted prices
- Create side-by-side performance comparison

### Step 9: Residual Analysis
- Analyze prediction errors
- Check for systematic biases
- Assess residual distribution normality
- Create Q-Q plots for each model

### Step 10: Business Recommendations
- Generate actionable insights for acquisition strategy
- Recommend optimal model for deployment
- Provide pricing guidance and market insights

## Key Findings

### Question 1: Which Variables Are Significant in Predicting House Prices?

#### Top 10 Most Significant Features

| Rank | Feature | Correlation | Strength | Business Impact |
|------|---------|-------------|----------|-----------------|
| 1 | OverallQual | 0.7910 | Very Strong | Quality rating - STRONGEST predictor |
| 2 | GrLivArea | 0.7086 | Very Strong | Ground living area - 2nd strongest |
| 3 | GarageCars | 0.6404 | Strong | Garage capacity for vehicles |
| 4 | GarageArea | 0.6234 | Strong | Garage size in sq ft |
| 5 | TotalBsmtSF | 0.6136 | Strong | Total basement square footage |
| 6 | 1stFlrSF | 0.6059 | Strong | First floor area |
| 7 | FullBath | 0.5607 | Strong | Number of full bathrooms |
| 8 | TotRmsAbvGrd | 0.5337 | Strong | Total rooms above ground |
| 9 | YearBuilt | 0.5229 | Strong | Property construction year |
| 10 | YearRemodAdd | 0.5071 | Strong | Year of remodeling/addition |

### Question 2: How Well Do These Variables Describe House Prices?

**Best Model: Ridge Regression with All Features**
- **Test R²**: 0.8780 (explains 87.8% of price variance)
- **Test RMSE**: $30,595 (average prediction error)
- **Test MAE**: $18,900 (typical deviation)
- **R² Gap**: 0.0163 (minimal overfitting - excellent generalization)

## Model Comparison

### Performance Metrics

| Model | Test R² | Test RMSE | Test MAE | R² Gap | Overfitting Status |
|-------|---------|-----------|----------|--------|-------------------|
| **Ridge_AllFeatures** | **0.8780** | **$30,595** | **$18,900** | **0.0163** | **Excellent (No Overfitting)** |
| Lasso_AllFeatures | 0.8271 | $36,412 | $18,987 | 0.1115 | Moderate Overfitting |
| Ridge_Top15 | 0.8123 | $37,944 | $24,100 | -0.0321 | Slight Underfitting |
| Lasso_Top15 | 0.8111 | $38,069 | $24,406 | -0.0304 | Slight Underfitting |
| Ridge_Top10 | 0.7970 | $39,455 | $24,774 | -0.0331 | Slight Underfitting |
| OLS_Top15 | 0.8111 | $38,069 | $24,406 | -0.0304 | Slight Underfitting |

### Why Ridge Regression Wins

**Ridge (L2) with All Features is the clear winner:**
1. **Highest Test R²**: 0.8780 (87.8% accuracy on unseen data)
2. **Lowest Prediction Errors**: RMSE = $30,595, MAE = $18,900
3. **Minimal Overfitting**: R² Gap = 0.0163 (excellent generalization)
4. **Handles Multicollinearity**: L2 regularization shrinks coefficients without eliminating features
5. **Outperforms Alternatives**: 
   - 6.15% better than Lasso on test accuracy
   - 7x less overfitting than Lasso
   - 32.77% better than OLS on average test performance

### Strategy Comparison

| Strategy | Avg Test R² | Avg R² Gap | Best For |
|----------|-------------|------------|----------|
| **Ridge (L2)** | **0.8291** | **-0.0163** | **Production deployment** |
| Lasso (L1) | 0.8117 | +0.0163 | Feature selection, interpretability |
| OLS | 0.6249 | +0.1840 | Simple cases only (not recommended) |

## Recommendations

### 1. Recommended Model for Production

**RECOMMENDED: Ridge Regression with All Features**

**Performance Metrics:**
- Test R²: 0.8780 (explains 87.8% of price variance)
- Test RMSE: $30,595 (average prediction error)
- Test MAE: $18,900 (typical prediction deviation)
- R² Gap: 0.0163 (minimal overfitting)

**Why Ridge?**
- Highest test accuracy among all models tested
- Excellent generalization to unseen properties
- Handles multicollinearity through L2 regularization
- Outperforms OLS by 32.77% on test data
- Keeps all features but shrinks their impact appropriately

**When to Use Ridge:**
- Primary pricing tool for property valuations
- Investment decision making
- Market analysis and forecasting
- Real-time pricing applications

### 2. Alternative: Lasso Regression

**Lasso (L1) - For Feature Selection**
- Test R²: 0.8271 (good but lower than Ridge)
- Automatic feature selection (sets some coefficients to zero)
- Easier to explain to stakeholders
- Use when interpretability is critical or computational resources are limited

### 3. NOT RECOMMENDED: OLS Without Regularization

**Why Avoid Plain OLS:**
- Avg Test R²: 0.6249 (poor performance on new data)
- Avg R² Gap: 0.1840 (severe overfitting)
- With all features: R² Gap = 0.84 (catastrophic overfitting)
- Cannot be trusted for production predictions

### 4. Business Strategy & Pricing Insights

| Factor | Impact Level | Strategic Action | Expected Value Impact |
|--------|--------------|------------------|----------------------|
| **Quality Metrics** | Dominates | Prioritize properties with high OverallQual ratings | Strongest price driver |
| **Property Size** | Very High | Target larger living areas (GrLivArea, TotalBsmtSF) | Direct correlation to price |
| **Location** | High | Focus on premium neighborhoods | $30-50k+ premium |
| **Age/Renovation** | Moderate | Prefer recently built or renovated properties | Significant value add |
| **Garage/Parking** | Moderate | Ensure adequate garage capacity and space | Convenience factor valued |

### 5. Implementation Guidelines

**Data Requirements:**
- Collect accurate measurements for all property features
- Ensure quality ratings are standardized and consistent
- Update model quarterly with new market data

**Model Deployment:**
- Use Ridge model for all property valuations
- Expected accuracy: ±$18,900 (MAE)
- Confidence level: HIGH (87.8% variance explained)

**Risk Management:**
- Validate predictions on diverse property samples
- Monitor model performance on new data
- Apply ensemble predictions for high-value investments

## Conclusions

### Key Takeaways

1. **Ridge Regression is the Superior Model**
   - Achieves 87.8% accuracy on unseen data
   - Minimal overfitting with excellent generalization
   - Robust predictions for real-world deployment

2. **Top 5 Price Drivers Identified**
   - OverallQual, GrLivArea, GarageCars, TotalBsmtSF, 1stFlrSF
   - Quality metrics dominate pricing decisions
   - Size and location features are critical

3. **Regularization is Essential**
   - Ridge (L2) prevents overfitting while maintaining accuracy
   - OLS without regularization fails catastrophically
   - L2 penalty successfully handles multicollinearity

4. **Business Value Delivered**
   - Reliable pricing model for Australian market entry
   - Data-driven acquisition strategy enabled
   - Average prediction error: $18,900 (excellent precision)

### Project Files

- `SurpriseHousingEDA&LR.ipynb` - Complete analysis notebook with EDA, modeling, and evaluation
- `train.csv` - Training dataset with house features and prices
- `README.md` - Project documentation and findings

## Acknowledgements

- This project was inspired by the Surprise Housing Case Study from UpGrad
- Dataset source: House prices data for regression analysis
- Analysis performed using Python with scikit-learn, pandas, and matplotlib
