# Home Credit Default Risk Analysis

A comprehensive machine learning project that builds and compares multiple predictive models to identify loan default risk factors using real-world credit application data. Conducted as a group assignment for the Algorithmic Thinking for Data Analysis course.

## 📊 Project Overview

This project implements a complete data science pipeline analyzing home credit applications to predict default risk. The analysis processes raw application data through feature engineering, trains five different machine learning models, and performs extensive hyperparameter optimization to identify the best predictive approach.

**Note**: The code provided here is an **optimized and simplified version** designed for smooth execution and demonstration purposes. By extending hyperparameter search ranges, increasing training iterations, and fine-tuning model parameters, performance can be significantly improved—though at the cost of substantially longer computation times. For example, XGBoost with `eta=0.01` and `nrounds=1500` (vs. the simplified `eta=0.05` and `nrounds=1000`) achieves noticeably better Kappa scores but takes 3-4x longer to train.

## 🔬 Analysis Pipeline

### 1. Data Preprocessing & Feature Engineering

The code performs extensive data preparation:

- **Variable Selection**: Focuses on 12 quantitative variables (income, credit amounts, external scores, employment history) and 5 categorical variables (contract type, gender, asset ownership)
- **Data Cleaning**: Removes anomalous employment records (365,243 days), filters missing annuity/goods price values
- **Date Conversions**: Transforms negative day values into interpretable age, years employed, and months since phone change
- **Missing Value Treatment**: Imputes missing external credit scores with median values and creates indicator flags to preserve information about missingness

### 2. Advanced Feature Engineering (20+ Derived Features)

The code creates sophisticated interaction and ratio features:

**Financial Ratios**:
- `CREDIT_TO_INCOME`: Loan burden relative to income
- `ANNUITY_TO_INCOME`: Payment burden metric
- `GOODS_PRICE_TO_INCOME`: Purchase affordability
- `ANNUITY_TO_CREDIT`: Repayment rate indicator

**Stability Metrics**:
- `EMPLOYED_TO_AGE`: Career consistency measure
- `INCOME_REALTY_OWN` / `INCOME_CAR_OWN`: Wealth-asset interactions

**External Score Interactions**:
- `EXT1_EXT2`, `EXT1_EXT3`: Combined credit reliability signals
- `EXT2_AGE`: Age-adjusted creditworthiness
- `EXT1_CREDITTOINCOME`: Risk-adjusted affordability

### 3. Handling Class Imbalance

The original dataset is severely imbalanced (92% non-default, 8% default). The code implements:
- **Undersampling strategy**: Creates balanced datasets by randomly sampling majority class to match minority class size
- **Comparative approach**: Trains all models on both balanced and unbalanced datasets to compare performance
- **Weighting mechanisms**: For models like GLM and kNN, implements class weights to handle imbalance algorithmically

### 4. Model Development & Hyperparameter Optimization

#### Decision Trees
- Tests 8 different parameter combinations varying complexity penalty (cp: 0.0005-0.002), minimum split size (1500-2500), and class priors
- Generates interpretable rule-based classification with variable importance rankings
- Creates visual decision tree diagrams showing splitting criteria
- **Optimization potential**: Expanding the cp grid to include values like 0.0001-0.0003 and testing minsplit values up to 5000 yields more nuanced trees

#### XGBoost (Best Performer)
- Implements gradient boosting with extensive hyperparameter tuning
- **Current settings** (optimized for speed): Learning rates (eta: 0.03-0.05), tree depth (4 levels), minimum child weights (200-250), 1000 rounds
- **High-performance settings** (slower): eta: 0.01, nrounds: 1500-2000, testing depth 5-6, min_child_weight: 100-300, subsample: 0.7-0.9
- Uses early stopping to prevent overfitting (monitors test set performance)
- Calculates scale_pos_weight automatically for unbalanced data

#### Logistic Regression (GLM)
- Tests multiple variable sets from minimal (5 variables) to comprehensive (7+ variables)
- Implements weighted regression for unbalanced datasets
- Serves as transparent baseline for comparison
- **Optimization potential**: Testing polynomial features and interaction terms can improve performance

#### k-Nearest Neighbors
- Tests different neighborhood sizes (k: 5, 7, 9, 11)
- Experiments with 4 different variable sets of increasing complexity
- Implements instance duplication to handle class weights
- Evaluates distance-based classification performance
- **Note**: k-NN shows minimal improvement with additional tuning; fundamental limitations of distance-based methods apply

#### Neural Networks
- Trains deep learning models with multiple architectures (4-2, 8-4 hidden layer configurations)
- Uses 3-fold cross-validation to ensure robust performance estimates
- **Current settings**: stepmax: 5e4, learningrate: 0.03, 6,000 sample subset

## 📈 Key Findings

### Model Performance Comparison

The code generates comprehensive performance metrics for all models:


**Performance vs. Computation Trade-off**: The optimized parameters listed can improve XGBoost Kappa by 5-10% but increase training time from ~5 minutes to ~20-30 minutes.

**Critical Insight**: Even with extensive optimization, the best models plateau around 55% sensitivity, demonstrating that credit default is influenced by unpredictable life events (illness, job loss) not captured in application data.

### Variable Importance Analysis

The XGBoost model reveals the most predictive features:

**Top 5 Predictors** (by Gain metric):
1. **EXT_SOURCE_2** & **EXT_SOURCE_3**: External credit bureau scores dominate predictions
2. **EXT1_EXT3**: Interaction between multiple credit scores captures comprehensive credit history
3. **ANNUITY_TO_CREDIT**: Payment-to-loan ratio indicates repayment feasibility
4. **AGE**: Life stage correlates with financial stability
5. **CREDIT_TO_INCOME**: Affordability ratio measures financial stress

**Key Pattern**: External credit scores contribute >40% of total predictive power, while engineered ratio features capture the remaining variance. This hierarchy remains consistent across both simplified and optimized parameter settings.

### Client Demographics

The analysis reveals typical applicant characteristics:
- **Age Distribution**: Peak between 30-45 years (high-expense life phase)
- **Income Profile**: Concentrated in low-to-moderate range with minimal financial buffers
- **Employment Variance**: Wide range of job stability, strongly correlating with default risk
- **Asset Ownership**: Car/home ownership significantly reduces default probability

### Dataset Balance Impact

Comparing balanced vs. unbalanced training:
- **Balanced datasets**: Improve sensitivity (default detection) by 15-20%
- **Unbalanced datasets**: Achieve higher overall accuracy but miss more defaults
- **Trade-off**: Balanced training sacrifices 5-10% overall accuracy to catch 20% more defaults
- **Optimal approach**: Use balanced data for XGBoost, which best navigates this trade-off

## 📊 Visualizations Generated

The code produces 9 analytical visualizations:

1. **Class distribution plots**: Show original imbalance and balanced datasets
2. **Variable importance charts**: Bar plots for Decision Tree and XGBoost feature rankings
3. **Model comparison plots**: 
   - Kappa score comparison with error bars
   - Balanced accuracy across all models
   - Scatter plot of Kappa vs. Balanced Accuracy (sized by AUC)
4. **Dataset comparison**: Side-by-side performance for balanced vs. unbalanced
5. **Performance heatmap**: Color-coded grid showing all metrics across all models
6. **Decision tree diagram**: Visual representation of splitting rules


## 📁 Output Files

The code generates these artifacts:
- `tree_results_unbalanced.csv` / `tree_results_balanced.csv`
- `xgb_results_unbalanced.csv` / `xgb_results_balanced.csv`
- `glm_results_unbalanced.csv` / `glm_results_balanced.csv`
- `kNN_results_unbalanced.csv` / `kNN_results_balanced.csv`
- `DL_balanced_results.csv`
- `all_models_comparison.csv` (comprehensive results)

## 🎯 Core Insight

The 50-55% sensitivity ceiling across all approaches (even with extensive optimization) indicates that **loan default prediction is fundamentally limited** by unmeasurable human factors. The models excel at risk stratification (ranking applicants by risk) but cannot perfectly predict individual outcomes. This finding emphasizes that predictive models should guide—not replace—human judgment in lending decisions.

## 👥 Authors

Mehdi Mouja, Badr Benesrighe, Jean Jiptner, Othman Bakki  
*Date: December 4, 2025*
