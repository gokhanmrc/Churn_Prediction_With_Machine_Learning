# Churn Prediction with Machine Learning

This repository contains a machine learning project focused on predicting customer churn using a dataset that includes various customer attributes and service usage information. The project involves data preprocessing, exploratory data analysis, feature engineering, model selection, and evaluation.

## 📌 Project Objectives

- Predict whether a customer is likely to churn or not.
- Perform exploratory data analysis (EDA) to understand key churn indicators.
- Apply feature engineering to enhance model performance.
- Train and compare multiple machine learning models.
- Evaluate model performance using appropriate metrics.

## 📁 Dataset

The dataset contains 7,043 rows and 21 variables related to customer demographics, services signed up for, and account information.  
It is a simulated Telco dataset, widely used in churn prediction exercises.

- Categorical, numerical, and binary variables included
- Target variable: `Churn`

## 🛠️ Technologies & Libraries

- **Python**
- `pandas`, `numpy` — for data manipulation
- `matplotlib`, `seaborn` — for data visualization
- `scikit-learn` — for machine learning models and evaluation
- `XGBoost`, `LightGBM`, `CatBoost` — advanced gradient boosting frameworks

## 🔍 Project Workflow

1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Outlier detection and treatment (if needed)
2. **Exploratory Data Analysis (EDA)**
   - Analyzing churn distribution
   - Investigating relationships between features and churn
3. **Feature Engineering**
   - Creating new meaningful variables
   - Encoding binary/multiclass features
4. **Modeling**
   - Logistic Regression
   - K-Nearest Neighbors
   - Decision Tree
   - Random Forest
   - XGBoost
   - LightGBM
   - CatBoost
   - Hyperparameter tuning with GridSearchCV
5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - ROC-AUC
   - Cross-validation
   - Model comparison

## 📊 Results

| Model          | Accuracy | F1-Score | ROC-AUC  |
|----------------|----------|----------|----------|
| Random Forest  | 0.7994   | 0.5648   | 0.8454   |
| XGBoost        | 0.8007   | 0.5822   | 0.8448   |
| LightGBM       | 0.8031   | 0.5890   | 0.8454   |
| **CatBoost**   | **0.8048** | **0.5860** | **0.8469** |

- Among 8 tested algorithms, **CatBoost** achieved the best performance with a **ROC-AUC score of 0.8469** and an **F1-score of 0.5860**, demonstrating strong predictive power for customer churn.

