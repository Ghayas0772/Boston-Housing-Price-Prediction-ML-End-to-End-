# Housing Prices Prediction Project

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green) ![GitHub Workflow](https://img.shields.io/badge/CI-CD-blue)  

This project analyzes the Boston Housing dataset and develops machine learning models to predict the median value of owner-occupied homes (`MEDV`). The workflow includes data cleaning, exploratory data analysis (EDA), feature engineering, outlier handling, scaling, and predictive modeling.

---

## Project Structure

- housing-prices-project/
  - data/
    - BostonHousing.csv           # Original raw dataset
    - BostonHousing_clean.csv     # Cleaned dataset after outlier handling
  - notebooks/
    - EDA_and_Preprocessing.ipynb # Exploratory Data Analysis and Preprocessing steps
  - models/
    - trained_models/             # Saved machine learning models (optional)
  - src/
    - preprocessing.py            # Data cleaning, outlier capping, log-transform
    - feature_engineering.py      # Feature selection and scaling
    - modeling.py                 # Model training and evaluation scripts
  - README.md                     # Project documentation

---

## Dataset Overview

The Boston Housing dataset contains **506 rows** and **14 columns**.  

| Feature      | Description |
|--------------|-------------|
| CRIM         | Per capita crime rate by town |
| ZN           | Proportion of residential land zoned for lots over 25,000 sq.ft. |
| INDUS        | Proportion of non-retail business acres per town |
| CHAS         | Charles River dummy variable (1 if tract bounds river; 0 otherwise) |
| NOX          | Nitric oxides concentration (parts per 10 million) |
| RM           | Average number of rooms per dwelling |
| AGE          | Proportion of owner-occupied units built prior to 1940 |
| DIS          | Weighted distances to five Boston employment centers |
| RAD          | Index of accessibility to radial highways |
| TAX          | Full-value property-tax rate per $10,000 |
| PTRATIO      | Pupil-teacher ratio by town |
| LSTAT        | % lower status of the population |
| MEDV         | Median value of owner-occupied homes in $1000s |
| CAT. MEDV    | Optional binary target (0 = low, 1 = high) |

---

## Exploratory Data Analysis (EDA)

- Checked data types and missing values (none found).  
- Correlation Analysis revealed top predictors of `MEDV`:
  - `RM` (0.698)  
  - `LSTAT` (-0.797)  
  - `PTRATIO` (-0.524)  
- Skewed features: `CRIM`, `TAX`, `LSTAT` → applied log-transform.  
- Outliers detected via IQR method and capped.

---

## Data Preprocessing

1. **Outlier Capping**: Limit extreme values using IQR.  
2. **Log Transformation**: Reduce skewness in `CRIM`, `TAX`, `LSTAT`.  
3. **Scaling**: StandardScaler applied to numerical features.  
4. **Feature Selection**: Selected top 8 correlated features for modeling.

---

## Machine Learning Models

| Model                         | RMSE   | MAE    | R²    |
|-------------------------------|--------|--------|-------|
| Linear Regression             | 3.210  | 2.220  | 0.789 |
| Random Forest Regressor       | 2.423  | 1.824  | 0.880 |
| K-Nearest Neighbors Regressor | 2.929  | 1.970  | 0.825 |
| Support Vector Regression     | 3.230  | 2.087  | 0.787 |

 **Best Model:** Random Forest Regressor  

**Insights:**  
- More rooms (`RM`) → higher price.  
- Lower status (`LSTAT`) → lower price.  
- Outlier handling improved performance.  

---

## Key Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
----
##  Visualizations

Heatmap for correlations
Distribution plots for features & target
Boxplots for outlier detection
Pairplots for key features
