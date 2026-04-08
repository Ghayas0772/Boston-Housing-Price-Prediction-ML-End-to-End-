Housing Prices Prediction Project

This project analyzes the Boston Housing dataset and develops machine learning models to predict the median value of owner-occupied homes (MEDV). The workflow includes data cleaning, exploratory data analysis (EDA), feature engineering, outlier handling, scaling, and predictive modeling using multiple algorithms.

Project Structure
housing-prices-project/
│
├── data/
│   ├── BostonHousing.csv           # Original raw dataset
│   └── BostonHousing_clean.csv     # Cleaned dataset after outlier handling
│
├── notebooks/
│   └── EDA_and_Preprocessing.ipynb # Exploratory Data Analysis and Preprocessing steps
│
├── models/
│   └── trained_models/             # Saved machine learning models (optional)
│
├── src/
│   ├── preprocessing.py            # Data cleaning, outlier capping, log-transform
│   ├── feature_engineering.py      # Feature selection and scaling
│   └── modeling.py                 # Model training and evaluation scripts
│
└── README.md                       # Project documentation
Dataset

The Boston Housing dataset contains 506 rows and 14 columns. It includes demographic, economic, and structural features to predict housing prices (MEDV).

Feature	Description
CRIM	Per capita crime rate by town
ZN	Proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS	Proportion of non-retail business acres per town
CHAS	Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX	Nitric oxides concentration (parts per 10 million)
RM	Average number of rooms per dwelling
AGE	Proportion of owner-occupied units built prior to 1940
DIS	Weighted distances to five Boston employment centers
RAD	Index of accessibility to radial highways
TAX	Full-value property-tax rate per $10,000
PTRATIO	Pupil-teacher ratio by town
LSTAT	% lower status of the population
MEDV	Median value of owner-occupied homes in $1000s
CAT. MEDV	Optional binary target (0 = low, 1 = high)
Exploratory Data Analysis (EDA)
Data Summary: Checked data types, missing values (none found), descriptive statistics.
Correlation Analysis: Identified top features correlated with MEDV:
RM (0.698)
LSTAT (-0.797)
PTRATIO (-0.524)
Skewness & Variability:
Right-skewed features: CRIM, TAX, LSTAT.
Features like RM and MEDV are roughly symmetric.
Outlier Detection:
Used the Interquartile Range (IQR) method.
Outliers were capped to improve model stability.
Data Preprocessing
Outlier Capping: Applied IQR method to limit extreme values.
Log Transformation: Skewed features (CRIM, TAX, LSTAT) were log-transformed to reduce skewness.
Scaling: StandardScaler was applied to numerical features to normalize values.
Feature Selection: Selected top 8 features highly correlated with MEDV for modeling.
Machine Learning Models

The following regression models were trained and evaluated:

Model	RMSE	MAE	R²
Linear Regression	3.210	2.220	0.789
Random Forest Regressor	2.423	1.824	0.880
K-Nearest Neighbors Regressor	2.929	1.970	0.825
Support Vector Regression	3.230	2.087	0.787

Observations:

Random Forest performed the best in terms of RMSE and R².
Linear Regression and KNN also showed reasonable predictive ability.
SVR performed slightly worse on this dataset.
Key Libraries
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
Visualizations
Heatmap for feature correlation
Distribution plots for target and features
Boxplots for outlier detection
Pairplots for key features (RM, LSTAT, PTRATIO, MEDV)
Insights
Neighborhoods with more rooms per dwelling (RM) have higher median prices.
Lower status population (LSTAT) negatively impacts housing price.
Outlier handling and log-transformation improved model performance.
Next Steps
Hyperparameter tuning for Random Forest and SVR.
Test ensemble models (e.g., Gradient Boosting, XGBoost).
Explore feature interactions or polynomial features.
Deploy model for interactive prediction.
How to Run
