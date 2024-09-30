#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset (assuming 'used_cars.csv' is your dataset)
data = pd.read_csv('used_cars.csv')

# Data Processing
# Handling missing values
data.dropna(inplace=True)

# Convert categorical variables to numerical
# Assuming 'make', 'model', 'condition', etc. are categorical features
categorical_features = ['make', 'model', 'condition']
numerical_features = ['year', 'mileage']

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Splitting the data into features and target variable
X = data.drop('price', axis=1)  # Features
y = data['price']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return mean_absolute_error(y_test, predictions)

# 1. Linear Regression
linear_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('linear_regressor', LinearRegression())])
linear_mae = evaluate_model(linear_pipeline, X_train, X_test, y_train, y_test)
print(f'Linear Regression MAE: {linear_mae}')

# 2. Logistic Regression (for classification)
# Creating price categories (e.g., low, medium, high)
bins = [0, 10000, 20000, 30000, np.inf]
labels = ['low', 'medium', 'high', 'very_high']
data['price_category'] = pd.cut(data['price'], bins=bins, labels=labels)

# Splitting the new data
X_class = data.drop(['price', 'price_category'], axis=1)
y_class = data['price_category']

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

logistic_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('logistic_regressor', LogisticRegression())])
logistic_pipeline.fit(X_train_class, y_train_class)
logistic_score = logistic_pipeline.score(X_test_class, y_test_class)
print(f'Logistic Regression Accuracy: {logistic_score}')

# 3. Ridge Regression
ridge_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('ridge_regressor', Ridge())])
ridge_mae = evaluate_model(ridge_pipeline, X_train, X_test, y_train, y_test)
print(f'Ridge Regression MAE: {ridge_mae}')

# 4. KNN Regression
knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('knn_regressor', KNeighborsRegressor(n_neighbors=5))])
knn_mae = evaluate_model(knn_pipeline, X_train, X_test, y_train, y_test)
print(f'KNN Regression MAE: {knn_mae}')

# Summary of results
print("\nModel Comparison:")
print(f"Linear Regression MAE: {linear_mae}")
print(f"Logistic Regression Accuracy: {logistic_score} (for classification only)")
print(f"Ridge Regression MAE: {ridge_mae}")
print(f"KNN Regression MAE: {knn_mae}")

# Best model conclusion
best_mae = min(linear_mae, ridge_mae, knn_mae)
if best_mae == knn_mae:
    print("KNN Regression is the best model with the lowest MAE.")
else:
    print("Another model performed best, check the MAE values.")


# In[ ]:




