'''
import kagglehub

# Download latest version
path = kagglehub.dataset_download("prokshitha/home-value-insights")

print("Path to dataset files:", path)
'''

import pandas as pd
import numpy as np


df = pd.read_csv('house_price_regression_dataset.csv')

df[['Num_Bedrooms', 'Num_Bathrooms', 'Garage_Size', 'Neighborhood_Quality']] = df[['Num_Bedrooms', 'Num_Bathrooms', 'Garage_Size', 'Neighborhood_Quality']].astype(str)
df.info() 

X = df[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']]
y = df['House_Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # test_size == % of dataset take out to test set, random_state == random number generator for shuffing data

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
print("R Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

print("Coefficients:", model.coef_)    
print("Intercept:", model.intercept_) 

