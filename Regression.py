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



import matplotlib.pyplot as plt
# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='darkorange', edgecolors='black', label='Actual vs Predicted Prices')

# Calculate min/max for the perfect prediction line
plot_min = min(y_test.min(), y_pred.min())
plot_max = max(y_test.max(), y_pred.max())

# Plot the line of perfect prediction (y = x)
plt.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=2, color='navy', label='Perfect Prediction Line (y=x)')

# Add labels and title
plt.title('Regression Model Performance: Actual vs Predicted House Prices (Synthetic Data)')
plt.xlabel('Actual House Price ($)')
plt.ylabel('Predicted House Price ($)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper left')

# Format axes labels to show dollar signs and commas
plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
# plt.ticklabel_format(style='plain', axis='both')
plt.tight_layout()

plt.show()
plt.savefig('regression_actual_vs_predicted.png')
