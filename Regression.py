import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import os

# Verify working directory
print("Current working directory:", os.getcwd())

# Load dataset
data_path = r"C:\Ascencia\4. Applied AI\House-Price_ML\house_price_regression_dataset.csv"
df = pd.read_csv(data_path)
print(df.info())

# Separate features and target
X = df.drop("House_Price", axis=1)
y = df["House_Price"]

# # Identify categorical columns
# categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
# numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# # Convert numeric columns stored as objects to int
# for col in categorical_features:
#     if X[col].dtype == 'object' and all(x.isdigit() for x in X[col]):
#         X[col] = X[col].astype(int)
#         numeric_features.append(col)
# categorical_features = [col for col in categorical_features if col not in numeric_features]

# # Preprocessing pipeline
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("cat", OneHotEncoder(drop='first'), categorical_features)
#     ],
#     remainder='passthrough'
# )

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline: preprocessing + regression
model = Pipeline(steps=[
    # ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Fit model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("R Score:", r2)
print("MSE:", mse)

# Save the model
model_path = r"C:\Ascencia\4. Applied AI\House-Price_ML\model2.pkl"
joblib.dump(model, model_path)
if os.path.exists(model_path):
    print(f"Model saved successfully as {model_path}")
else:
    print("Failed to save model!")

# Optional: Plot actual vs predicted
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plot_min = min(y_test.min(), y_pred.min())
plot_max = max(y_test.max(), y_pred.max())
plt.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=2, color='navy', label='Perfect Prediction Line (y=x)')
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Ascencia\4. Applied AI\House-Price_ML\regression_actual_vs_predicted.png")
plt.show()
