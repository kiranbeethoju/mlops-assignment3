import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# Load California Housing dataset
print("Loading California Housing dataset...")
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2:.4f}")

# Save model
print("Saving model...")
joblib.dump(model, 'model.joblib')

# Save test data for later use
test_data = {
    'X_test': X_test,
    'y_test': y_test
}
joblib.dump(test_data, 'test_data.joblib')

print("Model saved as 'model.joblib'")
print("Test data saved as 'test_data.joblib'") 