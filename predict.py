import joblib
import numpy as np
from sklearn.metrics import r2_score

# Load the saved model
print("Loading model...")
model = joblib.load('model.joblib')

# Load test data
print("Loading test data...")
test_data = joblib.load('test_data.joblib')
X_test = test_data['X_test']
y_test = test_data['y_test']

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2:.4f}")

# Show some sample predictions
print("\nSample predictions:")
for i in range(5):
    print(f"Actual: {y_test[i]:.2f}, Predicted: {y_pred[i]:.2f}")

print("\nModel verification completed successfully!") 