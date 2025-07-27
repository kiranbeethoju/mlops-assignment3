import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

# Load the saved scikit-learn model
print("Loading scikit-learn model...")
sklearn_model = joblib.load('model.joblib')

# Load test data
test_data = joblib.load('test_data.joblib')
X_test = test_data['X_test']
y_test = test_data['y_test']

# Extract parameters from scikit-learn model
coef = sklearn_model.coef_
intercept = sklearn_model.intercept_

# Store unquantized parameters
unquant_params = {
    'coef': coef,
    'intercept': intercept
}
joblib.dump(unquant_params, 'unquant_params.joblib')

# Manual quantization to 8-bit unsigned integer
def quantize_to_uint8(data):
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Handle case where min_val == max_val
    if min_val == max_val:
        quantized = np.zeros_like(data, dtype=np.uint8)
        return quantized, min_val, max_val, 1.0
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    max_val = max_val + epsilon
    
    # Scale to 0-255 range
    scale = 255.0 / (max_val - min_val)
    quantized = np.clip(np.round((data - min_val) * scale), 0, 255).astype(np.uint8)
    
    return quantized, min_val, max_val, scale

def dequantize_from_uint8(quantized_data, min_val, max_val, scale):
    if min_val == max_val:
        return np.full_like(quantized_data, min_val, dtype=np.float32)
    dequantized = quantized_data.astype(np.float32) / scale + min_val
    return dequantized

print("Performing manual quantization...")
quantized_coef, coef_min, coef_max, coef_scale = quantize_to_uint8(coef)
quantized_intercept, intercept_min, intercept_max, intercept_scale = quantize_to_uint8(intercept)

# Store quantized parameters
quant_params = {
    'quantized_coef': quantized_coef,
    'quantized_intercept': quantized_intercept,
    'coef_min': coef_min,
    'coef_max': coef_max,
    'coef_scale': coef_scale,
    'intercept_min': intercept_min,
    'intercept_max': intercept_max,
    'intercept_scale': intercept_scale
}
joblib.dump(quant_params, 'quant_params.joblib')

# Create PyTorch model with dequantized weights
class QuantizedLinearModel(nn.Module):
    def __init__(self, input_size):
        super(QuantizedLinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return self.linear(x)

# Dequantize parameters
dequantized_coef = dequantize_from_uint8(quantized_coef, coef_min, coef_max, coef_scale)
dequantized_intercept = dequantize_from_uint8(quantized_intercept, intercept_min, intercept_max, intercept_scale)

# Create PyTorch model
pytorch_model = QuantizedLinearModel(X_test.shape[1])

# Set weights manually
with torch.no_grad():
    pytorch_model.linear.weight.data = torch.FloatTensor(dequantized_coef.reshape(1, -1))
    pytorch_model.linear.bias.data = torch.FloatTensor([float(dequantized_intercept)])

# Save PyTorch model
torch.save(pytorch_model.state_dict(), 'quantized_model.pth')

# Test original sklearn model
print("Testing original sklearn model...")
y_pred_original = sklearn_model.predict(X_test)
r2_original = r2_score(y_test, y_pred_original)

# Test quantized model
print("Testing quantized model...")
X_test_tensor = torch.FloatTensor(X_test)
pytorch_model.eval()
with torch.no_grad():
    y_pred_quantized = pytorch_model(X_test_tensor).numpy().flatten()
r2_quantized = r2_score(y_test, y_pred_quantized)

# Calculate file sizes
import os
unquant_size = os.path.getsize('unquant_params.joblib') / 1024  # KB
quant_size = os.path.getsize('quant_params.joblib') / 1024  # KB

print("\n" + "="*50)
print("COMPARISON RESULTS")
print("="*50)
print(f"Original Sklearn Model R2 Score: {r2_original:.4f}")
print(f"Quantized Model R2 Score: {r2_quantized:.4f}")
print(f"Model Size - unquant_params.joblib: {unquant_size:.2f} KB")
print(f"Model Size - quant_params.joblib: {quant_size:.2f} KB")
print(f"Size Reduction: {((unquant_size - quant_size) / unquant_size * 100):.2f}%")
print("="*50) 