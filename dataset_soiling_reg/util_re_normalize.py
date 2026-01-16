import json

# Read normalization metadata
with open('dataset_soiling_reg/data/metadata/normalization.json', 'r') as f:
    norm_data = json.load(f)

norm_min = norm_data['norm_min']
norm_max = norm_data['norm_max']

# Normalized error values
normalized_mse = 0.030471
normalized_rmse = 0.174559

# Normalization range
value_range = norm_max - norm_min

# Convert errors to actual scale (CORRECT)
actual_rmse = normalized_rmse * value_range
actual_mse = normalized_mse * (value_range ** 2)

print(f"Actual RMSE: {actual_rmse:.6f}")
print(f"Actual MSE: {actual_mse:.6f}")
