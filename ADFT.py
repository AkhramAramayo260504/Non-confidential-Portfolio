import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# Sample data for unit testing
np.random.seed(42)
n_rows = 50
data = {
    'TARGET': np.random.randn(n_rows),  
    'X1': np.random.randn(n_rows),   
    'X2': np.random.randn(n_rows),   
    'X3': np.random.randn(n_rows),           
}

df_model = pd.DataFrame(data)
print(df_model.head())

# List for stationary and non-stationary features
stationary_features = []
non_stationary_features = []

# Augmented Dickey-Fuller Test for all features
for col in df_model.columns:
    series = df_model[col]
    try:
        result = adfuller(series)
        p_value = result[1]
        if p_value < 0.05:
            stationary_features.append(col)
        else:
            non_stationary_features.append(col)
    except Exception as e:
        print(f"Unsuitable feature for the test {col}: {e}")
        non_stationary_features.append(col)

# Filter data in stationary and non-stationary sets 
df_not_adf = df_model[non_stationary_features].copy()
df_model = df_model[stationary_features].copy()

# Print filtered sets
print("\n")
print("Non-stationary features:")
print(df_not_adf)
print("\n")
print("Stationary features:")
print(df_model)