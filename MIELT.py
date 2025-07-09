import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from itertools import combinations

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

# Redundancy thershold
threshold = 0.5

# Predictors and TARGET definition
X = df_model.drop(columns = 'TARGET')
Y = df_model['TARGET']

# Mutual information between predictors and TARGET MI(Y;X)
mutual_information_with_target = pd.Series(
    mutual_info_regression(X, Y, discrete_features = False),
    index = X.columns
)

# Mutual information between predictors MI(X;X)
mutual_information_matrix = pd.DataFrame(
    index = X.columns, columns = X.columns, dtype = float
)
for col1, col2 in combinations(X.columns, 2):
    mi = mutual_info_regression(X[[col1]], X[col2], discrete_features = False)[0]
    mutual_information_matrix.loc[col1, col2] = mi
    mutual_information_matrix.loc[col2, col1] = mi
np.fill_diagonal(mutual_information_matrix.values, np.nan)

# M(Y;Y) as scaler for normalization [0,1]
mutual_information_scaler = mutual_info_regression(
    df_model[['TARGET']], Y, discrete_features = False
)[0]

# Normalization of MI(Y;X)
mutual_information_with_target = (
    mutual_information_with_target / mutual_information_scaler
).round(10)

# Normalization of MI(X;X)
mutual_information_matrix = (
    mutual_information_matrix / mutual_information_scaler
).round(10)

# Print normalized mutual information
print("\n")
print("Mutual Information with TARGET:")
print(mutual_information_with_target)
print("\n")
print("Mutual Information Matrix between predictors:")
print(mutual_information_matrix)
print("\n")

# Mutual Information Elimination Test
to_remove = set()
columns = X.columns.tolist()

for col1, col2 in combinations(columns, 2):
    mi_value = mutual_information_matrix.loc[col1, col2]
    if pd.notnull(mi_value) and mi_value > threshold:
        if mutual_information_with_target[col1] < mutual_information_with_target[col2]:
            to_remove.add(col1)
        else:
            to_remove.add(col2)

# Filtered data sets 
df_not_mi = df_model[list(to_remove)]
df_model = df_model[['TARGET'] + [
    col for col in X.columns if col not in to_remove]]

# Print filtered data sets
print("Non-redundant features:")
print(df_model.head())
print("\n")
print("Eliminated features (redundant):")
print(df_not_mi.head())