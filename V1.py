####################################################
###### Importation of Libraries and Functions ######
####################################################

import numpy as np
import pandas as pd
import math
from scipy import stats, interpolate
from scipy.stats import theilslopes
from statsmodels.tsa.stattools import adfuller
from sklearn.feature_selection import mutual_info_regression
from itertools import combinations
from statsmodels.tsa.stattools import pacf
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy
from sklearn.model_selection import GridSearchCV
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, LinearRegression
import itertools
import warnings
from sklearn.neighbors import NearestNeighbors
from scipy.signal import argrelextrema
from scipy.spatial.distance import cdist
from typing import Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.exceptions import DataConversionWarning

# Pre section parameter
ARATTONE = 15

##########################################
###### Sample Data for Unit Testing ######
##########################################

# 1. Sample Data Parameters
n_records = 12500
n_features = 10
nan_injection_rate = 0.0001    
posinf_injection_rate = 0.00005 
neginf_injection_rate = 0.000025 

# 2. Sample DataFrame creation
np.random.seed(42)  
data = {"TARGET": np.random.normal(loc = 0, scale = 1, size = n_records)}
for i in range(1, n_features + 1):
    data[f"X{i}"] = np.random.normal(loc = 0, scale = 1, size = n_records)
df_model = pd.DataFrame(data)

# 3. Function to inject problems in a series
def inject_problems(series, nan_frac, posinf_frac, neginf_frac):
    series = series.copy()
    n = len(series)
    nan_indices = np.random.choice(n, int(n * nan_frac), replace = False)
    posinf_indices = np.random.choice(n, int(n * posinf_frac), replace = False)
    neginf_indices = np.random.choice(n, int(n * neginf_frac), replace = False)
    series.iloc[nan_indices] = np.nan
    series.iloc[posinf_indices] = np.inf
    series.iloc[neginf_indices] = -np.inf
    return series

for col in df_model.columns:
    df_model[col] = inject_problems(df_model[col],
                                    nan_injection_rate,
                                    posinf_injection_rate,
                                    neginf_injection_rate)

##############################################
###### Infinite & Null Values Treatment ######
##############################################

# Configurable Parameters
problematic_threshold = 0.002  
epsilon = np.finfo(float).eps
tol_multiplier = 1 / epsilon
pos_inf_replacement = np.nextafter(epsilon, np.inf)
neg_inf_replacement = -pos_inf_replacement
rolling_window_size = 5
trend_threshold_factor = 0.05
stat_test_alpha = 0.05 

# Initial problem statistics
print("Step 2: Initial Problem Statistics:")
for col in df_model.columns:
    total = len(df_model)
    n_nan = df_model[col].isna().sum()
    n_posinf = (df_model[col] == np.inf).sum()
    n_neginf = (df_model[col] == -np.inf).sum()
    print(f"{col}: NaN={n_nan} ({n_nan/total:.2%}), +INF={n_posinf} ({n_posinf/total:.2%}), -INF={n_neginf} ({n_neginf/total:.2%})")

# Problematic Values Identification
df_treated = df_model.copy()
problematic_proportions = {}

for col in df_treated.columns:
    total = len(df_treated)
    p_nan = df_treated[col].isna().sum() / total
    p_posinf = (df_treated[col] == np.inf).sum() / total
    p_neginf = (df_treated[col] == -np.inf).sum() / total
    p_total = p_nan + p_posinf + p_neginf
    problematic_proportions[col] = p_total

print("\nStep 4: Problem Proportions per Column:")
for col, prop in problematic_proportions.items():
    print(f"{col}: {prop:.2%}")

# Removal of extremely contaminated values
cols_to_drop = [col for col, prop in problematic_proportions.items() if prop >= problematic_threshold]
print(f"\nStep 5: Columns to drop (threshold >= {problematic_threshold:.2%}):")
print(cols_to_drop)
df_treated.drop(columns = cols_to_drop, inplace = True)
print("Dataframe shape after removal:", df_treated.shape)

# Near-Infinite Treatment and Inf Replacement
def treat_near_infinities(series, tol):
    condition = series.abs() >= tol
    series.loc[condition & (series > 0)] = np.inf
    series.loc[condition & (series < 0)] = -np.inf
    return series

for col in df_treated.columns:
    df_treated[col] = treat_near_infinities(df_treated[col], tol_multiplier)

df_treated.replace(np.inf, pos_inf_replacement, inplace = True)
df_treated.replace(-np.inf, neg_inf_replacement, inplace = True)

print("\nStep 6: Summary after infinity replacement:")
for col in df_treated.columns:
    total = len(df_treated)
    n_nan = df_treated[col].isna().sum()
    n_pos = (df_treated[col] == pos_inf_replacement).sum()
    n_neg = (df_treated[col] == neg_inf_replacement).sum()
    print(f"{col}: NaN={n_nan}, +INF replacements={n_pos}, -INF replacements={n_neg}")

# NaN Imputation with Rolling Median
def impute_with_rolling_median(series, window):
    series_imputed = series.copy()
    for i in range(len(series_imputed)):
        if pd.isna(series_imputed.iat[i]):
            start = max(0, i - window)
            valid_vals = series_imputed.iloc[start:i][~series_imputed.iloc[start:i].isna()]
            series_imputed.iat[i] = valid_vals.median() if not valid_vals.empty else 0.0
    return series_imputed

for col in df_treated.columns:
    df_treated[col] = impute_with_rolling_median(df_treated[col], rolling_window_size)

print("\nStep 7: Post-Imputation NaN Summary:")
for col in df_treated.columns:
    n_nan = df_treated[col].isna().sum()
    print(f"{col}: Remaining NaN = {n_nan}")

# Verification.1.0: Distribution Analysis
print("\nStep 8: Distribution Comparison:")
for col in df_model.columns:
    if col in df_treated.columns:
        original_series = df_model[col].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        treated_series = df_treated[col]
        
        original_mean = original_series.mean()
        original_std = original_series.std()
        treated_mean = treated_series.mean()
        treated_std = treated_series.std()
        
        print(f"{col} - Before: μ={original_mean:.4f}, σ={original_std:.4f} | After: μ={treated_mean:.4f}, σ={treated_std:.4f}")
        
        t_stat, p_value = stats.ttest_ind(original_series.dropna(), treated_series)
        print(f"  t-test p-value: {p_value:.4f}")
        
        f_stat = (original_std**2) / (treated_std**2) if treated_std != 0 else np.inf
        df_num = len(original_series.dropna()) - 1
        df_den = len(treated_series) - 1
        f_p_value = 1 - stats.f.cdf(f_stat, df_num, df_den)
        print(f"  F-test p-value: {f_p_value:.4f}")
    else:
        print(f"{col} was removed")

# Verification.2.0: Trend Analysis (Theil-Sen Regression)
def calculate_robust_trend(series):
    x = np.arange(len(series))
    slope, intercept, _, _ = theilslopes(series, x)
    predicted = intercept + slope * x
    residuals = series - predicted
    std_error = np.std(residuals)
    return slope, std_error

trend_diff_columns = []
print("\nStep 9: Trend Analysis Results:")
for col in df_treated.columns:
    original_series = df_model[col].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    treated_series = df_treated[col]
    
    slope_before, stderr_before = calculate_robust_trend(original_series)
    slope_after, stderr_after = calculate_robust_trend(treated_series)
    
    slope_diff = abs(slope_before - slope_after)
    trend_threshold = trend_threshold_factor * ((stderr_before + stderr_after) / 2)
    
    print(f"{col}: Slope before = {slope_before:.6f}, Slope after = {slope_after:.6f}")
    print(f"  Difference = {slope_diff:.6f}, Threshold = {trend_threshold:.6f}")
    
    if slope_diff >= trend_threshold:
        trend_diff_columns.append(col)
        print(f"  Significant trend difference detected in {col}")

# Hybrid Interpolation for Problem Points
def apply_hybrid_interpolation(series, original_series):
    problematic_idx = original_series.index[
        original_series.isna() | (original_series == np.inf) | (original_series == -np.inf)
    ]
    if len(problematic_idx) == 0:
        return series
    
    # 1. Linear interpolation
    valid_series = series.drop(problematic_idx)
    linear_imputed = series.copy()
    linear_imputed.loc[problematic_idx] = np.interp(problematic_idx, valid_series.index, valid_series)
    
    # 2. Spline interpolation if possible
    if len(valid_series) > 3:
        cs = interpolate.CubicSpline(valid_series.index, valid_series.values)
        spline_imputed = series.copy()
        spline_imputed.loc[problematic_idx] = cs(problematic_idx)
    else:
        spline_imputed = linear_imputed.copy()
    
    # 3. Select best method
    reference = series.median()
    sse_linear = np.nansum((linear_imputed.loc[problematic_idx] - reference) ** 2)
    sse_spline = np.nansum((spline_imputed.loc[problematic_idx] - reference) ** 2)
    
    return linear_imputed if sse_linear < sse_spline else spline_imputed

for col in trend_diff_columns:
    print(f"\nStep 10: Applying hybrid interpolation to {col}")
    df_treated[col] = apply_hybrid_interpolation(df_treated[col], df_model[col])

# Final Verification
print("\nStep 11: Final Verification:")
for col in df_treated.columns:
    if col in df_model.columns:
        original_series = df_model[col].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        treated_series = df_treated[col]
        
        slope_before, stderr_before = calculate_robust_trend(original_series)
        slope_after, stderr_after = calculate_robust_trend(treated_series)
        
        slope_diff = abs(slope_before - slope_after)
        trend_threshold = trend_threshold_factor * ((stderr_before + stderr_after) / 2)
        
        print(f"{col}: Slope difference = {slope_diff:.6f}, Threshold = {trend_threshold:.6f}")
        if slope_diff >= trend_threshold:
            print(f"  WARNING: Significant trend remains in {col}")
        else:
            print(f"  Trend acceptable in {col}")
    else:
        print(f"{col} was removed")

# Final Column Removal for non-treatable features
final_removal_list = []
for col in df_treated.columns:
    original_series = df_model[col].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    treated_series = df_treated[col]
    
    slope_before, stderr_before = calculate_robust_trend(original_series)
    slope_after, stderr_after = calculate_robust_trend(treated_series)
    
    slope_diff = abs(slope_before - slope_after)
    trend_threshold = trend_threshold_factor * ((stderr_before + stderr_after) / 2)
    
    if slope_diff >= trend_threshold:
        final_removal_list.append(col)
        print(f"Step 12: Removing {col} (slope difference: {slope_diff:.6f} >= {trend_threshold:.6f})")

df_treated.drop(columns = final_removal_list, inplace = True)
df_model = df_treated
print("\nStep 12: Final dataframe shape:", df_model.shape)
print(df_model)

##########################################
###### Augmented Dickey-Fuller Test ######
##########################################

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

#################################################
###### Mutual Information Elimination Test ######
#################################################

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

############################################
###### Transfer Entropy for Causality ######
############################################

# Configurable parameters
max_lag_target = 10  
max_lag_predictors = 10  
bandwidth_values = np.linspace(0.1, 2.0, 20)  
grid_points_values = [5, 10, 15, 20, 25]  

# ACPF for the target
def optimal_lag_acpf(series, max_lag = max_lag_target):
    pacf_values = pacf(series, nlags = max_lag)
    conf_interval = 1.96 / np.sqrt(len(series)) 

    for lag in range(1, max_lag + 1): 
        if abs(pacf_values[lag]) > conf_interval:  
            return lag
    return 1

target_series = df_model['TARGET']
optimal_lag_target = optimal_lag_acpf(target_series)  
print(f"\nOptimal lag for the target: {optimal_lag_target}")

# ACPF for the predictors
def optimal_lags_for_predictors(df_model, target_col, max_lag = max_lag_predictors):
    lags = {} 
    for col in df_model.columns:
        if col != target_col:
            lags[col] = optimal_lag_acpf(df_model[col], max_lag)
    return lags

lags_for_predictors = optimal_lags_for_predictors(df_model, 'TARGET')  
print("\nOptimal lags for predictors:")
print(lags_for_predictors)

# Marginal Shannon Entropy of the target using optimized KDE
def optimize_kde_params(series):
    grid_search = GridSearchCV(KernelDensity(kernel = 'gaussian'), 
                               {'bandwidth': bandwidth_values}, cv = 5)
    grid_search.fit(series.values.reshape(-1, 1))      
    optimal_bandwidth = grid_search.best_params_['bandwidth']

    best_entropy = np.inf
    best_grid_points = None
    for grid_points in grid_points_values:
        entropy_value = kde_entropy(series, bandwidth = optimal_bandwidth, 
                                    grid_points = grid_points)
        if entropy_value < best_entropy:
            best_entropy = entropy_value
            best_grid_points = grid_points

    return optimal_bandwidth, best_grid_points, best_entropy

def kde_entropy(series, bandwidth, grid_points):
    kde = KernelDensity(kernel = 'gaussian', bandwidth = bandwidth)
    series = series.values.reshape(-1, 1)
    kde.fit(series)
    grid = np.linspace(min(series), max(series), grid_points).reshape(-1, 1)
    log_dens = kde.score_samples(grid)
    density = np.exp(log_dens)
    density /= np.sum(density)
    return entropy(density)

bw_marginal, gp_marginal, entropy_target = optimize_kde_params(df_model['TARGET'])
print(f"\nShannon Entropy for TARGET: {entropy_target:.4f}")
print(f"Optimal bandwidth: {bw_marginal:.3f}, Optimal grid points: {gp_marginal}")

# Conditional Shannon Entropy of the target given each predictors using optimized KDE
def optimize_kde_params_conditional(series_target, series_predictor):
    best_entropy = np.inf
    best_bandwidth = None
    best_grid_points = None

    for bandwidth in bandwidth_values:
        for grid_points in grid_points_values:
            ce = kde_conditional_entropy(series_target, series_predictor, 
                                         bandwidth, grid_points)
            if ce < best_entropy:
                best_entropy = ce
                best_bandwidth = bandwidth
                best_grid_points = grid_points

    return best_bandwidth, best_grid_points, best_entropy

def conditional_entropy_for_each_predictor(df_model, target_col):
    conditional_entropies = {}
    optimal_params = {}

    for col in df_model.columns:
        if col != target_col:
            bw, gp, ce = optimize_kde_params_conditional(df_model[target_col], 
                                                         df_model[col])
            conditional_entropies[col] = ce
            optimal_params[col] = (bw, gp)

    return conditional_entropies, optimal_params

def kde_conditional_entropy(target_series, predictor_series, bandwidth, grid_points):
    combined = pd.concat([target_series, predictor_series], axis = 1).dropna()
    y = combined.iloc[:, 0].values.reshape(-1, 1)
    x = combined.iloc[:, 1].values.reshape(-1, 1)
    xy = np.hstack((y, x))

    kde_joint = KernelDensity(kernel = 'gaussian', bandwidth = bandwidth).fit(xy)
    kde_x = KernelDensity(kernel = 'gaussian', bandwidth=bandwidth).fit(x)

    y_grid = np.linspace(y.min(), y.max(), grid_points)
    x_grid = np.linspace(x.min(), x.max(), grid_points)
    grid = np.array(np.meshgrid(y_grid, x_grid)).T.reshape(-1, 2)

    joint_density = np.exp(kde_joint.score_samples(grid))
    x_density = np.exp(kde_x.score_samples(grid[:, 1].reshape(-1, 1)))

    joint_density /= joint_density.sum()
    x_density /= x_density.sum()

    return entropy(joint_density, x_density)

conditional_entropies, optimal_params = conditional_entropy_for_each_predictor(df_model, 
                                                                               'TARGET')
print("\nConditional Shannon Entropies (TARGET | Xi):")
for predictor, ent in conditional_entropies.items():
    bw, gp = optimal_params[predictor]
    print(f"{predictor}: Entropy = {ent:.4f}, Bandwidth = {bw:.3f}, Grid Points = {gp}")

# Normalized Transfer Entropy for the predictors
transfer_entropies = {}
for predictor, h_y_given_x in conditional_entropies.items():
    te_value = entropy_target - h_y_given_x  
    transfer_entropies[predictor] = te_value

max_te = max(transfer_entropies.values())
normalized_te = {k: v / max_te for k, v in transfer_entropies.items()}

# Transfer Entropy Score Rank
sorted_te = sorted(normalized_te.items(), key = lambda x: x[1], reverse = True)
te_rank = pd.DataFrame(sorted_te, columns = ['Variable', 'Normalized_Transfer_Entropy'])
te_rank['Rank'] = te_rank['Normalized_Transfer_Entropy'].rank(
    ascending = False, method = 'dense').astype(int)
print("\nNormalized Transfer Entropy Rank:")
print(te_rank)

###########################################################
###### Causal Bayesian Networks for Robust Causality ######
###########################################################

# Data standardization
def standardize_data(df):
    scaler = StandardScaler()
    df_std = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
    return df_std

df_std = standardize_data(df_model)

# Configurable parameters

# 1. General parameters
alpha = 0.05                        # Significance threshold for independence test
delta = 1.0                         # Intervention value for do(X)
# 2. PC algorithm parameters
max_cond_subset = 5                 # Maximum size of conditioning subset
# 3. NOTEARS parameters
lasso_cv_folds = 10                 # Number of folds in LassoCV
lasso_threshold = 0.05              # Minimun threshold for coefficient
# 4. RandomForest parameters
rf_n_estimators = 100               # Number of trees in RandomForest
rf_max_depth = None                 # Maximum depth in RandomForest
rf_random_state = 42                # RandomForest seed
# 5. GradientBoosting parameters
gb_n_estimators = 100               # Number of estimators in GradientBoosting
gb_random_state = 42                # GradientBoosting seed
# 6. KFold parameters
kf_splits = 10                      # Number of divisions in KFold
kf_shuffle = False                  # Random or not in KFold
kf_random_state = None              # KFold seed
# 7. Additional parameters in PC 
target_edge_weight = 0.05           # Weight of relation from X to TARGET
# 8. Consensus parameters
consensus_threshold_factor = 0.5    # Consensus threshold factor
# 9. IQR parameters 
iqr_multiplier = 1                  # IQR multiplier for outliers


# Causal structure definition by the integration of multiple algorithms
def discover_causal_structure(data, alpha = alpha, method = 'consensus'):
    n_vars = data.shape[1]
    var_names = data.columns
    structures = []
    
    # 1. PC Method based on conditional independencies
    G_pc = nx.DiGraph()
    skeleton = nx.Graph()

    for i in range(n_vars):
        skeleton.add_node(var_names[i])
    
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            skeleton.add_edge(var_names[i], var_names[j])

    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if j <= i:
                continue
            if not skeleton.has_edge(vi, vj):
                continue

            for subset_size in range(0, min(n_vars - 2, max_cond_subset)):
                other_nodes = [vk for k, vk in enumerate(var_names) if k != i 
                               and k != j]
                
                for conditioning_set in itertools.combinations(other_nodes,
                                                                subset_size):
                    X_cond = data[list(conditioning_set)]
                    
                    if len(conditioning_set) > 0:
                        model_i = LinearRegression().fit(X_cond, data[vi])
                        residual_i = data[vi] - model_i.predict(X_cond)
                        
                        model_j = LinearRegression().fit(X_cond, data[vj])
                        residual_j = data[vj] - model_j.predict(X_cond)
                        
                        partial_corr = abs(np.corrcoef(residual_i, residual_j)[0, 1])
                        indep = partial_corr < alpha
                    else:
                        corr = abs(np.corrcoef(data[vi], data[vj])[0, 1])
                        indep = corr < alpha
                    
                    if indep and skeleton.has_edge(vi, vj):
                        skeleton.remove_edge(vi, vj)
                        break
    
    G_pc = nx.DiGraph(skeleton)
    
    if 'TARGET' in var_names:
        target = 'TARGET'
        
        for pred in list(G_pc.neighbors(target)):
            if G_pc.has_edge(target, pred):
                G_pc.remove_edge(target, pred)
            if not G_pc.has_edge(pred, target):
                G_pc.add_edge(pred, target, weight = target_edge_weight)
        
        target_corrs = {}
        for var in var_names:
            if var != target:
                corr = abs(np.corrcoef(data[var], data[target])[0, 1])
                target_corrs[var] = corr
        
        for i, vi in enumerate(var_names):
            if vi == target:
                continue
            for j, vj in enumerate(var_names):
                if vj == target or j <= i:
                    continue
                
                if G_pc.has_edge(vi, vj) and G_pc.has_edge(vj, vi):
                    if target_corrs.get(vi, 0) > target_corrs.get(vj, 0):
                        G_pc.remove_edge(vj, vi)
                    else:
                        G_pc.remove_edge(vi, vj)
    
    structures.append(G_pc)
    
    # 2. NOTEARS Method of approximation
    G_notears = nx.DiGraph()

    for j, vj in enumerate(var_names):
        X = data.drop(columns = [vj])
        y = data[vj]
        
        model = LassoCV(cv = lasso_cv_folds, random_state = rf_random_state)
        model.fit(X, y)
        
        for i, vi in enumerate(X.columns):
            coef = model.coef_[i]
            if abs(coef) > lasso_threshold:
                G_notears.add_edge(vi, vj, weight = abs(coef))
    
    structures.append(G_notears)
    
    # 3. Consensus Structure Method
    G_consensus = nx.DiGraph()
    edge_counts = {}

    for G in structures:
        for u, v in G.edges():
            edge = (u, v)
            if edge not in edge_counts:
                edge_counts[edge] = 0
            edge_counts[edge] += 1
    
    threshold = max(1, int(len(structures) * consensus_threshold_factor))

    for edge, count in edge_counts.items():
        if count >= threshold:
            u, v = edge
            G_consensus.add_edge(u, v, weight=count / len(structures))
    if len(G_consensus.edges()) == 0:
        G_consensus = G_pc
    
    return G_consensus

# Cofounders (Z) identification for each causal relation
def identify_confounders(G, target, treatment):
    descendants = set()
    if treatment in G:
        descendants = nx.descendants(G, treatment)
        
    treatment_parents = set()
    if treatment in G:
        treatment_parents = set(G.predecessors(treatment))
    
    target_parents = set()
    if target in G:
        target_parents = set(G.predecessors(target))
    
    confounders = list((treatment_parents | target_parents) - 
                       {treatment, target} - descendants)
    
    return confounders

# P(Y|do(X)) estimation for each predictor (X)
def estimate_causal_effect(data, treatment, outcome, confounders):
    causal_estimates = []
    
    if not confounders:
        X = data[[treatment]].values
        Y = data[outcome].values
        
        kf = KFold(n_splits = kf_splits, shuffle = kf_shuffle, 
                   random_state = kf_random_state)
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            models = [
                RandomForestRegressor(n_estimators = rf_n_estimators, 
                                      max_depth = rf_max_depth, 
                                      random_state = rf_random_state),
                GradientBoostingRegressor(n_estimators = gb_n_estimators, 
                                          random_state = gb_random_state)
            ]
            
            model_effects = []
            for model in models:
                model.fit(X_train, Y_train)
                
                # Note: do(X) intervention
                X_test_base = X_test.copy()
                X_test_intervention = X_test.copy() + delta
                
                effect = np.mean(model.predict(X_test_intervention) - 
                                 model.predict(X_test_base))
                model_effects.append(effect)
            
            fold_effect = np.mean(model_effects)
            causal_estimates.append(fold_effect)
        
        final_effect = np.mean(causal_estimates)
        return final_effect
    
    else:
        kf = KFold(n_splits = kf_splits, shuffle = kf_shuffle, 
                   random_state = kf_random_state)
        
        features = confounders + [treatment]
        X = data[features].values
        Y = data[outcome].values
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            models = [
                RandomForestRegressor(n_estimators = rf_n_estimators, 
                                      random_state = rf_random_state),
                GradientBoostingRegressor(n_estimators = gb_n_estimators, 
                                          random_state = gb_random_state)
            ]
            
            for model in models:
                model.fit(X_train, Y_train)
                
                X_test_base = X_test.copy()
                X_test_intervention = X_test.copy()
                
                treatment_idx = len(confounders)
                X_test_intervention[:, treatment_idx] += delta
                
                Y_base = model.predict(X_test_base)
                Y_intervention = model.predict(X_test_intervention)
                
                effect = np.mean(Y_intervention - Y_base)
                causal_estimates.append(effect)
        
        # Note: Mean of all estimations 
        if causal_estimates:
            q1 = np.percentile(causal_estimates, 25)
            q3 = np.percentile(causal_estimates, 75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            
            filtered_estimates = [e for e in causal_estimates 
                                  if lower_bound <= e <= upper_bound]
            
            if filtered_estimates:
                return np.mean(filtered_estimates)
            return np.mean(causal_estimates)
        
        return 0.0

# Main function for causality analysis
def main():
    causal_structure = discover_causal_structure(df_std)
    
    predictors = [col for col in df_model.columns if col != 'TARGET']
    causal_effects = {}
    confounders_dict = {}
    
    for predictor in predictors:
        # Note: Cofounders (Z) for each predictor
        confounders = identify_confounders(causal_structure, 'TARGET', predictor)
        confounders_dict[predictor] = confounders
        
        # Note: P(Y|do(X)) for each predictor
        effect = estimate_causal_effect(df_std, predictor, 'TARGET', confounders)
        causal_effects[predictor] = effect
    
    # Dataframe of causality for each predictor
    caus_cbn = pd.DataFrame({
        'Variable': list(causal_effects.keys()),
        'Causal_Effect': list(causal_effects.values())
    })
    
    # Order for absolute magnitude
    caus_cbn['Abs_Effect'] = np.abs(caus_cbn['Causal_Effect'])
    caus_cbn = caus_cbn.sort_values('Abs_Effect', 
                                    ascending = False).drop('Abs_Effect', axis = 1)
    
    return caus_cbn

# Complete analysis execution, standarize causality [0,1] and add rank
caus_cbn = main()
caus_cbn['Causal_Effect'] = caus_cbn['Causal_Effect'].abs()
max_value = caus_cbn['Causal_Effect'].max()
caus_cbn['Causal_Effect'] = caus_cbn['Causal_Effect'] / max_value
caus_cbn['Rank'] = caus_cbn['Causal_Effect'].rank(ascending = False, method = 'min')
cbn_rank = caus_cbn
print(cbn_rank)

#####################################################
###### Convergent Cross Mappings for Causality ######
#####################################################

# Configurable parameters
MAX_LAG = 50
FNN_MAX_DIM = 50
ALPHA = 0.05
FNN_THRESHOLD = 0.05
RTOL = 15.0
ATOL = 2.0
K_NEIGHBORS = 50
P_DECAY = 0.5
POINT_INDEX = -1
USE_ALL_POINTS = True
MI_NEIGHBORS = 50

# Time delay computation
# 1. ACPF
def compute_time_delay(series: np.ndarray, max_lag: int) -> int:
    n = len(series)
    pacf_vals, confint = pacf(series, nlags=max_lag, alpha=ALPHA)
    threshold = 1.96/np.sqrt(n)
    pacf_lag = next((i for i in range(1, len(pacf_vals)) 
                    if abs(pacf_vals[i]) < threshold), max_lag)
    
    # 2. Mutual Information
    lags = range(1, max_lag+1)
    mi_scores = [mutual_info_regression(series[:-tau].reshape(-1,1), 
                                      series[tau:])[0] 
                for tau in lags]
    
    window_size = max(3, max_lag//10)
    smoothed = pd.Series(mi_scores).rolling(window_size, 
                     center=True).mean().values
    minima = argrelextrema(smoothed, np.less)[0]
    ami_lag = minima[0]+1 if len(minima)>0 else max_lag
    
    # 3. Methodlogies integration
    integrated_lag = int(round(0.7*ami_lag + 0.3*pacf_lag))
    return max(1, min(integrated_lag, max_lag))

# Embedding dimension computation
def determine_embedding_dimension(series: np.ndarray, tau: int) -> int:
    n = len(series)
    for d in range(1, FNN_MAX_DIM+1):
        if n < d*tau+1:
            return max(1, d-1)
        
        emb_d = takens_embedding(series, d, tau)
        emb_d1 = takens_embedding(series, d+1, tau)
        
        if len(emb_d1) < 2:
            return d
        
        # 1. False Nearest Neighboors (FNN)
        nbrs = NearestNeighbors(n_neighbors=2).fit(emb_d)
        distances, indices = nbrs.kneighbors(emb_d)
        
        false_neighbors = 0
        valid_points = 0
        
        for i in range(len(emb_d)):
            if i >= len(emb_d1) or indices[i,1] >= len(emb_d1):
                continue
                
            valid_points += 1
            R_d = distances[i,1]
            delta = emb_d1[i] - emb_d1[indices[i,1]]
            R_d1 = np.linalg.norm(delta)
            
            if (R_d1/R_d > RTOL) or (abs(R_d1-R_d)/R_d > ATOL):
                false_neighbors += 1
        
        if valid_points == 0:
            return d
        
        fnn_ratio = false_neighbors/valid_points
        if fnn_ratio < FNN_THRESHOLD:
            return d
            
    return FNN_MAX_DIM

# Takens Embedding for TARGET
def takens_embedding(series: np.ndarray, m: int, tau: int) -> np.ndarray:
    n = len(series)
    L = n - (m-1)*tau
    if L <= 0:
        raise ValueError(f"m={m} and tau={tau} produce L={L} (invalid)")
    return np.array([series[i:i+(m-1)*tau+1:tau] for i in range(L)])

# Attractor for Takens Embedding
def analyze_attractor(embeddings: np.ndarray) -> Dict[str, Any]:
    distances = cdist(embeddings, embeddings)
    mask = np.ones_like(distances, dtype=bool)
    np.fill_diagonal(mask, False)
    masked_distances = distances[mask].flatten()
    
    if len(masked_distances) == 0:
        return {
            'dimensionality': embeddings.shape[1],
            'n_points': embeddings.shape[0],
            'distance_stats': {
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'q90': np.nan},
            'determinism_ratio': np.nan,
            'recurrence_plot': distances}
    
    max_dist = np.max(masked_distances)
    return {
        'dimensionality': embeddings.shape[1],
        'n_points': embeddings.shape[0],
        'distance_stats': {
            'mean': np.mean(masked_distances),
            'std': np.std(masked_distances),
            'min': np.min(masked_distances),
            'max': max_dist,
            'q90': np.quantile(masked_distances, 0.9)},
        'determinism_ratio': np.mean(masked_distances < 0.1*max_dist),
        'recurrence_plot': distances}

# Main Function of the Analysis
def main_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    target_series = df['TARGET'].values
    
    try:
        tau = compute_time_delay(target_series, MAX_LAG)
        m = determine_embedding_dimension(target_series, tau)
        embeddings = takens_embedding(target_series, m, tau)
    except Exception as e:
        raise RuntimeError(f"Embedding construction error: {str(e)}") from e
    
    attractor_metrics = analyze_attractor(embeddings)
    
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS+1).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    weights = np.exp(-(distances[:,1:]**2)/(2*P_DECAY**2))
    weights = weights/weights.sum(axis=1)[:, np.newaxis]
    
    analysis_results = {
        'indices': indices[:,1:],
        'weights': weights,
        'coordinates': embeddings,
        'local_entropy': -np.sum(weights*np.log(weights+1e-12), axis=1)}
    
    return {
        'embedding_parameters': {'m': m, 'tau': tau},
        'attractor_metrics': attractor_metrics,
        'point_analysis': analysis_results,
        'recommendations': {
            'suggest_multipoint': len(embeddings)<1000 and not USE_ALL_POINTS,
            'optimal_parameters': {
                'suggested_k': max(3, int(np.sqrt(len(embeddings)))),
                'suggested_p': attractor_metrics['distance_stats']['mean']/2}}}

# Execution of Analysis
if __name__ == "__main__":    
    # 2. Execution
    try:
        results = main_analysis(df_model)
        
        print("\n=== Embedding Parameters ===")
        print(f"Dimension (m): {results['embedding_parameters']['m']}")
        print(f"Delay (τ): {results['embedding_parameters']['tau']}")
        
        print("\n=== Attractor Metrics ===")
        stats = results['attractor_metrics']['distance_stats']
        print(f"Data points: {results['attractor_metrics']['n_points']}")
        print(f"Mean distance: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"Distance range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"Determinism ratio: {results['attractor_metrics']['determinism_ratio']:.2%}")
        
        if results['recommendations']['suggest_multipoint']:
            print("\nRECOMMENDATION: Enable USE_ALL_POINTS for full analysis")

        target_series = df_model['TARGET'].values
        m = results['embedding_parameters']['m']
        tau = results['embedding_parameters']['tau']
        
        try:
            embeddings = takens_embedding(target_series, m, tau)
            plt.figure(1, figsize=(10,6))
            plt.imshow(results['attractor_metrics']['recurrence_plot'], 
                     cmap='binary', origin='lower', aspect='auto')
            plt.title(f"Recurrence Plot (m={m}, τ={tau})")
            plt.colorbar(label='Distance')
            
            plt.figure(2, figsize=(10,6))
            if m == 1:
                plt.plot(embeddings, 'b.-', alpha=0.5)
                plt.title(f"1D Attractor (τ={tau})")
            elif m == 2:
                plt.scatter(embeddings[:,0], embeddings[:,1], 
                           c=np.arange(len(embeddings)), cmap='viridis', alpha=0.5)
                plt.title(f"2D Attractor (m={m}, τ={tau})")
                plt.colorbar(label='Time index')
            else:
                fig = plt.figure(2, figsize=(10,8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(embeddings[:,0], embeddings[:,1], embeddings[:,2], 
                          c=np.arange(len(embeddings)), cmap='viridis', alpha=0.5)
                ax.set_title(f"3D Projection of {m}D Attractor")
                fig.colorbar(ax.collections[0], label='Time index')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"\nVisualization error: {str(e)}")
            
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Troubleshooting: Reduce MAX_LAG/FNN_MAX_DIM or check data variance")

# Causality Analysis
if USE_ALL_POINTS and 'indices' in results['point_analysis']:
    print("\n=== Causality Analysis ===")
    m = results['embedding_parameters']['m']
    tau = results['embedding_parameters']['tau']
    L = len(results['point_analysis']['coordinates'])
    start_idx = (m-1)*tau
    
    causality_results = {}
    all_neighbors = results['point_analysis']['indices']
    all_weights = results['point_analysis']['weights']
    
    # 1. Reconstruction of predictor series from TARGET attractor
    for col in df_model.columns:
        if col == 'TARGET': continue
        
        X_true = df_model[col].values[start_idx:start_idx+L]
        X_neighbors = df_model[col].values[start_idx + all_neighbors]
        X_pred = np.sum(all_weights * X_neighbors, axis=1)
        
        # 2. Mutual information between reconstructed and real series 
        mi_score = mutual_info_regression(X_true.reshape(-1,1), 
                                 X_pred.reshape(-1,1),
                                 n_neighbors=MI_NEIGHBORS)[0]
        
        causality_results[col] = {
            'MI_Score': mi_score,
            'X_true': X_true,
            'X_pred': X_pred}
    
    # 3. Display the results
    print("\nCausality Scores:")
    for col, data in causality_results.items():
        print(f"{col}: {data['MI_Score']:.4f}")
    
    plt.figure(figsize=(10,6))
    for col, data in causality_results.items():
        plt.plot(data['X_true'], data['X_pred'], 'o', alpha=0.3, label=col)
    plt.plot([-3,3], [-3,3], 'k--', label='Reference')
    plt.xlabel('Real Values')
    plt.ylabel('Reconstructed Values')
    plt.legend()
    plt.show()


else:
    print("\nCausality Analysis requires USE_ALL_POINTS = True")

# CCM Causality Rank
if 'causality_results' in globals():  
    # 1. Mutual Information Normalization
    mi_scores = {col: data['MI_Score'] for col, data in causality_results.items()}
    max_mi = max(mi_scores.values()) if mi_scores else 1  
    
    # 2. Set frame
    ccm_rank = pd.DataFrame({
        'variable': mi_scores.keys(),
        'causality': [score/max_mi for score in mi_scores.values()]
    })
    
    # 3. Order and rank on causality score
    ccm_rank = ccm_rank.sort_values('causality', ascending=False)
    ccm_rank['rank'] = range(1, len(ccm_rank)+1)
    
    print("\nDataframe ccm_rank:")
    print(ccm_rank)

#####################################
###### Eschaton Selection Test ######
#####################################

# Reconfigouration of set
te_ranks = te_rank[["Variable", "Rank"]].rename(columns={"Rank": "TE_Rank"})
cbn_ranks = cbn_rank[["Variable", "Rank"]].rename(columns={"Rank": "CBN_Rank"})
ccm_ranks = ccm_rank[["variable", "rank"]].rename(columns={"variable": "Variable", "rank": "CCM_Rank"})

# Merge of the causality dataframes and ranks
merged_ranks = te_ranks.merge(cbn_ranks, on="Variable").merge(ccm_ranks, on="Variable")
merged_ranks["Sum_Ranks"] = merged_ranks["TE_Rank"] + merged_ranks["CBN_Rank"] + merged_ranks["CCM_Rank"]
total_vars = len(merged_ranks)
merged_ranks["Borda_Score"] = total_vars - merged_ranks["Sum_Ranks"]

# Eschaton ranking
eschaton_rank = merged_ranks[["Variable", "Borda_Score"]].copy()
eschaton_rank = eschaton_rank.sort_values("Borda_Score", ascending=True)
eschaton_rank["Final_Rank"] = eschaton_rank["Borda_Score"].rank(method="min", ascending=True).astype(int)
eschaton_rank.columns = ["Variable", "Borda Score", "Final Rank"]
print(eschaton_rank)

# Eschaton Selection Test
top_variables = eschaton_rank['Variable'].head(ARATTONE).tolist()
df_model = df_model[['TARGET'] + top_variables]
print(df_model)