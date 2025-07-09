import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import pacf
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy
from sklearn.model_selection import GridSearchCV

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