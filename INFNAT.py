import numpy as np
import pandas as pd
import math
from scipy import stats, interpolate
from scipy.stats import theilslopes

# Configurable parameters
n_records = 200
n_features = 5

nan_injection_rate = 0.0001    
posinf_injection_rate = 0.00005 
neginf_injection_rate = 0.000025 

problematic_threshold = 0.002  

epsilon = np.finfo(float).eps

tol_multiplier = 1 / epsilon

pos_inf_replacement = np.nextafter(epsilon, np.inf)
neg_inf_replacement = -pos_inf_replacement

rolling_window_size = 5

trend_threshold_factor = 0.05
stat_test_alpha = 0.05  

# Sample contaminated dataframe for unit testing
np.random.seed(42)  
data = {"TARGET": np.random.normal(loc = 0, scale = 1, size = n_records)}
for i in range(1, n_features + 1):
    data[f"X{i}"] = np.random.normal(loc = 0, scale = 1, size = n_records)
df_model = pd.DataFrame(data)

# 1. Function to inject problems in a series
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

# 2. Initial problem statistics
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