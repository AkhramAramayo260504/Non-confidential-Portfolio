import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from statsmodels.tsa.stattools import pacf
from sklearn.feature_selection import mutual_info_regression
from scipy.signal import argrelextrema
from scipy.spatial.distance import cdist
from typing import Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings 
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)

# Sample Data for unit testing
np.random.seed(42)
data = {
    'TARGET': np.random.randn(20000),
    'X1': np.random.randn(20000),
    'X2': np.random.randn(20000),
    'X3': np.random.randn(20000)}
df_model = pd.DataFrame(data)

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