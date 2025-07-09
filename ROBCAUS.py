import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, LinearRegression
import itertools
import warnings

warnings.filterwarnings('ignore')

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
print(df_model)

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