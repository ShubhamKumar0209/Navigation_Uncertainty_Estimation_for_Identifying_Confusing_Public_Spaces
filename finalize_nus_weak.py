import os
import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

NUS_WEAK_WEIGHTS = {
    "stopping_frequency": 0.2,
    "loop_backtrack": 0.2,
    "turning_angle_variance": 0.15,
    "local_hesitation": 0.15,
    "path_efficiency": 0.1,
    "speed_variance": 0.05,
    "waiting_time": 0.05,
    "interaction_change_rate": 0.1,
    "crowd_density": 0.0,
}

def main():
    assert abs(sum(NUS_WEAK_WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"
    
    input_files = glob.glob(os.path.join("processed_all", "*", "features_windows_with_graph.csv"))
    if not input_files:
        print("Error: No features_windows_with_graph.csv found in processed_all/*/")
        return
        
    print(f"Loading {len(input_files)} feature files...")
    dfs = []
    for f in input_files:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    
    # Save old for comparison
    df['old_NUS_weak_expert'] = df['NUS_weak_expert'].copy()
    
    # 1. Compute interaction_change_rate_norm per scene
    print("Computing interaction_change_rate_norm...")
    norm_dfs = []
    for scene_id, group in df.groupby('scene_id'):
        norm_group = group.copy()
        
        val = group['interaction_change_rate'].dropna()
        if len(val) == 0:
            median, iqr_safe = 0.0, 1.0
        else:
            median = np.median(val)
            q75, q25 = np.percentile(val, [75, 25])
            iqr = q75 - q25
            iqr_safe = iqr if iqr > 0.01 else 1.0
            
        raw_scaled = (group['interaction_change_rate'] - median) / iqr_safe
        norm_group['interaction_change_rate_norm'] = np.clip(raw_scaled, -5, 5)
        norm_dfs.append(norm_group)
        
    df = pd.concat(norm_dfs, ignore_index=True)
    
    # 2. Recompute NUS_weak_expert
    print("Recomputing NUS_weak_expert...")
    df['NUS_weak_expert'] = 0.0
    for feat, weight in NUS_WEAK_WEIGHTS.items():
        if weight > 0:
            df['NUS_weak_expert'] += df[feat + '_norm'] * weight
            
    # 3. Recompute NUS_weak_pca
    print("Recomputing NUS_weak_pca...")
    active_features = [f for f, w in NUS_WEAK_WEIGHTS.items() if w > 0]
    norm_cols = [f + '_norm' for f in active_features]
    
    valid_idx = df[norm_cols].dropna().index
    if len(valid_idx) > 0:
        pca = PCA(n_components=1)
        pca.fit(df.loc[valid_idx, norm_cols])
        loadings = pca.components_[0]
        
        if loadings[0] < 0: # Check stopping_frequency
            loadings = -loadings
            
        loadings = loadings / np.sum(np.abs(loadings))
        
        print("\nNew PCA Loadings (normalized):")
        for feat, load in zip(active_features, loadings):
            print(f"  {feat:<25}: {load:.4f}")
            
        df.loc[valid_idx, 'NUS_weak_pca'] = np.dot(df.loc[valid_idx, norm_cols], loadings)
    
    # 4. Global min-max scaling to [0, 1]
    print("\nScaling NUS_weak targets to [0, 1]...")
    scaling_params = {}
    
    for col in ['NUS_weak_expert', 'NUS_weak_pca']:
        c_min = float(df[col].min())
        c_max = float(df[col].max())
        scaling_params[col] = {'global_min': c_min, 'global_max': c_max}
        
        # Avoid div by zero just in case
        if c_max - c_min > 1e-6:
            df[col + '_scaled'] = (df[col] - c_min) / (c_max - c_min)
        else:
            df[col + '_scaled'] = 0.0
            
    import json
    
    os.makedirs(os.path.join("combined", "labels"), exist_ok=True)
    param_file = os.path.join("combined", "labels", "nus_weak_scaling_params.json")
    with open(param_file, 'w') as f:
        json.dump(scaling_params, f, indent=4)
    print(f"Saved scaling params to {param_file}")
    
    # Save final results
    out_file = os.path.join("combined", "labels", "features_final.csv")
    df.drop(columns=['old_NUS_weak_expert'], inplace=True, errors='ignore')
    df.to_csv(out_file, index=False)
    print(f"\nSaved finalized features to {out_file}")
    
    # Print stats
    print("\n--- Final Statistics ---")
    valid_df = df.dropna(subset=['NUS_weak_expert', 'NUS_weak_pca', 'NUS_weak_expert_scaled', 'NUS_weak_pca_scaled'])
    print(f"NUS_weak_expert: Min={valid_df['NUS_weak_expert'].min():.4f}, Max={valid_df['NUS_weak_expert'].max():.4f}, Mean={valid_df['NUS_weak_expert'].mean():.4f}")
    print(f"NUS_weak_pca   : Min={valid_df['NUS_weak_pca'].min():.4f}, Max={valid_df['NUS_weak_pca'].max():.4f}, Mean={valid_df['NUS_weak_pca'].mean():.4f}")
    print(f"NUS_weak_expert_scaled: Min={valid_df['NUS_weak_expert_scaled'].min():.4f}, Max={valid_df['NUS_weak_expert_scaled'].max():.4f}, Mean={valid_df['NUS_weak_expert_scaled'].mean():.4f}")
    print(f"NUS_weak_pca_scaled   : Min={valid_df['NUS_weak_pca_scaled'].min():.4f}, Max={valid_df['NUS_weak_pca_scaled'].max():.4f}, Mean={valid_df['NUS_weak_pca_scaled'].mean():.4f}")
    
    top_10 = (valid_df['NUS_weak_expert_scaled'] > 0.9).sum()
    bottom_10 = (valid_df['NUS_weak_expert_scaled'] < 0.1).sum()
    total_valid = len(valid_df)
    print(f"\nDistribution Check (NUS_weak_expert_scaled):")
    print(f"  Top 10% (>0.9)   : {top_10} windows ({top_10/total_valid*100:.1f}%)")
    print(f"  Bottom 10% (<0.1): {bottom_10} windows ({bottom_10/total_valid*100:.1f}%)")
    
    # Load old dataset temporarily just to correlate safely in case row order shifted
    # Skip correlation check for global dataset since we don't have a single old dataset to compare to.
    print(f"\nTotal combined windows: {total_valid}")

if __name__ == "__main__":
    main()
