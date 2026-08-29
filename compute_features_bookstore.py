import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Configurable constants
# V_MIN: velocity threshold for stopping. Needs recalibration once homography is applied!
# Currently using an equivalent in pixel units (e.g. 5.0 pixels/frame or /sec).
# Setting it to 5.0 pixels/second as a placeholder.
V_MIN = 5.0

# Expert weights for NUS_weak (can be easily tuned)
EXPERT_WEIGHTS = {
    "stopping_frequency": 0.2,
    "loop_backtrack": 0.2,
    "turning_angle_variance": 0.15,
    "local_hesitation": 0.15,
    "path_efficiency": 0.1,
    "speed_variance": 0.1,
    "waiting_time": 0.1,
    "interaction_change_rate": 0.0,
    "crowd_density": 0.0
}

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Loads preprocessed trajectory CSV with explicit dtypes.
    """
    # NOTE: The prompt asked for agent_id to be int, but in the preprocessing script 
    # we generated IDs like '12_b' to handle gaps. Therefore, agent_id MUST be read as a string.
    dtypes = {
        'dataset': str,
        'scene_id': str,
        'agent_id': str,
        'frame': int,
        'timestamp': float,
        'x_world': float,
        'y_world': float,
        'agent_type': str
    }
    df = pd.read_csv(filepath, dtype=dtypes)
    return df

def compute_interaction_change_rate(window_df, all_pedestrians_at_time=None):
    # TODO: requires Part 6 graph
    return 0.0

def compute_crowd_density(window_df, all_pedestrians_at_time=None):
    # TODO: requires Part 6 graph
    return 0.0

def extract_windows_and_features(df: pd.DataFrame, window_size: float = 5.0, stride: float = 2.5, min_samples: int = 8):
    """
    Segments tracks into overlapping windows and computes features per window.
    """
    results = []
    degenerate_count = 0
    
    for (scene_id, agent_id), track in df.groupby(['scene_id', 'agent_id']):
        track = track.sort_values('timestamp')
        t_min = track['timestamp'].min()
        t_max = track['timestamp'].max()
        
        w_start = t_min
        while w_start + window_size <= t_max + 1e-5:
            w_end = w_start + window_size
            w_df = track[(track['timestamp'] >= w_start - 1e-5) & (track['timestamp'] <= w_end + 1e-5)]
            
            if len(w_df) >= min_samples:
                x = w_df['x_world'].values
                y = w_df['y_world'].values
                t = w_df['timestamp'].values
                
                dx = np.diff(x)
                dy = np.diff(y)
                dt = np.diff(t)
                dt[dt == 0] = 1e-5  # safeguard
                
                speeds = np.sqrt(dx**2 + dy**2) / dt
                
                # Check for degenerate windows (e.g. 0 length path, all identical points)
                if len(speeds) == 0 or np.all(speeds == 0):
                    degenerate_count += 1
                    # Append NaN row
                    res = {k: np.nan for k in EXPERT_WEIGHTS.keys()}
                    res['u_turn_count'] = np.nan
                    res['oscillation_count'] = np.nan
                else:
                    # 1. stopping_frequency
                    stopping_freq = np.mean(speeds < V_MIN)
                    
                    # 2. speed_variance
                    mean_speed = np.mean(speeds)
                    speed_var = np.var(speeds) / (mean_speed**2 + 1e-8)
                    
                    # 3. path_efficiency
                    straight_line_dist = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
                    total_path_len = np.sum(np.sqrt(dx**2 + dy**2))
                    path_eff = 1.0 - (straight_line_dist / (total_path_len + 1e-8))
                    
                    # 4. turning_angle_variance & loop_backtrack
                    headings = np.arctan2(dy, dx)
                    heading_changes = np.diff(headings)
                    # wrap angles to [-pi, pi]
                    heading_changes = (heading_changes + np.pi) % (2 * np.pi) - np.pi
                    
                    if len(heading_changes) > 0:
                        turn_var = np.var(heading_changes)
                        u_turns = np.abs(heading_changes) > (150 * np.pi / 180)
                        u_turn_count = np.sum(u_turns)
                        
                        signs = np.sign(heading_changes)
                        sign_diffs = np.diff(signs)
                        oscillation_count = np.sum(sign_diffs != 0)
                    else:
                        turn_var = 0.0
                        u_turn_count = 0
                        oscillation_count = 0
                        
                    duration = t[-1] - t[0]
                    if duration == 0: duration = 1e-5
                    loop_backtrack = u_turn_count / duration
                    
                    # 5. local_hesitation
                    if len(speeds) >= 3:
                        local_minima = np.sum((speeds[1:-1] < speeds[:-2]) & (speeds[1:-1] < speeds[2:]))
                    else:
                        local_minima = 0
                    local_hesitation = local_minima / duration
                    
                    # 6. waiting_time
                    is_slow = speeds < V_MIN
                    max_run = 0
                    current_run = 0
                    for slow in is_slow:
                        if slow:
                            current_run += 1
                            max_run = max(max_run, current_run)
                        else:
                            current_run = 0
                    # using approximate count * fixed dt
                    waiting_time = (max_run * 0.4) / duration
                    
                    # stubs
                    interaction_rate = compute_interaction_change_rate(w_df)
                    crowd_den = compute_crowd_density(w_df)
                    
                    res = {
                        'stopping_frequency': stopping_freq,
                        'speed_variance': speed_var,
                        'path_efficiency': path_eff,
                        'turning_angle_variance': turn_var,
                        'local_hesitation': local_hesitation,
                        'loop_backtrack': loop_backtrack,
                        'interaction_change_rate': interaction_rate,
                        'crowd_density': crowd_den,
                        'waiting_time': waiting_time,
                        'u_turn_count': u_turn_count,
                        'oscillation_count': oscillation_count
                    }
                
                # window metadata
                res_meta = {
                    'scene_id': scene_id,
                    'agent_id': agent_id,
                    'window_id': f"{scene_id}_{agent_id}_{w_start:.1f}_{w_end:.1f}",
                    'window_start_time': w_start,
                    'window_end_time': w_end
                }
                res.update(res_meta)
                results.append(res)
                
            w_start += stride
            
    print(f"Generated {len(results)} windows. Found {degenerate_count} degenerate windows filled with NaN.")
    return pd.DataFrame(results)

def robust_scale_features(df, features):
    """
    Normalizes features per scene using median and IQR.
    """
    norm_params = {}
    normalized_dfs = []
    
    for scene_id, group in df.groupby('scene_id'):
        scene_params = {}
        norm_group = group.copy()
        
        for feat in features:
            val = group[feat].dropna()
            if len(val) == 0:
                median, iqr_safe = 0.0, 1.0
            else:
                median = np.median(val)
                q75, q25 = np.percentile(val, [75, 25])
                iqr = q75 - q25
                if iqr == 0:
                    print(f"WARNING: IQR is 0 for feature '{feat}' in scene '{scene_id}'. Using epsilon 1e-6 for scaling.")
                iqr_safe = max(iqr, 1e-6)
            
            scene_params[feat] = {'median': float(median), 'iqr': float(iqr_safe)}
            norm_group[feat + '_norm'] = (group[feat] - median) / iqr_safe
            
        norm_params[scene_id] = scene_params
        normalized_dfs.append(norm_group)
        
    with open('normalization_params.json', 'w') as f:
        json.dump(norm_params, f, indent=4)
        
    return pd.concat(normalized_dfs, ignore_index=True)

def compute_nus_weak(df, active_features):
    """
    Computes NUS_weak two ways: Expert weights and PCA.
    """
    norm_cols = [f + "_norm" for f in active_features]
    
    # 1. Expert weights
    df['NUS_weak_expert'] = 0.0
    for feat in active_features:
        df['NUS_weak_expert'] += df[feat + "_norm"] * EXPERT_WEIGHTS[feat]
        
    # 2. PCA weights
    valid_idx = df[norm_cols].dropna().index
    if len(valid_idx) > 0:
        pca = PCA(n_components=1)
        pca.fit(df.loc[valid_idx, norm_cols])
        loadings = pca.components_[0]
        
        # Sign check: stopping_frequency (usually index 0) should ideally load positively
        # If it's negative, flip all signs
        if loadings[0] < 0:
            loadings = -loadings
            
        # Normalize weights to sum to 1 (using absolute sum to keep proportions)
        loadings = loadings / np.sum(np.abs(loadings))
        
        print("\nPCA Loadings (normalized):")
        for feat, load in zip(active_features, loadings):
            print(f"  {feat:<25}: {load:.4f}")
            
        df['NUS_weak_pca'] = np.nan
        df.loc[valid_idx, 'NUS_weak_pca'] = np.dot(df.loc[valid_idx, norm_cols].values, loadings)
    else:
        df['NUS_weak_pca'] = np.nan
        
    return df

def main():
    print("WARNING: interaction_change_rate and crowd_density are currently set to 0.0 (requires Part 6 full-scene graph).")
    
    input_file = os.path.join("processed", "bookstore", "bookstore_all.csv")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    print(f"Loading data from {input_file}...")
    df = load_csv(input_file)
    
    print("Segmenting tracks into windows and computing features...")
    window_df = extract_windows_and_features(df)
    if len(window_df) == 0:
        print("No windows generated.")
        return
        
    active_features = [
        "stopping_frequency",
        "speed_variance",
        "path_efficiency",
        "turning_angle_variance",
        "local_hesitation",
        "loop_backtrack",
        "waiting_time"
    ]
    # excluding interaction_change_rate and crowd_density from PCA/Normalization since they are 0
    
    print("Normalizing features per scene using robust scaling...")
    window_df = robust_scale_features(window_df, active_features)
    
    print("Computing NUS_weak scores (Expert and PCA)...")
    window_df = compute_nus_weak(window_df, active_features)
    
    output_file = os.path.join("processed", "bookstore", "features_windows.csv")
    # explicitly cast before saving
    output_dtypes = {
        'scene_id': str,
        'agent_id': str,
        'window_id': str
    }
    for col, dtype in output_dtypes.items():
        if col in window_df.columns:
            window_df[col] = window_df[col].astype(dtype)
            
    window_df.to_csv(output_file, index=False)
    print(f"\nSaved processed features to {output_file}")
    
    # Verification of scaling logic
    print("\n--- Verification of IQR=0 scaling logic ---")
    with open('normalization_params.json', 'r') as f:
        params = json.load(f)
        
    sample_row = window_df.iloc[0]
    sample_scene = sample_row['scene_id']
    sample_feat = active_features[0]
    
    raw_val = sample_row[sample_feat]
    norm_col_val = sample_row[sample_feat + '_norm']
    
    p_med = params[sample_scene][sample_feat]['median']
    p_iqr = params[sample_scene][sample_feat]['iqr']
    manual_norm = (raw_val - p_med) / p_iqr
    
    print(f"Sample Scene: {sample_scene} | Feature: {sample_feat}")
    print(f"  JSON Median: {p_med:.6f} | JSON IQR: {p_iqr:.6f}")
    print(f"  Raw Value: {raw_val:.6f}")
    print(f"  Computed _norm in CSV: {norm_col_val:.6f}")
    print(f"  Manually computed:    {manual_norm:.6f}")
    if np.isclose(norm_col_val, manual_norm, equal_nan=True):
        print("  -> SUCCESS: Match verified!")
    else:
        print("  -> ERROR: Mismatch detected!")
    
    # Summary
    print("\n--- Summary ---")
    print(f"Total windows: {len(window_df)}")
    print("\nFeature value distributions (raw):")
    for feat in active_features:
        print(f"  {feat:<25}: Min={window_df[feat].min():.4f}, Max={window_df[feat].max():.4f}, Mean={window_df[feat].mean():.4f}, Std={window_df[feat].std():.4f}")
        
    if not window_df[['NUS_weak_expert', 'NUS_weak_pca']].dropna().empty:
        corr = window_df['NUS_weak_expert'].corr(window_df['NUS_weak_pca'])
        print(f"\nCorrelation between NUS_weak_expert and NUS_weak_pca: {corr:.4f}")
    else:
        print("\nCould not compute correlation due to NaNs.")

if __name__ == "__main__":
    main()
