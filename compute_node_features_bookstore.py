import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import warnings

# Suppress some noisy pandas warnings when working with correlations
warnings.filterwarnings('ignore', category=RuntimeWarning)

# GNN_INPUT_FEATURES: Explicit list of feature names to be used as GNN node inputs.
# Exclusions:
# - speed_max and net_displacement were dropped for near-total redundancy with speed_mean (r > 0.92, expected due to fixed window duration).
# - u_turn_count was dropped because it is a near-duplicate (r = 0.999) of loop_backtrack, a direct weighted component of NUS_weak_expert (label leak).
GNN_INPUT_FEATURES = [
    "speed_mean", "acceleration_mean", "acceleration_std", "jerk_mean",
    "heading_mean", "angular_velocity_mean", "deceleration_event_count",
    "local_path_curvature", "bounding_box_area", "time_since_track_start",
    "dwell_duration",
    # graph-derived, currently stubbed at 0.0, pending Part 6:
    "nearest_neighbor_distance", "local_pedestrian_density",
    "relative_velocity_to_nearest", "node_degree", "node_centrality",
]

DECEL_THRESHOLD = -0.5 # Configurable negative threshold for deceleration events (pixel/sec^2)

def load_gnn_input_features(node_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns only the columns specified in GNN_INPUT_FEATURES (plus the necessary ID keys).
    """
    keys = ['scene_id', 'agent_id', 'window_id']
    return node_features_df[keys + GNN_INPUT_FEATURES]

def load_csv(filepath: str) -> pd.DataFrame:
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

def compute_curvature(dx, dy):
    """
    Computes curvature using first and second derivatives.
    k = |dx*d2y - dy*d2x| / (dx^2 + dy^2)^(3/2)
    """
    if len(dx) < 2:
        return np.array([])
    d2x = np.diff(dx)
    d2y = np.diff(dy)
    dx = dx[:-1]
    dy = dy[:-1]
    
    numerator = np.abs(dx * d2y - dy * d2x)
    denominator = (dx**2 + dy**2)**(1.5)
    
    # Avoid div by zero
    valid = denominator > 1e-6
    curvature = np.zeros_like(numerator)
    curvature[valid] = numerator[valid] / denominator[valid]
    return curvature

def extract_node_features(df: pd.DataFrame, window_size: float = 5.0, stride: float = 2.5, min_samples: int = 8):
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
                dt[dt == 0] = 1e-5
                
                speeds = np.sqrt(dx**2 + dy**2) / dt
                
                if len(speeds) < 2 or np.all(speeds == 0):
                    degenerate_count += 1
                    res = {
                        'speed_mean': np.nan, 'speed_max': np.nan,
                        'acceleration_mean': np.nan, 'acceleration_std': np.nan,
                        'jerk_mean': np.nan, 'heading_mean': np.nan,
                        'angular_velocity_mean': np.nan, 'deceleration_event_count': np.nan,
                        'local_path_curvature': np.nan, 'bounding_box_area': np.nan,
                        'net_displacement': np.nan, 'time_since_track_start': np.nan
                    }
                else:
                    acc = np.diff(speeds) / dt[1:]
                    if len(acc) > 1:
                        jerk = np.diff(acc) / dt[2:]
                    else:
                        jerk = np.array([0.0])
                        
                    headings = np.arctan2(dy, dx)
                    heading_changes = np.diff(headings)
                    heading_changes = (heading_changes + np.pi) % (2 * np.pi) - np.pi
                    ang_vel = heading_changes / dt[1:]
                    
                    curvature = compute_curvature(dx, dy)
                    
                    # Circular mean for heading
                    heading_mean = np.arctan2(np.mean(np.sin(headings)), np.mean(np.cos(headings)))
                    
                    res = {
                        'speed_mean': np.mean(speeds),
                        'speed_max': np.max(speeds),
                        'acceleration_mean': np.mean(acc),
                        'acceleration_std': np.std(acc) if len(acc) > 0 else 0.0,
                        'jerk_mean': np.mean(jerk) if len(jerk) > 0 else 0.0,
                        'heading_mean': heading_mean,
                        'angular_velocity_mean': np.mean(ang_vel) if len(ang_vel) > 0 else 0.0,
                        'deceleration_event_count': np.sum(acc < DECEL_THRESHOLD),
                        'local_path_curvature': np.mean(np.abs(curvature)) if len(curvature) > 0 else 0.0,
                        'bounding_box_area': (np.max(x) - np.min(x)) * (np.max(y) - np.min(y)),
                        'net_displacement': np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2),
                        'time_since_track_start': w_start - t_min
                    }
                
                # Metadata
                res['scene_id'] = scene_id
                res['agent_id'] = agent_id
                res['window_id'] = f"{scene_id}_{agent_id}_{w_start:.1f}_{w_end:.1f}"
                res['window_start_time'] = w_start
                res['window_end_time'] = w_end
                
                # Stubs
                res['nearest_neighbor_distance'] = 0.0
                res['local_pedestrian_density'] = 0.0
                res['relative_velocity_to_nearest'] = 0.0
                res['node_degree'] = 0.0
                res['node_centrality'] = 0.0
                
                results.append(res)
            w_start += stride
            
    print(f"Generated {len(results)} windows. Found {degenerate_count} degenerate windows filled with NaN.")
    return pd.DataFrame(results)

def main():
    stub_features = [
        'nearest_neighbor_distance', 'local_pedestrian_density', 
        'relative_velocity_to_nearest', 'node_degree', 'node_centrality'
    ]
    print(f"WARNING: The following graph-based features are currently set to 0.0 (requires Part 6 full-scene graph):")
    for s in stub_features:
        print(f"  - {s}")
        
    input_file = os.path.join("processed", "bookstore", "bookstore_all.csv")
    print(f"\nLoading raw trajectories from {input_file}...")
    df = load_csv(input_file)
    
    print("Segmenting tracks into windows and computing node features...")
    node_df = extract_node_features(df)
    
    if len(node_df) == 0:
        print("No windows generated.")
        return
        
    # Load features_windows.csv to merge waiting_time (dwell_duration) and NUS_weak_expert
    features_file = os.path.join("processed", "bookstore", "features_windows.csv")
    print(f"\nLoading part 4 features from {features_file}...")
    feat_df = pd.read_csv(features_file)
    
    # Merge using window_id
    # We bring in 'waiting_time' (as dwell_duration) and 'NUS_weak_expert'
    merged_df = pd.merge(
        node_df, 
        feat_df[['window_id', 'waiting_time', 'NUS_weak_expert', 
                 'stopping_frequency', 'speed_variance', 'path_efficiency', 
                 'turning_angle_variance', 'local_hesitation', 'loop_backtrack',
                 'u_turn_count', 'oscillation_count']], 
        on='window_id', how='left'
    )
    
    # Assign dwell_duration
    merged_df['dwell_duration'] = merged_df['waiting_time']
    
    # Save the output node features (exclude the extra Part 4 features we just joined for analysis, 
    # except dwell_duration which belongs to Part 5).
    # Wait, the prompt says: "Save the full node-feature set (all features, including stubs) to processed/bookstore/node_features.csv"
    output_cols = [
        'scene_id', 'agent_id', 'window_id', 'window_start_time', 'window_end_time',
        'speed_mean', 'speed_max', 'acceleration_mean', 'acceleration_std',
        'jerk_mean', 'heading_mean', 'angular_velocity_mean', 'deceleration_event_count',
        'local_path_curvature', 'bounding_box_area', 'net_displacement',
        'time_since_track_start', 'dwell_duration', 'u_turn_count'
    ] + stub_features
    
    out_file = os.path.join("processed", "bookstore", "node_features.csv")
    # Cast IDs
    for col in ['scene_id', 'agent_id', 'window_id']:
        merged_df[col] = merged_df[col].astype(str)
        
    merged_df[output_cols].to_csv(out_file, index=False)
    print(f"Saved processed node features to {out_file}")
    
    # Save feature manifest
    import json
    
    # EXCLUDED_BUT_RETAINED logic:
    # - speed_max, net_displacement: excluded for near-total redundancy with speed_mean (r > 0.92)
    # - u_turn_count: excluded because it's a near-duplicate (r = 0.999) of loop_backtrack, a weighted component of NUS_weak_expert — including it as a GNN input would leak label information
    excluded = [c for c in output_cols if c not in GNN_INPUT_FEATURES and c not in ['scene_id', 'agent_id', 'window_id', 'window_start_time', 'window_end_time']]
    if 'u_turn_count' not in excluded:
        excluded.append('u_turn_count')
        
    manifest = {
        "GNN_INPUT_FEATURES": GNN_INPUT_FEATURES,
        "EXCLUDED_BUT_RETAINED": excluded,
        "STUBBED_FEATURES": stub_features
    }
    manifest_file = os.path.join("processed", "bookstore", "feature_manifest.json")
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=4)
    print(f"Saved feature manifest to {manifest_file}")
    
    # 1. Correlation Pruning (restricted to GNN_INPUT_FEATURES, minus stubs)
    print("\n--- Final Correlation Pruning (GNN_INPUT_FEATURES |r| > 0.9) ---")
    active_gnn_features = [f for f in GNN_INPUT_FEATURES if f not in stub_features]
    corr_df = merged_df[active_gnn_features].corr()
    
    highly_correlated = []
    for i in range(len(active_gnn_features)):
        for j in range(i+1, len(active_gnn_features)):
            feat_i = active_gnn_features[i]
            feat_j = active_gnn_features[j]
            r = corr_df.loc[feat_i, feat_j]
            if abs(r) > 0.9:
                highly_correlated.append((feat_i, feat_j, r))
                
    if highly_correlated:
        for f1, f2, r in highly_correlated:
            print(f"  {f1} <--> {f2}: r = {r:.4f}")
    else:
        print("  No feature pairs found with |r| > 0.9.")
        
    # 2. Permutation Importance
    print("\n--- Permutation Importance (Target: NUS_weak_expert) ---")
    rf_features = [
        'speed_mean', 'speed_max', 'acceleration_mean', 'acceleration_std',
        'jerk_mean', 'heading_mean', 'angular_velocity_mean', 'deceleration_event_count',
        'local_path_curvature', 'bounding_box_area', 'net_displacement',
        'time_since_track_start', 'dwell_duration'
    ]
    
    # Drop NaNs
    valid_data = merged_df[rf_features + ['NUS_weak_expert']].dropna()
    if len(valid_data) > 0:
        X = valid_data[rf_features]
        y = valid_data['NUS_weak_expert']
        
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X, y)
        
        result = permutation_importance(rf, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        importances = result.importances_mean
        
        # Rank
        indices = np.argsort(importances)[::-1]
        for idx in indices:
            print(f"  {rf_features[idx]:<25}: {importances[idx]:.4f}")
    else:
        print("  Could not compute permutation importance due to NaNs or lack of data.")
        
    print("\n--- Summary ---")
    print(f"GNN_INPUT_FEATURES count: {len(GNN_INPUT_FEATURES)} (11 real + 5 stubbed)")
    computed_count = len(output_cols) - len(stub_features) - 5 # minus metadata
    print(f"Total computed node features (saved to CSV): {computed_count}")
    print(f"Stubbed node features: {len(stub_features)}")

if __name__ == "__main__":
    main()
