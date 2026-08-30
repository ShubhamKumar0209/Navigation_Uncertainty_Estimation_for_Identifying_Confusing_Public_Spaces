import os
import json
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree

def load_csv(filepath: str) -> pd.DataFrame:
    dtypes = {
        'dataset': str, 'scene_id': str, 'agent_id': str,
        'frame': int, 'timestamp': float, 'x_world': float,
        'y_world': float, 'agent_type': str
    }
    return pd.read_csv(filepath, dtype=dtypes)

def compute_agent_window_stats(track_df):
    """Computes mean position and velocity vector for an agent within a global window."""
    mean_x = track_df['x_world'].mean()
    mean_y = track_df['y_world'].mean()
    
    # Velocity vector using endpoints or simple diff
    t = track_df['timestamp'].values
    x = track_df['x_world'].values
    y = track_df['y_world'].values
    
    if len(t) > 1 and (t[-1] - t[0]) > 0:
        dt = t[-1] - t[0]
        vx = (x[-1] - x[0]) / dt
        vy = (y[-1] - y[0]) / dt
    else:
        vx, vy = 0.0, 0.0
        
    return mean_x, mean_y, vx, vy

def main():
    print("Loading data...")
    input_file = os.path.join("processed", "bookstore", "bookstore_all.csv")
    df = load_csv(input_file)
    
    window_length = 5.0
    stride = 2.5
    
    # 1. Global Scene-wide Windows Extraction
    print("\nSegmenting global windows and computing positions...")
    global_windows = {} # dict: (scene_id, window_idx) -> list of agent_stats
    
    for scene_id, scene_df in df.groupby('scene_id'):
        t_min = scene_df['timestamp'].min()
        t_max = scene_df['timestamp'].max()
        
        w_start = t_min
        win_idx = 0
        while w_start + window_length <= t_max + 1e-5:
            w_end = w_start + window_length
            mask = (scene_df['timestamp'] >= w_start - 1e-5) & (scene_df['timestamp'] <= w_end + 1e-5)
            w_df = scene_df[mask]
            
            agents_in_window = []
            for agent_id, track in w_df.groupby('agent_id'):
                mx, my, vx, vy = compute_agent_window_stats(track)
                agents_in_window.append({
                    'agent_id': agent_id,
                    'x': mx, 'y': my, 'vx': vx, 'vy': vy
                })
                
            global_windows[(scene_id, win_idx)] = {
                'start': w_start,
                'end': w_end,
                'agents': agents_in_window
            }
            w_start += stride
            win_idx += 1
            
    # 2. Dynamic Radius Calculation per scene
    print("\nComputing dynamic radii (90th percentile of nearest neighbors)...")
    scene_radii = {}
    
    # Pre-accumulate distances
    for scene_id in df['scene_id'].unique():
        nn_dists = []
        # find all windows for this scene
        scene_keys = [k for k in global_windows.keys() if k[0] == scene_id]
        
        for k in scene_keys:
            agents = global_windows[k]['agents']
            if len(agents) > 1:
                pts = np.array([[a['x'], a['y']] for a in agents])
                tree = cKDTree(pts)
                # query k=2 (closest is self at dist 0, next is NN)
                dists, _ = tree.query(pts, k=2)
                nn_dists.extend(dists[:, 1])
                
        if len(nn_dists) > 0:
            r = np.percentile(nn_dists, 90)
        else:
            r = 1.0 # fallback
        scene_radii[scene_id] = float(r)
        print(f"  {scene_id}: r = {r:.4f}m")
        
    os.makedirs(os.path.join("processed", "bookstore", "graphs"), exist_ok=True)
    with open(os.path.join("processed", "bookstore", "graph_radius_params.json"), 'w') as f:
        json.dump(scene_radii, f, indent=4)
        
    # 3 & 4. Build Radius Graphs and Extract Features
    print("\nBuilding graphs and extracting features...")
    graph_features_list = []
    
    # Track neighbor sets for interaction_change_rate
    # prev_neighbors: (scene_id, agent_id) -> set of neighbor agent_ids from previous global window
    prev_neighbors = {}
    
    for (scene_id, win_idx), win_data in sorted(global_windows.items()):
        w_start = win_data['start']
        w_end = win_data['end']
        agents = win_data['agents']
        
        r = scene_radii[scene_id]
        sigma = r / 2.0
        
        # Build KDTree
        if len(agents) > 0:
            pts = np.array([[a['x'], a['y']] for a in agents])
            tree = cKDTree(pts)
        else:
            continue
            
        G = nx.Graph()
        
        # Area of bounding box
        if len(agents) > 0:
            min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
            min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
            area = (max_x - min_x) * (max_y - min_y)
            if area < 1e-3: area = 1e-3 # fallback for collinear/single
        else:
            area = 1.0
            
        crowd_density = len(agents)
        local_ped_density = crowd_density / area
        
        # Add nodes
        for i, a in enumerate(agents):
            G.add_node(a['agent_id'], pos=(a['x'], a['y']), v=(a['vx'], a['vy']))
            
        # Connect nodes within r and cap max degree to 8 globally
        if len(agents) > 1:
            all_edges = []
            for i, a in enumerate(agents):
                # query all neighbors within r
                indices = tree.query_ball_point(pts[i], r)
                for j in indices:
                    if i < j: # Only process undirected edges once
                        d = np.linalg.norm(pts[i] - pts[j])
                        all_edges.append((i, j, d))
            
            # Sort all candidate edges by distance (ascending)
            all_edges.sort(key=lambda x: x[2])
            
            # Greedily add edges if both endpoints are under degree-8 cap
            degree_counts = {i: 0 for i in range(len(agents))}
            for i, j, d in all_edges:
                if degree_counts[i] < 8 and degree_counts[j] < 8:
                    w = np.exp(-(d**2) / (2 * sigma**2))
                    G.add_edge(agents[i]['agent_id'], agents[j]['agent_id'], weight=w, distance=d)
                    degree_counts[i] += 1
                    degree_counts[j] += 1
                    
        # Extract features per agent
        global_window_id = f"{scene_id}_{w_start:.1f}_{w_end:.1f}"
        
        if len(G.nodes) > 0:
            centralities = nx.degree_centrality(G)
        
        for i, a in enumerate(agents):
            aid = a['agent_id']
            # Find neighbors in Graph
            neighbors = list(G.neighbors(aid)) if aid in G else []
            node_degree = len(neighbors)
            node_cent = centralities.get(aid, 0.0) if aid in G else 0.0
            
            if node_degree > 0:
                # Find closest neighbor from Graph edges
                edge_dists = [G[aid][nbr]['distance'] for nbr in neighbors]
                min_idx = np.argmin(edge_dists)
                nn_dist = edge_dists[min_idx]
                nn_id = neighbors[min_idx]
                
                # Relative velocity
                vx_diff = a['vx'] - G.nodes[nn_id]['v'][0]
                vy_diff = a['vy'] - G.nodes[nn_id]['v'][1]
                rel_v = np.sqrt(vx_diff**2 + vy_diff**2)
            else:
                nn_dist = 999.0
                rel_v = 0.0
                
            # Interaction change rate
            curr_nbr_set = set(neighbors)
            prev_key = (scene_id, aid)
            if prev_key in prev_neighbors:
                prev_nbr_set = prev_neighbors[prev_key]
                union = len(curr_nbr_set | prev_nbr_set)
                if union == 0:
                    inter_change = 0.0
                else:
                    intersection = len(curr_nbr_set & prev_nbr_set)
                    inter_change = 1.0 - (intersection / union)
            else:
                inter_change = 0.0 # First window
                
            # Update history
            prev_neighbors[prev_key] = curr_nbr_set
            
            graph_features_list.append({
                'scene_id': scene_id,
                'agent_id': aid,
                'global_window_id': global_window_id,
                'global_window_start': w_start,
                'global_window_end': w_end,
                'nearest_neighbor_distance': nn_dist,
                'local_pedestrian_density': local_ped_density,
                'relative_velocity_to_nearest': rel_v,
                'node_degree': node_degree,
                'node_centrality': node_cent,
                'interaction_change_rate': inter_change,
                'crowd_density': crowd_density
            })
            
        # Serialize graph
        graph_path = os.path.join("processed", "bookstore", "graphs", f"{global_window_id}.gpickle")
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f)
            
    gf_df = pd.DataFrame(graph_features_list)
    
    # 4b. Robust Normalization for graph features per scene
    print("\nNormalizing graph features per scene...")
    norm_cols = ['local_pedestrian_density', 'crowd_density', 'nearest_neighbor_distance', 'relative_velocity_to_nearest']
    norm_params = {}
    normalized_dfs = []
    
    for scene_id, group in gf_df.groupby('scene_id'):
        scene_params = {}
        norm_group = group.copy()
        
        for feat in norm_cols:
            val = group[feat].dropna()
            if len(val) == 0:
                median, iqr_safe = 0.0, 1.0
            else:
                median = np.median(val)
                q75, q25 = np.percentile(val, [75, 25])
                iqr = q75 - q25
                iqr_safe = iqr if iqr > 0.01 else 1.0
            
            scene_params[feat] = {'median': float(median), 'iqr': float(iqr_safe)}
            
            # Scale and clip to [-5, 5]
            raw_scaled = (group[feat] - median) / iqr_safe
            norm_group[feat + '_norm'] = np.clip(raw_scaled, -5, 5)
            
        norm_params[scene_id] = scene_params
        normalized_dfs.append(norm_group)
        
    gf_df = pd.concat(normalized_dfs, ignore_index=True)
    
    with open(os.path.join("processed", "bookstore", "graph_feature_normalization_params.json"), 'w') as f:
        json.dump(norm_params, f, indent=4)
    
    gf_file = os.path.join("processed", "bookstore", "graph_derived_features.csv")
    gf_df.to_csv(gf_file, index=False)
    print(f"Saved global graph features to {gf_file}")
    
    # 5. Join onto agent windows
    print("\nJoining graph features onto local agent windows...")
    def merge_graph_features(local_df, global_df):
        merged_rows = []
        zero_overlap_count = 0
        
        for idx, row in local_df.iterrows():
            l_start = row['window_start_time']
            l_end = row['window_end_time']
            scene = row['scene_id']
            agent = row['agent_id']
            
            # Find overlaps
            g_cand = global_df[(global_df['scene_id'] == scene) & (global_df['agent_id'] == agent)]
            # Overlap condition: max(l_start, g_start) < min(l_end, g_end)
            overlap_mask = (np.maximum(l_start, g_cand['global_window_start']) < 
                            np.minimum(l_end, g_cand['global_window_end']))
            overlaps = g_cand[overlap_mask]
            
            out_row = row.copy()
            if len(overlaps) > 0:
                out_row['nearest_neighbor_distance'] = overlaps['nearest_neighbor_distance'].mean()
                out_row['local_pedestrian_density'] = overlaps['local_pedestrian_density'].mean()
                out_row['relative_velocity_to_nearest'] = overlaps['relative_velocity_to_nearest'].mean()
                out_row['node_degree'] = overlaps['node_degree'].mean()
                out_row['node_centrality'] = overlaps['node_centrality'].mean()
                # These were not stubbed in part 5 but were stubbed in part 4
                out_row['interaction_change_rate'] = overlaps['interaction_change_rate'].mean()
                out_row['crowd_density'] = overlaps['crowd_density'].mean()
                out_row['local_pedestrian_density_norm'] = overlaps['local_pedestrian_density_norm'].mean()
                out_row['crowd_density_norm'] = overlaps['crowd_density_norm'].mean()
                out_row['nearest_neighbor_distance_norm'] = overlaps['nearest_neighbor_distance_norm'].mean()
                out_row['relative_velocity_to_nearest_norm'] = overlaps['relative_velocity_to_nearest_norm'].mean()
            else:
                zero_overlap_count += 1
                out_row['nearest_neighbor_distance'] = 999.0
                out_row['local_pedestrian_density'] = 0.0
                out_row['relative_velocity_to_nearest'] = 0.0
                out_row['node_degree'] = 0.0
                out_row['node_centrality'] = 0.0
                out_row['interaction_change_rate'] = 0.0
                out_row['crowd_density'] = 1.0 # at least themselves
                out_row['local_pedestrian_density_norm'] = 0.0
                out_row['crowd_density_norm'] = 0.0
                out_row['nearest_neighbor_distance_norm'] = 0.0
                out_row['relative_velocity_to_nearest_norm'] = 0.0
                
            merged_rows.append(out_row)
            
        print(f"  Found {zero_overlap_count} agent windows with zero overlapping global windows.")
        return pd.DataFrame(merged_rows)
        
    feat_df = pd.read_csv(os.path.join("processed", "bookstore", "features_windows.csv"))
    node_df = pd.read_csv(os.path.join("processed", "bookstore", "node_features.csv"))
    
    # Pre-cast for safe joining
    gf_df['scene_id'] = gf_df['scene_id'].astype(str)
    gf_df['agent_id'] = gf_df['agent_id'].astype(str)
    feat_df['scene_id'] = feat_df['scene_id'].astype(str)
    feat_df['agent_id'] = feat_df['agent_id'].astype(str)
    node_df['scene_id'] = node_df['scene_id'].astype(str)
    node_df['agent_id'] = node_df['agent_id'].astype(str)
    
    print("  Processing features_windows...")
    feat_graph_df = merge_graph_features(feat_df, gf_df)
    feat_graph_df.to_csv(os.path.join("processed", "bookstore", "features_windows_with_graph.csv"), index=False)
    
    print("  Processing node_features...")
    node_graph_df = merge_graph_features(node_df, gf_df)
    node_graph_df.to_csv(os.path.join("processed", "bookstore", "node_features_with_graph.csv"), index=False)
    
    # 6. Print summary
    print("\n--- Summary Statistics ---")
    print(f"Average node degree per scene: {gf_df.groupby('scene_id')['node_degree'].mean().to_dict()}")
    isolated_frac = (gf_df['node_degree'] == 0).mean()
    print(f"Fraction of isolated (degree-0) nodes: {isolated_frac:.4f}")
    
    max_degree = gf_df['node_degree'].max()
    print(f"Maximum node degree across all windows: {max_degree} (Assert <= 8: {max_degree <= 8})")
    
    print(f"\nInteraction Change Rate Distribution:")
    print(f"  Min={gf_df['interaction_change_rate'].min():.4f}, Max={gf_df['interaction_change_rate'].max():.4f}, Mean={gf_df['interaction_change_rate'].mean():.4f}")
    
    print(f"Crowd Density Distribution:")
    print(f"  Min={gf_df['crowd_density'].min():.4f}, Max={gf_df['crowd_density'].max():.4f}, Mean={gf_df['crowd_density'].mean():.4f}")
    
    print("\nNormalized Feature Distributions:")
    for feat in ['local_pedestrian_density_norm', 'crowd_density_norm', 'nearest_neighbor_distance_norm', 'relative_velocity_to_nearest_norm']:
        print(f"  {feat}: Min={gf_df[feat].min():.4f}, Max={gf_df[feat].max():.4f}, Mean={gf_df[feat].mean():.4f}")
    
    # Assert no NaNs
    cols_to_check = [
        'nearest_neighbor_distance', 'local_pedestrian_density', 
        'relative_velocity_to_nearest', 'node_degree', 'node_centrality',
        'local_pedestrian_density_norm', 'crowd_density_norm', 
        'nearest_neighbor_distance_norm', 'relative_velocity_to_nearest_norm'
    ]
    nan_count = node_graph_df[cols_to_check].isna().sum().sum()
    print(f"\nRemaining NaNs in stubbed columns after join: {nan_count} (Expected: 0)")

if __name__ == "__main__":
    main()
