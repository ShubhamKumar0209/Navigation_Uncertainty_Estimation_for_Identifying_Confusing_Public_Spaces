# Combined Python Code for Project

## `build_graphs_bookstore.py`

```python
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
```

## `build_pyg_dataset.py`

```python
import os
import json
import glob
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

def load_data():
    manifest_path = os.path.join("processed", "bookstore", "feature_manifest.json")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    gnn_features = manifest["GNN_INPUT_FEATURES"]
    
    ff_path = os.path.join("processed", "bookstore", "features_final.csv")
    targets_df = pd.read_csv(ff_path)
    
    nf_path = os.path.join("processed", "bookstore", "node_features_with_graph.csv")
    nodes_df = pd.read_csv(nf_path)
    
    # Pre-cast IDs
    targets_df['scene_id'] = targets_df['scene_id'].astype(str)
    targets_df['agent_id'] = targets_df['agent_id'].astype(str)
    nodes_df['scene_id'] = nodes_df['scene_id'].astype(str)
    nodes_df['agent_id'] = nodes_df['agent_id'].astype(str)
    
    # Fill NaNs for degenerate node features with 0.0
    nodes_df[gnn_features] = nodes_df[gnn_features].fillna(0.0)
    
    return targets_df, nodes_df, gnn_features

def compute_overlap(start1, end1, start2, end2):
    return max(0.0, min(end1, end2) - max(start1, start2))

def build_global_graphs(nodes_df, gnn_features):
    print("Building PyG Data objects for all global windows...")
    
    # Pre-group node features for fast lookup
    print("  Pre-grouping node features...")
    node_groups = {}
    for (scene_id, agent_id), group in nodes_df.groupby(['scene_id', 'agent_id']):
        node_groups[(scene_id, agent_id)] = group.sort_values('window_start_time')
        
    graphs_dir = os.path.join("processed", "bookstore", "graphs")
    graph_files = glob.glob(os.path.join(graphs_dir, "*.gpickle"))
    
    global_data = {} # Key: global_window_id -> dict with 'data' (PyG Data), 'node_map', 'end_time'
    
    for gf in graph_files:
        filename = os.path.basename(gf)
        global_window_id = filename.replace('.gpickle', '')
        
        # parse scene_id, start, end from filename: scene_id_start_end.gpickle
        # Wait, scene_id might have underscores (e.g. bookstore_video0).
        # We know window_id format is f"{scene_id}_{w_start:.1f}_{w_end:.1f}"
        parts = global_window_id.split('_')
        w_end = float(parts[-1])
        w_start = float(parts[-2])
        scene_id = "_".join(parts[:-2])
        
        with open(gf, 'rb') as f:
            G = pickle.load(f)
            
        nodes = list(G.nodes())
        node_idx_map = {n: i for i, n in enumerate(nodes)}
        
        # Build X matrix
        x_list = []
        for agent_id in nodes:
            key = (scene_id, agent_id)
            if key in node_groups:
                agent_df = node_groups[key]
                # Find max overlap
                overlaps = agent_df.apply(lambda row: compute_overlap(w_start, w_end, row['window_start_time'], row['window_end_time']), axis=1)
                best_idx = overlaps.idxmax()
                best_row = agent_df.loc[best_idx]
                feat_vec = best_row[gnn_features].values.astype(np.float32)
            else:
                # Should practically never happen if preprocessing is consistent
                feat_vec = np.zeros(len(gnn_features), dtype=np.float32)
            x_list.append(feat_vec)
            
        if len(x_list) > 0:
            x_tensor = torch.tensor(np.vstack(x_list), dtype=torch.float)
        else:
            x_tensor = torch.empty((0, len(gnn_features)), dtype=torch.float)
            
        # Build edges (undirected -> symmetric)
        edges = []
        edge_attrs = []
        for u, v, d in G.edges(data=True):
            ui = node_idx_map[u]
            vi = node_idx_map[v]
            w = float(d['weight'])
            edges.append([ui, vi])
            edge_attrs.append([w])
            edges.append([vi, ui])
            edge_attrs.append([w])
            
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
            
        data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)
        
        global_data[global_window_id] = {
            'scene_id': scene_id,
            'start_time': w_start,
            'end_time': w_end,
            'data': data,
            'node_map': node_idx_map
        }
        
    print(f"  Built {len(global_data)} global PyG graphs.")
    return global_data

def build_sequences(targets_df, global_data):
    print("\nBuilding L=6 temporal sequences for each target window...")
    
    # Organize global windows by scene, sorted by end time
    scene_globals = {}
    for gw_id, info in global_data.items():
        scene_id = info['scene_id']
        if scene_id not in scene_globals:
            scene_globals[scene_id] = []
        scene_globals[scene_id].append((info['end_time'], gw_id))
        
    for scene_id in scene_globals:
        scene_globals[scene_id].sort(key=lambda x: x[0])
        
    dataset = []
    pad_count = 0
    pad_amounts = []
    
    for idx, row in targets_df.iterrows():
        scene_id = row['scene_id']
        agent_id = row['agent_id']
        target_val = row['NUS_weak_expert_scaled']
        l_end = row['window_end_time']
        
        # Find all global windows in this scene ending <= local window end
        if scene_id not in scene_globals:
            continue
            
        valid_globals = [gw_id for (end_time, gw_id) in scene_globals[scene_id] if end_time <= l_end + 1e-5]
        
        if len(valid_globals) == 0:
            # Agent's local window ends before ANY global window ends. Extremely rare/impossible with proper alignment.
            continue
            
        # Take up to 6 most recent
        seq_gw_ids = valid_globals[-6:]
        
        # Pad if less than 6 by repeating the earliest
        if len(seq_gw_ids) < 6:
            pad_needed = 6 - len(seq_gw_ids)
            pad_count += 1
            pad_amounts.append(pad_needed)
            # Prepend copies of the earliest available
            seq_gw_ids = [seq_gw_ids[0]] * pad_needed + seq_gw_ids
            
        seq_data = []
        seq_node_idx = []
        
        for gw_id in seq_gw_ids:
            info = global_data[gw_id]
            seq_data.append(info['data'])
            
            # Map target agent to node index. -1 if absent.
            node_map = info['node_map']
            idx_in_graph = node_map.get(agent_id, -1)
            seq_node_idx.append(idx_in_graph)
            
        # Check for NaNs in target
        if pd.isna(target_val):
            continue
            
        # Append to dataset
        # sample = (list_of_6_Data_objects, target_agent_row_index_per_snapshot, NUS_weak_expert_scaled, scene_id)
        dataset.append({
            'sequence': seq_data,
            'target_indices': torch.tensor(seq_node_idx, dtype=torch.long),
            'label': torch.tensor([target_val], dtype=torch.float),
            'scene_id': scene_id,
            'agent_id': agent_id,
            'window_start_time': row['window_start_time'],
            'window_end_time': row['window_end_time']
        })
        
    print(f"  Assembled {len(dataset)} valid samples.")
    print(f"  {pad_count} sequences required padding (avg pad length among padded: {np.mean(pad_amounts):.2f}).")
    
    return dataset

def main():
    targets_df, nodes_df, gnn_features = load_data()
    global_data = build_global_graphs(nodes_df, gnn_features)
    dataset = build_sequences(targets_df, global_data)
    
    # Save
    out_file = os.path.join("processed", "bookstore", "dataset.pt")
    torch.save(dataset, out_file)
    print(f"\nSaved complete PyG dataset to {out_file}")
    
    # Print summary
    print("\n--- Summary ---")
    scene_counts = {}
    label_mins = {}
    label_maxs = {}
    label_sums = {}
    
    nan_features = 0
    
    for sample in dataset:
        sid = sample['scene_id']
        val = sample['label'].item()
        
        scene_counts[sid] = scene_counts.get(sid, 0) + 1
        
        if sid not in label_mins:
            label_mins[sid] = val
            label_maxs[sid] = val
            label_sums[sid] = 0.0
            
        label_mins[sid] = min(label_mins[sid], val)
        label_maxs[sid] = max(label_maxs[sid], val)
        label_sums[sid] += val
        
        for data in sample['sequence']:
            if torch.isnan(data.x).any():
                nan_features += 1
                
    for sid in sorted(scene_counts.keys()):
        count = scene_counts[sid]
        mean_val = label_sums[sid] / count
        print(f"  {sid:<18}: {count:<5} samples | Label [Min={label_mins[sid]:.4f}, Max={label_maxs[sid]:.4f}, Mean={mean_val:.4f}]")
        
    print(f"\nSamples with NaN node features: {nan_features} (Expected: 0)")
    
if __name__ == "__main__":
    main()
```

## `compute_features_bookstore.py`

```python
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
                if iqr <= 0.01:
                    print(f"WARNING: IQR ({iqr:.4f}) is <= 0.01 for feature '{feat}' in scene '{scene_id}'. Using fallback IQR=1.0 for scaling.")
                iqr_safe = iqr if iqr > 0.01 else 1.0
            
            scene_params[feat] = {'median': float(median), 'iqr': float(iqr_safe)}
            
            # Scale and clip to [-5, 5]
            raw_scaled = (group[feat] - median) / iqr_safe
            norm_group[feat + '_norm'] = np.clip(raw_scaled, -5, 5)
            
        norm_params[scene_id] = scene_params
        normalized_dfs.append(norm_group)
        
    with open('normalization_params.json', 'w') as f:
        json.dump(norm_params, f, indent=4)
        
    normalized_df = pd.concat(normalized_dfs, ignore_index=True)
    
    # Print clipping stats globally
    print("\n--- Clipping Stats ([-5, 5]) ---")
    for feat in features:
        raw_scaled = (normalized_df[feat] - normalized_df.apply(lambda row: norm_params[row['scene_id']][feat]['median'], axis=1)) / \
                     normalized_df.apply(lambda row: norm_params[row['scene_id']][feat]['iqr'], axis=1)
        clipped_count = np.sum((raw_scaled < -5) | (raw_scaled > 5))
        print(f"  {feat:<25}: {clipped_count} values clipped")
        
    return normalized_df

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
    manual_norm = np.clip((raw_val - p_med) / p_iqr, -5, 5)
    
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
        print("\nNUS_weak Distributions:")
        print(f"  NUS_weak_expert : Min={window_df['NUS_weak_expert'].min():.4f}, Max={window_df['NUS_weak_expert'].max():.4f}, Mean={window_df['NUS_weak_expert'].mean():.4f}")
        print(f"  NUS_weak_pca    : Min={window_df['NUS_weak_pca'].min():.4f}, Max={window_df['NUS_weak_pca'].max():.4f}, Mean={window_df['NUS_weak_pca'].mean():.4f}")
        corr = window_df['NUS_weak_expert'].corr(window_df['NUS_weak_pca'])
        print(f"\nCorrelation between NUS_weak_expert and NUS_weak_pca: {corr:.4f}")
    else:
        print("\nCould not compute correlation due to NaNs.")

if __name__ == "__main__":
    main()
```

## `compute_node_features_bookstore.py`

```python
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
```

## `explainability.py`

```python
import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import shap
from torch.utils.data import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer

from model import NavUncertaintyModel
from train import NavDataset, custom_collate

with open("processed/bookstore/feature_manifest.json", "r") as f:
    manifest = json.load(f)
FEATURE_NAMES = manifest["GNN_INPUT_FEATURES"]

def train_shap_surrogate(dataset, model, device):
    print("Building and training SHAP Surrogate Model...")
    
    loader = DataLoader(NavDataset(dataset), batch_size=32, shuffle=False, collate_fn=custom_collate)
    all_preds = []
    
    model.eval()
    with torch.no_grad():
        for seq_batch, target_idx, _ in loader:
            seq_batch = [b.to(device) for b in seq_batch]
            target_idx = target_idx.to(device)
            reg_out, _ = model(seq_batch, target_idx)
            preds = torch.clamp(reg_out, 0.0, 1.0)
            all_preds.extend(preds.cpu().numpy())
            
    y_pred = np.array(all_preds)
    
    # Flatten sequential features
    X_list = []
    for item in dataset:
        seq = item['sequence']
        t_idx = item['target_indices']
        
        node_feats = []
        for t in range(6):
            idx = t_idx[t].item()
            if idx != -1:
                node_feats.append(seq[t].x[idx].numpy())
            else:
                node_feats.append(np.zeros(16))
                
        # Average across the 6 snapshots
        X_list.append(np.mean(node_feats, axis=0))
        
    X_train = np.vstack(X_list)
    
    surrogate = GradientBoostingRegressor(n_estimators=100, random_state=42)
    surrogate.fit(X_train, y_pred)
    
    r2 = r2_score(y_pred, surrogate.predict(X_train))
    print(f"  Surrogate Model R² against GAT+GRU: {r2:.4f}")
    
    background = shap.sample(X_train, 100)
    explainer = shap.TreeExplainer(surrogate, data=background, feature_perturbation='interventional')
    return surrogate, explainer, X_train, y_pred

def extract_gat_attention(model, item, device):
    # Format single item into a batch of size 1
    loader = DataLoader(NavDataset([item]), batch_size=1, collate_fn=custom_collate)
    seq_batch, target_idx, _ = next(iter(loader))
    
    seq_batch = [b.to(device) for b in seq_batch]
    target_idx = target_idx.to(device)
    
    model.eval()
    with torch.no_grad():
        _, _, attentions = model(seq_batch, target_idx, return_attention=True)
        
    # extract top-3 edges for each snapshot
    snapshot_top_edges = []
    
    for t in range(6):
        idx = target_idx[0, t].item()
        if idx == -1:
            snapshot_top_edges.append([])
            continue
            
        edge_index, alpha = attentions[t]
        edge_index = edge_index.cpu()
        # alpha has shape [num_edges, heads]. Average across heads to get a single weight per edge.
        alpha = alpha.cpu().mean(dim=1)
        
        # incoming edges TO the target node
        mask = edge_index[1] == idx
        if not mask.any():
            snapshot_top_edges.append([])
            continue
            
        in_edges = edge_index[0, mask]
        in_weights = alpha[mask]
        
        # DEBUG: Print raw attention for the final snapshot (t=5)
        if t == 5:
            print(f"    [DEBUG Attention] Node {idx} in-edges: {in_edges.tolist()}, weights: {in_weights.tolist()}")
        
        # Sort by weight descending
        sorted_indices = torch.argsort(in_weights, descending=True)
        top_k = min(3, len(sorted_indices))
        
        top_edges = []
        for i in range(top_k):
            si = sorted_indices[i]
            neighbor = in_edges[si].item()
            w = in_weights[si].item()
            # Optionally skip self-loops in the top reporting
            if neighbor == idx:
                continue
            top_edges.append({
                'neighbor_idx': neighbor,
                'weight': w
            })
            
        snapshot_top_edges.append(top_edges)
        
    return snapshot_top_edges

class SingleSnapshotWrapper(nn.Module):
    def __init__(self, model, seq_batch, target_idx):
        super().__init__()
        self.model = model
        self.seq_batch = seq_batch
        self.target_idx = target_idx
        
        # Precompute the sequence embeddings for t=0..4 to avoid triggering Explainer hooks
        B = target_idx.size(0)
        self.seq_embeddings = torch.zeros((B, 6, 64), device=target_idx.device)
        
        with torch.no_grad():
            for t in range(5):
                batch_data = seq_batch[t]
                node_embeddings = model._encode_snapshot(batch_data)
                idx_t = target_idx[:, t]
                present_mask = (idx_t != -1)
                absent_mask = ~present_mask
                if present_mask.any():
                    valid_idx = idx_t[present_mask]
                    self.seq_embeddings[present_mask, t, :] = node_embeddings[valid_idx]
                if absent_mask.any():
                    self.seq_embeddings[absent_mask, t, :] = model.absent_embedding
        
    def forward(self, x, edge_index):
        from torch_geometric.data import Data
        # Only process t=5 dynamically so PyG explain hooks only see this graph!
        batch_data = Data(x=x, edge_index=edge_index, edge_attr=torch.ones((edge_index.size(1), 1), device=x.device))
        
        node_embeddings = self.model._encode_snapshot(batch_data)
        seq_emb = self.seq_embeddings.clone()
        
        idx_t = self.target_idx[:, 5]
        present_mask = (idx_t != -1)
        absent_mask = ~present_mask
        
        if present_mask.any():
            valid_idx = idx_t[present_mask]
            seq_emb[present_mask, 5, :] = node_embeddings[valid_idx]
        if absent_mask.any():
            seq_emb[absent_mask, 5, :] = self.model.absent_embedding
            
        gru_out, hn = self.model.gru(seq_emb)
        final_h = hn.squeeze(0)
        final_h = self.model.head_dropout(final_h)
        reg_out = self.model.reg_head(final_h).squeeze(-1) # shape [1]
        
        # Expand to trick PyG's node-level explainer indexing
        num_nodes = x.size(0)
        idx_t5 = self.target_idx[0, 5].item()
        trick_out = torch.zeros(num_nodes, device=x.device)
        trick_out[idx_t5] = reg_out[0]
        
        return trick_out

def run_gnn_explainer(model, item, device):
    loader = DataLoader(NavDataset([item]), batch_size=1, collate_fn=custom_collate)
    seq_batch, target_idx, _ = next(iter(loader))
    seq_batch = [b.to(device) for b in seq_batch]
    target_idx = target_idx.to(device)
    
    idx_t5 = target_idx[0, 5].item()
    if idx_t5 == -1:
        return None
        
    wrapper = SingleSnapshotWrapper(model, seq_batch, target_idx)
    wrapper.eval()
    
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='object',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='node',
            return_type='raw',
        ),
    )
    
    x = seq_batch[5].x
    edge_index = seq_batch[5].edge_index
    
    explanation = explainer(x, edge_index, index=idx_t5)
    
    node_mask = explanation.node_mask.cpu()
    edge_mask = explanation.edge_mask.cpu()
    
    print(f"    [DEBUG Explainer] node_mask min: {node_mask.min().item():.4f}, max: {node_mask.max().item():.4f}, mean: {node_mask.mean().item():.4f}")
    
    # Adaptive thresholding: use 50% of the max node mask value, rather than a hard 0.5
    threshold = node_mask.max().item() * 0.5
    if threshold < 1e-4:
        threshold = 1e-4
        
    important_nodes = (node_mask > threshold).nonzero(as_tuple=True)[0].tolist()
    if idx_t5 in important_nodes:
        important_nodes.remove(idx_t5)
        
    return {
        'num_important_neighbors': len(important_nodes),
        'neighbors': important_nodes
    }

def explain_instance(sample_idx, dataset, model, explainer, X_train, y_pred, device):
    item = dataset[sample_idx]
    
    # 1. SHAP
    shap_vals = explainer.shap_values(X_train[sample_idx:sample_idx+1])[0]
    top_indices = np.argsort(np.abs(shap_vals))[::-1][:3]
    
    shap_summary = []
    for i in top_indices:
        feat = FEATURE_NAMES[i]
        val = shap_vals[i]
        direction = "high" if X_train[sample_idx][i] > np.mean(X_train[:, i]) else "low"
        shap_summary.append(f"{feat}: {direction}, {val:+.3f} contribution")
        
    # 2. GAT Attention
    attention_edges = extract_gat_attention(model, item, device)
    last_snap_attn = attention_edges[5] if len(attention_edges) == 6 else []
    
    # 3. GNNExplainer
    gnn_expl = run_gnn_explainer(model, item, device)
    
    # 4. Text Generation
    text = f"Pedestrian {item['agent_id']} in {item['scene_id']} (Predicted NUS: {y_pred[sample_idx]:.3f}):\n"
    text += f"  - Top Features (SHAP): {', '.join(shap_summary)}.\n"
    if last_snap_attn:
        top_n = last_snap_attn[0]
        text += f"  - Social Attention: Strongest focus on neighbor {top_n['neighbor_idx']} (weight: {top_n['weight']:.2f}).\n"
    else:
        text += f"  - Social Attention: No active neighbors in final snapshot.\n"
        
    if gnn_expl:
        text += f"  - Subgraph Cause: GNNExplainer flagged {gnn_expl['num_important_neighbors']} critical neighbor(s) driving the prediction.\n"
        
    return text, shap_vals, last_snap_attn

def profile_flagged_zones(flagged_path, pos_path, X_train, explainer, dataset):
    print("\nProfiling Cell-Level Dominant Causes...")
    flagged_df = pd.read_csv(flagged_path)
    pos_df = pd.read_csv(pos_path)
    
    # Pre-calculate SHAP values for all samples
    print("  Computing SHAP values for full dataset (this takes a moment)...")
    all_shap = explainer.shap_values(X_train)
    
    # Map dataset index to (scene_id, agent_id, window_start)
    sample_map = {}
    for i, item in enumerate(dataset):
        key = (item['scene_id'], str(item['agent_id']), item['window_start_time'])
        sample_map[key] = i
        
    zone_profiles = []
    
    for _, zone in flagged_df.iterrows():
        scene = zone['scene_id']
        cx = zone['cell_x_index']
        cy = zone['cell_y_index']
        
        # Reconstruct scene bounds to find world coords
        scene_pos = pos_df[pos_df['scene_id'] == scene]
        min_x = scene_pos['x_world'].min()
        min_y = scene_pos['y_world'].min()
        
        # Find all points falling into this cell (GRID_CELL_SIZE = 25)
        cell_points = scene_pos[
            ((scene_pos['x_world'] - min_x) // 25 == cx) &
            ((scene_pos['y_world'] - min_y) // 25 == cy)
        ]
        
        if cell_points.empty:
            continue
            
        # Get unique windows passing through here
        unique_windows = cell_points[['agent_id', 'window_start_time']].drop_duplicates()
        
        cell_shaps = []
        for _, w in unique_windows.iterrows():
            key = (scene, str(w['agent_id']), w['window_start_time'])
            if key in sample_map:
                idx = sample_map[key]
                cell_shaps.append(all_shap[idx])
                
        if len(cell_shaps) == 0:
            continue
            
        mean_shap = np.mean(cell_shaps, axis=0)
        top_indices = np.argsort(np.abs(mean_shap))[::-1][:2]
        top_feats = [FEATURE_NAMES[i] for i in top_indices]
        
        nl_desc = f"Zone (x={cx}, y={cy}) in {scene}: high NUS driven primarily by {top_feats[0]} and {top_feats[1]}."
        
        zone_profiles.append({
            'scene_id': scene,
            'cell_x_index': cx,
            'cell_y_index': cy,
            'top_feature_1': top_feats[0],
            'top_feature_2': top_feats[1],
            'description': nl_desc
        })
        
    out_df = pd.DataFrame(zone_profiles)
    out_df.to_csv("zone_explanations.csv", index=False)
    print(f"  Saved {len(out_df)} zone explanations to zone_explanations.csv")
    
    # Print a few examples
    for i in range(min(3, len(out_df))):
        print("  " + out_df.iloc[i]['description'])

def generate_visualizations(dataset, X_train, explainer):
    print("\nGenerating Visualizations for top 3 highest NUS samples...")
    os.makedirs("explanation_examples", exist_ok=True)
    
    # Find top 3 predictions
    model = NavUncertaintyModel()
    model.load_state_dict(torch.load("best_model_rotation_C.pt", weights_only=True))
    model.eval()
    
    loader = DataLoader(NavDataset(dataset), batch_size=32, collate_fn=custom_collate)
    preds = []
    with torch.no_grad():
        for seq, tgt, _ in loader:
            p, _ = model(seq, tgt)
            preds.extend(p.cpu().numpy())
            
    # Deduplicate by scene and agent
    seen_agents = set()
    diverse_top_idx = []
    
    sorted_idx = np.argsort(preds)[::-1]
    for idx in sorted_idx:
        item = dataset[idx]
        key = (item['scene_id'], str(item['agent_id']))
        if key not in seen_agents:
            seen_agents.add(key)
            diverse_top_idx.append(idx)
        if len(diverse_top_idx) == 3:
            break
            
    top_3_idx = diverse_top_idx
    
    pos_df = pd.read_csv("predictions_with_positions.csv")
    
    for idx in top_3_idx:
        item = dataset[idx]
        scene = item['scene_id']
        agent = str(item['agent_id'])
        w_start = item['window_start_time']
        w_end = item['window_end_time']
        
        traj = pos_df[(pos_df['scene_id'] == scene) & 
                      (pos_df['agent_id'] == agent) & 
                      (pos_df['window_start_time'] == w_start)]
                      
        shap_vals = explainer.shap_values(X_train[idx:idx+1])[0]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Trajectory Scatter
        axes[0].plot(traj['x_world'], traj['y_world'], 'b-', marker='o', label='Trajectory')
        axes[0].set_title(f"{scene} | Agent {agent}")
        axes[0].legend()
        
        # 2. SHAP Bar Plot
        top_k = 5
        top_feat_idx = np.argsort(np.abs(shap_vals))[::-1][:top_k]
        
        y_pos = np.arange(top_k)
        axes[1].barh(y_pos, shap_vals[top_feat_idx], align='center', color='green')
        axes[1].set_yticks(y_pos, labels=[FEATURE_NAMES[i] for i in top_feat_idx])
        axes[1].invert_yaxis()
        axes[1].set_xlabel('SHAP Value (Impact on Model Output)')
        axes[1].set_title('Top Contributing Features')
        
        plt.tight_layout()
        plt.savefig(f"explanation_examples/{scene}_{agent}_{w_start}.png")
        plt.close()
        
    print("  Saved visualizations to explanation_examples/")

def main():
    device = torch.device('cpu')
    
    dataset_path = os.path.join("processed", "bookstore", "dataset.pt")
    dataset = torch.load(dataset_path, weights_only=False)
    
    model = NavUncertaintyModel().to(device)
    model.load_state_dict(torch.load("best_model_rotation_C.pt", weights_only=True))
    model.eval()
    
    surrogate, explainer, X_train, y_pred = train_shap_surrogate(dataset, model, device)
    
    print("\nExtracting per-pedestrian explanations for 3 random samples...")
    # Seed for reproducibility of random debug samples
    np.random.seed(42)
    for i in np.random.choice(len(dataset), 3, replace=False):
        text, _, _ = explain_instance(i, dataset, model, explainer, X_train, y_pred, device)
        print(text)
        
    print("\nExtracting top SHAP feature for 20 random instances...")
    top_feat_counts = {}
    for i in np.random.choice(len(dataset), 20, replace=False):
        shap_vals = explainer.shap_values(X_train[i:i+1])[0]
        top_idx = np.argsort(np.abs(shap_vals))[-1]
        top_f = FEATURE_NAMES[top_idx]
        top_feat_counts[top_f] = top_feat_counts.get(top_f, 0) + 1
    print("  Top-1 Feature Distribution across 20 samples:", top_feat_counts)
        
    profile_flagged_zones("flagged_zones.csv", "predictions_with_positions.csv", X_train, explainer, dataset)
    
    generate_visualizations(dataset, X_train, explainer)

if __name__ == "__main__":
    main()
```

## `finalize_nus_weak.py`

```python
import os
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
    
    input_file = os.path.join("processed", "bookstore", "features_windows_with_graph.csv")
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
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
    param_file = os.path.join("processed", "bookstore", "nus_weak_scaling_params.json")
    with open(param_file, 'w') as f:
        json.dump(scaling_params, f, indent=4)
    print(f"Saved scaling params to {param_file}")
    
    # Save final results
    out_file = os.path.join("processed", "bookstore", "features_final.csv")
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
    old_expert = pd.read_csv(input_file)['NUS_weak_expert']
    corr = valid_df['NUS_weak_expert'].corr(old_expert.loc[valid_df.index])
    print(f"\nCorrelation between old and new NUS_weak_expert: {corr:.4f}")

if __name__ == "__main__":
    main()
```

## `generate_heatmap.py`

```python
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize

from model import NavUncertaintyModel
from train import NavDataset, custom_collate

GRID_CELL_SIZE = 25  # pixel-space approximation, not true metric scale

def run_inference(model, dataset, device):
    print("Running inference on full dataset...")
    loader = DataLoader(NavDataset(dataset), batch_size=32, shuffle=False, collate_fn=custom_collate)
    
    all_preds = []
    
    with torch.no_grad():
        for seq_batch, target_idx, _ in loader:
            seq_batch = [b.to(device) for b in seq_batch]
            target_idx = target_idx.to(device)
            
            reg_out, _ = model(seq_batch, target_idx)
            # The model outputs raw regression (Huber loss target).
            # The user asked for "predicted NUS (sigmoid output, already in [0,1])"
            # Note: The model reg_out is bound to ~[0,1] organically via target scaling,
            # but we can clamp it to [0,1] to be safe as a 'score'.
            preds = torch.clamp(reg_out, 0.0, 1.0)
            all_preds.extend(preds.cpu().numpy())
            
    # Combine predictions with dataset metadata
    pred_data = []
    for i, item in enumerate(dataset):
        pred_data.append({
            'scene_id': item['scene_id'],
            'agent_id': str(item['agent_id']),
            'window_start_time': item['window_start_time'],
            'window_end_time': item['window_end_time'],
            'predicted_nus': all_preds[i]
        })
        
    preds_df = pd.DataFrame(pred_data)
    print(f"  Generated {len(preds_df)} window predictions.")
    return preds_df

def join_raw_positions(preds_df):
    print("Joining predictions with raw trajectory coordinates...")
    scenes = preds_df['scene_id'].unique()
    
    joined_frames = []
    
    for scene in scenes:
        raw_path = os.path.join("processed", "bookstore", f"{scene}.csv")
        if not os.path.exists(raw_path):
            print(f"Warning: Raw tracking file {raw_path} not found. Skipping scene.")
            continue
            
        raw_df = pd.read_csv(raw_path)
        raw_df['agent_id'] = raw_df['agent_id'].astype(str)
        
        # Merge by agent_id
        scene_preds = preds_df[preds_df['scene_id'] == scene]
        
        # For each prediction window, extract points
        for _, row in scene_preds.iterrows():
            agent_traj = raw_df[(raw_df['agent_id'] == row['agent_id']) & 
                                (raw_df['timestamp'] >= row['window_start_time']) & 
                                (raw_df['timestamp'] <= row['window_end_time'])].copy()
            
            agent_traj['predicted_nus'] = row['predicted_nus']
            agent_traj['window_start_time'] = row['window_start_time']
            agent_traj['window_end_time'] = row['window_end_time']
            
            joined_frames.append(agent_traj)
            
    full_joined_df = pd.concat(joined_frames, ignore_index=True)
    out_path = "predictions_with_positions.csv"
    full_joined_df.to_csv(out_path, index=False)
    print(f"  Saved {len(full_joined_df)} position points to {out_path}")
    return full_joined_df

def generate_heatmaps(joined_df):
    print("\nGenerating Spatial Heatmaps...")
    
    os.makedirs("heatmaps", exist_ok=True)
    scenes = joined_df['scene_id'].unique()
    
    flagged_zones = []
    
    for scene in scenes:
        scene_df = joined_df[joined_df['scene_id'] == scene].sort_values(['agent_id', 'timestamp'])
        
        if scene_df.empty:
            continue
            
        min_x = scene_df['x_world'].min()
        max_x = scene_df['x_world'].max()
        min_y = scene_df['y_world'].min()
        max_y = scene_df['y_world'].max()
        
        w = max(1, int((max_x - min_x) / GRID_CELL_SIZE) + 1)
        h = max(1, int((max_y - min_y) / GRID_CELL_SIZE) + 1)
        
        # Accumulators
        nus_sum_grid = np.zeros((h, w))
        time_sum_grid = np.zeros((h, w))
        visit_windows = [[set() for _ in range(w)] for _ in range(h)]
        
        # Dwell-time weighted aggregation
        for agent_id, group in scene_df.groupby('agent_id'):
            group = group.sort_values('timestamp')
            xs = group['x_world'].values
            ys = group['y_world'].values
            ts = group['timestamp'].values
            nus_vals = group['predicted_nus'].values
            starts = group['window_start_time'].values
            
            for i in range(len(group) - 1):
                dt = ts[i+1] - ts[i]
                if dt <= 0 or dt > 2.0: # Skip massive gaps
                    continue
                    
                cell_x = int((xs[i] - min_x) / GRID_CELL_SIZE)
                cell_y = int((ys[i] - min_y) / GRID_CELL_SIZE)
                
                # Bounds check
                cell_x = min(max(cell_x, 0), w - 1)
                cell_y = min(max(cell_y, 0), h - 1)
                
                nus_sum_grid[cell_y, cell_x] += nus_vals[i] * dt
                time_sum_grid[cell_y, cell_x] += dt
                # Track unique window-visits
                visit_windows[cell_y][cell_x].add((agent_id, starts[i]))
                
        # Compute mean NUS grid and count grid
        mean_nus_grid = np.zeros_like(nus_sum_grid)
        count_grid = np.zeros_like(nus_sum_grid)
        
        mask = time_sum_grid > 0
        mean_nus_grid[mask] = nus_sum_grid[mask] / time_sum_grid[mask]
        
        for cy in range(h):
            for cx in range(w):
                count_grid[cy, cx] = len(visit_windows[cy][cx])
                
        # Compute density and adaptive sigma
        density = np.sum(mask) / (w * h)
        # Empirical density range across scenes is roughly 0.14 (video5) to 0.52 (video0)
        norm_density = np.clip((density - 0.14) / (0.52 - 0.14), 0.0, 1.0)
        adaptive_sigma = 1.5 + norm_density * 1.0
        
        # Gaussian Smoothing (adaptive sigma for contiguous flagged zones)
        smoothed_nus = gaussian_filter(mean_nus_grid, sigma=adaptive_sigma)
        smoothed_count = gaussian_filter(count_grid, sigma=adaptive_sigma)
        
        # Traffic normalization / Confidence map
        # Normalize the sample count strictly to [0,1] for confidence
        max_count = np.max(smoothed_count)
        if max_count > 0:
            confidence_grid = smoothed_count / max_count
        else:
            confidence_grid = np.zeros_like(smoothed_count)
            
        # Display NUS grid (mask out areas with zero traffic completely)
        display_nus = smoothed_nus.copy()
        display_nus[count_grid == 0] = np.nan
        
        # Save arrays
        np.save(f"heatmaps/{scene}_mean_nus.npy", display_nus)
        np.save(f"heatmaps/{scene}_confidence.npy", confidence_grid)
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        im0 = axes[0].imshow(display_nus, cmap='viridis', origin='lower')
        axes[0].set_title(f"{scene} - Smoothed Mean NUS")
        fig.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(confidence_grid, cmap='viridis', origin='lower')
        axes[1].set_title(f"{scene} - Confidence (Normalized Sample Count)")
        fig.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(f"heatmaps/{scene}_heatmap.png")
        plt.close()
        
        # Flag discrete confusing zones
        if np.any(mask):
            nus_90th = np.nanpercentile(display_nus, 90)
            nonzero_counts = count_grid[count_grid > 0]
            if len(nonzero_counts) > 0:
                count_5th = np.percentile(nonzero_counts, 5)
            else:
                count_5th = 0
                
            y_idx, x_idx = np.where((display_nus > nus_90th) & (count_grid > count_5th))
            
            for cy, cx in zip(y_idx, x_idx):
                flagged_zones.append({
                    'scene_id': scene,
                    'cell_x_index': cx,
                    'cell_y_index': cy,
                    'world_x': min_x + cx * GRID_CELL_SIZE + GRID_CELL_SIZE/2,
                    'world_y': min_y + cy * GRID_CELL_SIZE + GRID_CELL_SIZE/2,
                    'mean_nus': display_nus[cy, cx],
                    'sample_count': count_grid[cy, cx]
                })
                
            print(f"  {scene:<18} | Grid: {w}x{h} ({w*h} cells) | Non-Empty: {np.sum(mask)} | Sigma: {adaptive_sigma:.2f} | NUS Range: [{np.nanmin(display_nus):.3f}, {np.nanmax(display_nus):.3f}] | Flagged: {len(x_idx)}")

    if flagged_zones:
        fz_df = pd.DataFrame(flagged_zones)
        fz_df.to_csv("flagged_zones.csv", index=False)
        print(f"\nSaved {len(fz_df)} confusing zones to flagged_zones.csv")

def main():
    device = torch.device('cpu')
    print("Loading model and PyG dataset...")
    
    # Load dataset
    dataset_path = os.path.join("processed", "bookstore", "dataset.pt")
    dataset = torch.load(dataset_path, weights_only=False)
    
    # Load Rotation C model (test scene = video5)
    model = NavUncertaintyModel().to(device)
    model.load_state_dict(torch.load("best_model_rotation_C.pt", weights_only=True))
    model.eval()
    
    # 1. Run Inference
    preds_df = run_inference(model, dataset, device)
    
    # 2. Join Raw Positions
    joined_df = join_raw_positions(preds_df)
    
    # 3-8. Generate Spatial Heatmaps and Flag Zones
    generate_heatmaps(joined_df)

if __name__ == "__main__":
    main()
```

## `model.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class NavUncertaintyModel(nn.Module):
    def __init__(self, node_in_dim=16, node_hidden_dim=64, gru_hidden=128):
        super().__init__()
        
        # Spatial Encoder (GAT)
        self.gat1 = GATConv(node_in_dim, node_hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(node_hidden_dim * 4, node_hidden_dim, heads=1, concat=False)
        self.dropout = nn.Dropout(0.3)
        
        # Fallback for absent nodes (-1 index)
        self.absent_embedding = nn.Parameter(torch.zeros(node_hidden_dim))
        
        # Temporal Aggregator
        self.gru = nn.GRU(input_size=node_hidden_dim, hidden_size=gru_hidden, batch_first=True)
        self.head_dropout = nn.Dropout(0.2)
        
        # Output Heads
        self.reg_head = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.aux_head = nn.Linear(gru_hidden, 2)
        
    def _encode_snapshot(self, batch_data, return_attention_weights=False):
        """Passes a single graph batch through the GAT layers."""
        x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
        
        if x.size(0) == 0:
            if return_attention_weights:
                return torch.empty((0, 64), device=x.device), None
            return torch.empty((0, 64), device=x.device)
            
        if return_attention_weights:
            x, attn_weights = self.gat1(x, edge_index, edge_attr, return_attention_weights=True)
        else:
            x = self.gat1(x, edge_index, edge_attr)
            attn_weights = None
            
        x = F.elu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index, edge_attr)
        x = F.elu(x)
        
        if return_attention_weights:
            return x, attn_weights
        return x
        
        
    def forward(self, sequence_batches, target_indices, return_attention=False):
        """
        sequence_batches: list of 6 PyG Batch objects (one per timestep t in [0..5]).
        target_indices: LongTensor of shape [B, 6]. Values are the node row-index in that batch's x tensor, or -1 if absent.
        """
        B = target_indices.size(0)
        L = len(sequence_batches)
        
        # Shape: [B, L, 64]
        seq_embeddings = torch.zeros((B, L, 64), device=target_indices.device)
        all_attentions = []
        
        for t in range(L):
            batch_data = sequence_batches[t]
            # Encode spatial graphs for timestep t
            if return_attention:
                node_embeddings, attn = self._encode_snapshot(batch_data, return_attention_weights=True)
                all_attentions.append(attn)
            else:
                node_embeddings = self._encode_snapshot(batch_data)
            
            # Extract target agents
            idx_t = target_indices[:, t]
            
            # Mask for present vs absent
            present_mask = (idx_t != -1)
            absent_mask = ~present_mask
            
            # Gather present
            if present_mask.any():
                valid_idx = idx_t[present_mask]
                seq_embeddings[present_mask, t, :] = node_embeddings[valid_idx]
                
            # Fill absent
            if absent_mask.any():
                seq_embeddings[absent_mask, t, :] = self.absent_embedding
                
        # Temporal aggregation
        # seq_embeddings is [B, L, 64]
        gru_out, hn = self.gru(seq_embeddings)
        
        # Final hidden state from the last timestep -> hn is [1, B, 128]
        final_h = hn.squeeze(0)
        final_h = self.head_dropout(final_h)
        
        # Heads
        reg_out = self.reg_head(final_h).squeeze(-1) # Shape [B]
        aux_out = self.aux_head(final_h)             # Shape [B, 2]
        
        if return_attention:
            return reg_out, aux_out, all_attentions
        return reg_out, aux_out
```

## `preprocess_bookstore.py`

```python
import os
import glob
import pandas as pd
import numpy as np

def load_and_parse_annotations(filepath: str) -> pd.DataFrame:
    """
    Loads and parses the annotations.txt file from the Stanford Drone Dataset.
    Columns: track_id, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label
    """
    columns = ["track_id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]
    
    # Read CSV. It's space-separated
    df = pd.read_csv(filepath, sep=r'\s+', names=columns)
    
    # Cast to correct dtypes
    df['track_id'] = df['track_id'].astype(int)
    df['xmin'] = df['xmin'].astype(int)
    df['ymin'] = df['ymin'].astype(int)
    df['xmax'] = df['xmax'].astype(int)
    df['ymax'] = df['ymax'].astype(int)
    df['frame'] = df['frame'].astype(int)
    df['lost'] = df['lost'].astype(bool)
    df['occluded'] = df['occluded'].astype(bool)
    df['generated'] = df['generated'].astype(bool)
    df['label'] = df['label'].astype(str).str.replace('"', '').str.replace("'", "")
    
    return df

def filter_pedestrians(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the dataframe to keep only rows where label is 'Pedestrian'.
    Drops other classes like Biker, Cart, Car, Bus, Skater.
    """
    return df[df['label'] == "Pedestrian"].copy()

def drop_lost_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where the 'lost' flag is True (1).
    These are placeholder/out-of-frame coordinates, not real positions.
    """
    return df[~df['lost']].copy()

def compute_centroid_and_timestamp(df: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
    """
    Computes the centroid position in pixels and the timestamp in seconds based on frame rate.
    """
    df['x_pixel'] = (df['xmin'] + df['xmax']) / 2.0
    df['y_pixel'] = (df['ymin'] + df['ymax']) / 2.0
    df['timestamp'] = df['frame'] / fps
    return df

def apply_homography(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for applying a homography transformation from pixel coordinates to world coordinates.
    Currently returns the dataframe unchanged, assigning pixel coordinates directly to world coordinates.
    """
    # TODO: homography - apply actual pixel-to-metric homography here
    df['x_world'] = df['x_pixel']
    df['y_world'] = df['y_pixel']
    return df

def build_unified_schema(df: pd.DataFrame, scene_id: str) -> pd.DataFrame:
    """
    Transforms the dataframe into the unified schema:
    [dataset, scene_id, agent_id, frame, timestamp, x_world, y_world, agent_type]
    """
    df = apply_homography(df)
    
    unified_df = pd.DataFrame()
    unified_df['dataset'] = ["SDD"] * len(df)
    unified_df['scene_id'] = [scene_id] * len(df)
    unified_df['agent_id'] = df['track_id'].astype(str)
    unified_df['frame'] = df['frame']
    unified_df['timestamp'] = df['timestamp']
    unified_df['x_world'] = df['x_world']
    unified_df['y_world'] = df['y_world']
    unified_df['agent_type'] = ["pedestrian"] * len(df)
    
    return unified_df

def split_tracks_by_gap(df: pd.DataFrame, max_gap_sec: float = 1.0) -> pd.DataFrame:
    """
    Splits tracks that have consecutive observations with a time gap greater than `max_gap_sec`.
    New tracks are assigned synthetic agent_ids (e.g., '{original_id}_b').
    """
    if len(df) == 0:
        return df
        
    # Sort by agent_id and timestamp
    df = df.sort_values(['agent_id', 'timestamp']).copy()
    
    # Calculate time difference between consecutive rows within the same agent
    df['time_diff'] = df.groupby('agent_id')['timestamp'].diff()
    
    # Identify where a new split should happen (time gap > max_gap_sec)
    df['split_flag'] = (df['time_diff'] > max_gap_sec).astype(int)
    
    # Cumulative sum gives a split index (0, 1, 2...) for each agent
    df['split_idx'] = df.groupby('agent_id')['split_flag'].cumsum()
    
    def generate_new_id(row):
        base_id = row['agent_id']
        idx = row['split_idx']
        if pd.isna(idx):
            idx = 0
        idx = int(idx)
        if idx == 0:
            return base_id
        else:
            # Generate a suffix based on the split index (1 -> b, 2 -> c, etc.)
            suffix = chr(ord('a') + idx)
            return f"{base_id}_{suffix}"
            
    df['agent_id'] = df.apply(generate_new_id, axis=1)
    
    # Clean up temporary columns
    df = df.drop(columns=['time_diff', 'split_flag', 'split_idx'])
    
    return df

def resample_and_filter_tracks(df: pd.DataFrame, dt: float = 0.4, min_duration: float = 5.0) -> pd.DataFrame:
    """
    Resamples tracks to a fixed timestep `dt` using linear interpolation and 
    discards tracks shorter than `min_duration`.
    """
    resampled_rows = []
    
    # Group by the potentially split agent_ids
    for agent_id, track in df.groupby('agent_id'):
        min_time = track['timestamp'].min()
        max_time = track['timestamp'].max()
        
        # Check track duration
        if (max_time - min_time) < min_duration:
            continue
            
        # Create a fixed time grid for resampling
        # Starting from the track's min_time to ensure we capture the full track bounds appropriately
        new_times = np.arange(min_time, max_time + 1e-9, dt)
        
        # Interpolate x and y
        new_x = np.interp(new_times, track['timestamp'], track['x_world'])
        new_y = np.interp(new_times, track['timestamp'], track['y_world'])
        new_frames = np.interp(new_times, track['timestamp'], track['frame']).round().astype(int)
        
        # Build the new track dataframe
        first_row = track.iloc[0]
        new_track = pd.DataFrame({
            'dataset': first_row['dataset'],
            'scene_id': first_row['scene_id'],
            'agent_id': agent_id,
            'frame': new_frames,
            'timestamp': new_times,
            'x_world': new_x,
            'y_world': new_y,
            'agent_type': first_row['agent_type']
        })
        resampled_rows.append(new_track)
        
    if not resampled_rows:
        return pd.DataFrame(columns=df.columns)
        
    return pd.concat(resampled_rows, ignore_index=True)

def process_video(video_path: str, output_dir: str) -> dict:
    """
    Processes a single video directory and saves the CSV file.
    Returns a dictionary of statistics for the sanity check summary.
    """
    scene_id = f"bookstore_{os.path.basename(video_path)}"
    annotations_file = os.path.join(video_path, 'annotations.txt')
    
    if not os.path.exists(annotations_file):
        print(f"Warning: {annotations_file} not found.")
        return None
        
    # 1. Load and parse
    df = load_and_parse_annotations(annotations_file)
    total_objects_before = df['track_id'].nunique()
    
    # 2. Filter pedestrians
    df = filter_pedestrians(df)
    pedestrians_before = df['track_id'].nunique()
    
    if len(df) == 0:
        return None
        
    # 3. Drop lost
    df = drop_lost_annotations(df)
    
    # 4 & 5. Compute centroid and timestamp
    df = compute_centroid_and_timestamp(df)
    
    # 6. Build unified schema
    df = build_unified_schema(df, scene_id)
    
    # Track split stats
    original_track_count = df['agent_id'].nunique()
    
    # 7. Split tracks by gap
    df = split_tracks_by_gap(df)
    split_track_count = df['agent_id'].nunique()
    tracks_split = split_track_count - original_track_count
    
    # 8. Resample and discard tiny tracks
    df = resample_and_filter_tracks(df)
    final_pedestrians = df['agent_id'].nunique()
    
    if len(df) > 0:
        avg_track_length = df.groupby('agent_id')['timestamp'].apply(lambda x: x.max() - x.min()).mean()
    else:
        avg_track_length = 0.0
        
    # 9. Save output
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{scene_id}.csv")
    df.to_csv(out_file, index=False)
    
    stats = {
        'scene_id': scene_id,
        'total_objects_raw': total_objects_before,
        'pedestrians_raw': pedestrians_before,
        'final_tracks': final_pedestrians,
        'tracks_split': tracks_split,
        'avg_track_length_sec': avg_track_length,
        'df': df
    }
    
    return stats

def main():
    base_dir = "bookstore"
    output_dir = os.path.join("processed", "bookstore")
    
    video_folders = sorted(glob.glob(os.path.join(base_dir, "video*")))
    if not video_folders:
        print(f"No video folders found in {base_dir}")
        return
        
    all_dfs = []
    
    print("Processing videos...")
    print("-" * 85)
    print(f"{'Scene':<18} | {'Raw Objs':<10} | {'Raw Peds':<10} | {'Splits':<8} | {'Final Tracks':<12} | {'Avg Len (s)':<12}")
    print("-" * 85)
    
    for folder in video_folders:
        stats = process_video(folder, output_dir)
        if stats:
            print(f"{stats['scene_id']:<18} | {stats['total_objects_raw']:<10} | {stats['pedestrians_raw']:<10} | {stats['tracks_split']:<8} | {stats['final_tracks']:<12} | {stats['avg_track_length_sec']:<12.2f}")
            if len(stats['df']) > 0:
                all_dfs.append(stats['df'])
                
    print("-" * 85)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_out = os.path.join(output_dir, "bookstore_all.csv")
        combined_df.to_csv(combined_out, index=False)
        print(f"\nSaved combined CSV file to {combined_out}")
        # agent_id is only unique per scene, so we group by scene_id and agent_id
        total_unique = combined_df.groupby(['scene_id', 'agent_id']).ngroups
        print(f"Total final pedestrian tracks across all scenes: {total_unique}")
    else:
        print("\nNo valid tracks found to combine.")

if __name__ == "__main__":
    main()
```

## `train.py`

```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

from model import NavUncertaintyModel

# Config
LAMBDA_1 = 0.3 # Ranking loss weight
LAMBDA_2 = 0.2 # Aux class loss weight
MARGIN = 0.1
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 5

ROTATIONS = {
    'A': {'train': ['bookstore_video0', 'bookstore_video1', 'bookstore_video2', 'bookstore_video3', 'bookstore_video6'], 'val': ['bookstore_video5'], 'test': ['bookstore_video4']},
    'B': {'train': ['bookstore_video0', 'bookstore_video1', 'bookstore_video2', 'bookstore_video3', 'bookstore_video5'], 'val': ['bookstore_video4'], 'test': ['bookstore_video6']},
    'C': {'train': ['bookstore_video0', 'bookstore_video1', 'bookstore_video2', 'bookstore_video4', 'bookstore_video6'], 'val': ['bookstore_video3'], 'test': ['bookstore_video5']}
}

class NavDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        return self.data_list[idx]

def custom_collate(batch):
    L = len(batch[0]['sequence'])
    B = len(batch)
    
    batched_sequences = []
    batched_targets = torch.zeros((B, L), dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.float)
    
    for t in range(L):
        graphs_t = [item['sequence'][t] for item in batch]
        pyg_batch_t = Batch.from_data_list(graphs_t)
        batched_sequences.append(pyg_batch_t)
        
        ptr = pyg_batch_t.ptr
        for i, item in enumerate(batch):
            orig_idx = item['target_indices'][t].item()
            if orig_idx != -1:
                batched_targets[i, t] = orig_idx + ptr[i]
            else:
                batched_targets[i, t] = -1
                
    for i, item in enumerate(batch):
        labels[i] = item['label'].item()
        
    return batched_sequences, batched_targets, labels

def pairwise_ranking_loss(preds, labels, margin=MARGIN):
    B = preds.size(0)
    if B < 2:
        return torch.tensor(0.0, device=preds.device)
        
    idx = torch.randperm(B, device=preds.device)
    idx1, idx2 = idx[:B//2], idx[B//2: (B//2)*2]
    
    p1, p2 = preds[idx1], preds[idx2]
    y1, y2 = labels[idx1], labels[idx2]
    
    sign_y = torch.sign(y1 - y2)
    diff_p = p1 - p2
    
    loss = F.relu(margin - diff_p * sign_y)
    
    # Mask out pairs that are exactly equal (no order to preserve)
    mask = (y1 != y2).float()
    if mask.sum() > 0:
        return (loss * mask).sum() / mask.sum()
    return torch.tensor(0.0, device=preds.device)

def print_bucketed_eval(preds, labels):
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    for low, high in buckets:
        mask = (labels_np >= low) & (labels_np <= high)
        if mask.any():
            mae = np.mean(np.abs(preds_np[mask] - labels_np[mask]))
            if mask.sum() > 1:
                spearman, _ = spearmanr(preds_np[mask], labels_np[mask])
            else:
                spearman = np.nan
            count = mask.sum()
            print(f"  Bucket [{low:.1f}, {high:.1f}]: count={count:<4} | MAE={mae:.4f} | Spearman={spearman:.4f}")

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_huber = 0.0
    
    with torch.no_grad():
        for seq_batch, target_idx, labels in loader:
            seq_batch = [b.to(device) for b in seq_batch]
            target_idx = target_idx.to(device)
            labels = labels.to(device)
            
            reg_out, _ = model(seq_batch, target_idx)
            
            huber = F.smooth_l1_loss(reg_out, labels, reduction='sum')
            total_huber += huber.item()
            
            all_preds.append(reg_out)
            all_labels.append(labels)
            
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    mae = F.l1_loss(all_preds, all_labels).item()
    mean_huber = total_huber / len(all_labels)
    spearman, _ = spearmanr(all_preds.cpu().numpy(), all_labels.cpu().numpy())
    
    return mean_huber, mae, spearman, all_preds, all_labels

def main():
    # Force CPU for PyG: MPS has extreme dispatch latency for small sequential graphs
    device = torch.device('cpu')
    print(f"Using device: {device} (Forced CPU for PyG performance)")
    
    print("Loading dataset...")
    full_data = torch.load(os.path.join("processed", "bookstore", "dataset.pt"), weights_only=False)
    
    results = []
    
    for rot_name, splits in ROTATIONS.items():
        print(f"\n{'='*50}")
        print(f"Running Rotation {rot_name}")
        print(f"Train: {splits['train']}")
        print(f"Val  : {splits['val']}")
        print(f"Test : {splits['test']}")
        print(f"{'='*50}")
        
        train_list = [s for s in full_data if s['scene_id'] in splits['train']]
        val_list = [s for s in full_data if s['scene_id'] in splits['val']]
        test_list = [s for s in full_data if s['scene_id'] in splits['test']]
        
        train_loader = DataLoader(NavDataset(train_list), batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(NavDataset(val_list), batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
        test_loader = DataLoader(NavDataset(test_list), batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
        
        model = NavUncertaintyModel().to(device)
        optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
        cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS-5)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
        
        best_val_huber = float('inf')
        best_train_huber_at_best_val = 0.0
        best_epoch = 0
        patience_counter = 0
        
        history = []
        
        print("\nStarting Training...")
        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            epoch_huber = 0.0
            
            for seq_batch, target_idx, labels in train_loader:
                seq_batch = [b.to(device) for b in seq_batch]
                target_idx = target_idx.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                reg_out, aux_out = model(seq_batch, target_idx)
                
                huber = F.smooth_l1_loss(reg_out, labels)
                rank_loss = pairwise_ranking_loss(reg_out, labels)
                
                aux_labels = (labels >= 0.5).long()
                aux_loss = F.cross_entropy(aux_out, aux_labels)
                
                loss = huber + LAMBDA_1 * rank_loss + LAMBDA_2 * aux_loss
                loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item() * len(labels)
                epoch_huber += huber.item() * len(labels)
                
            scheduler.step()
            
            train_loss_mean = epoch_loss / len(train_list)
            train_huber_mean = epoch_huber / len(train_list)
            
            val_huber, val_mae, val_spearman, _, _ = evaluate(model, val_loader, device)
            
            history.append({
                'epoch': epoch,
                'train_loss': train_loss_mean,
                'train_huber': train_huber_mean,
                'val_huber': val_huber,
                'val_mae': val_mae,
                'val_spearman': val_spearman
            })
            
            print(f"Epoch {epoch:03d} | Train Total: {train_loss_mean:.4f}, Train Huber: {train_huber_mean:.4f} | Val Huber: {val_huber:.4f}, Val MAE: {val_mae:.4f}, Val Spearman: {val_spearman:.4f}")
            
            if val_huber < best_val_huber:
                best_val_huber = val_huber
                best_train_huber_at_best_val = train_huber_mean
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), f"best_model_rotation_{rot_name}.pt")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
        # Save learning curve for this rotation
        pd.DataFrame(history).to_csv(f"training_curve_rotation_{rot_name}.csv", index=False)
        
        # Evaluate on Test for this rotation
        test_scene_name = splits['test'][0]
        print(f"\n--- Final Evaluation Rotation {rot_name} (Test Scene: {test_scene_name}) ---")
        model.load_state_dict(torch.load(f"best_model_rotation_{rot_name}.pt", weights_only=True))
        test_huber, test_mae, test_spearman, test_preds, test_labels = evaluate(model, test_loader, device)
        
        print(f"Test Huber   : {test_huber:.4f}")
        print(f"Test MAE     : {test_mae:.4f}")
        print(f"Test Spearman: {test_spearman:.4f}")
        print("\nBucketed MAE / Spearman (Test):")
        print_bucketed_eval(test_preds, test_labels)
        
        train_val_gap = best_val_huber - best_train_huber_at_best_val
        
        results.append({
            'Rotation': rot_name,
            'Test_Scene': test_scene_name,
            'Test_Huber': test_huber,
            'Test_MAE': test_mae,
            'Test_Spearman': test_spearman,
            'Best_Epoch': best_epoch,
            'Final_TrainVal_Huber_Gap': train_val_gap
        })
        
    print("\n" + "="*80)
    print("ROTATION COMPARISON TABLE")
    print("="*80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    results_df.to_csv("rotation_comparison.csv", index=False)
    print("\nSaved rotation_comparison.csv")

if __name__ == "__main__":
    main()
```

