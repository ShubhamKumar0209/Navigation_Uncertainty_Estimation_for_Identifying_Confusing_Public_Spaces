import os
import json
import glob
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

def load_data():
    manifest_paths = glob.glob(os.path.join("processed_all", "*", "feature_manifest.json"))
    if not manifest_paths:
        raise Exception("No feature_manifest.json found in processed_all/*/")
    with open(manifest_paths[0], 'r') as f:
        manifest = json.load(f)
    gnn_features = manifest["GNN_INPUT_FEATURES"]
    
    ff_path = os.path.join("combined", "labels", "features_final.csv")
    targets_df = pd.read_csv(ff_path)
    
    nf_paths = glob.glob(os.path.join("processed_all", "*", "node_features_with_graph.csv"))
    nodes_dfs = []
    for p in nf_paths:
        nodes_dfs.append(pd.read_csv(p))
    nodes_df = pd.concat(nodes_dfs, ignore_index=True)
    
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
        
    graph_files = glob.glob(os.path.join("processed_all", "*", "graphs", "*.gpickle"))
    
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
    os.makedirs(os.path.join("combined", "dataset"), exist_ok=True)
    out_file = os.path.join("combined", "dataset", "dataset.pt")
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
