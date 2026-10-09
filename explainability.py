import os
import glob
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

manifest_paths = glob.glob(os.path.join("processed_all", "*", "feature_manifest.json"))
if not manifest_paths:
    raise Exception("No feature_manifest.json found in processed_all/*/")
with open(manifest_paths[0], "r") as f:
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
                node_feats.append(np.zeros(len(FEATURE_NAMES)))
                
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
        
    os.makedirs(os.path.join("outputs", "explanations"), exist_ok=True)
    out_df = pd.DataFrame(zone_profiles)
    out_df.to_csv(os.path.join("outputs", "explanations", "zone_explanations.csv"), index=False)
    print(f"  Saved {len(out_df)} zone explanations to outputs/explanations/zone_explanations.csv")
    
    # Print a few examples
    for i in range(min(3, len(out_df))):
        print("  " + out_df.iloc[i]['description'])

def generate_visualizations(dataset, X_train, explainer):
    print("\nGenerating Visualizations for top 3 highest NUS samples...")
    vis_dir = os.path.join("outputs", "explanations", "explanation_examples")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Find top 3 predictions
    model = NavUncertaintyModel()
    model.load_state_dict(torch.load(os.path.join("models", "global", "best_model.pt"), weights_only=True))
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
    
    pos_df = pd.read_csv(os.path.join("outputs", "heatmaps", "predictions_with_positions.csv"))
    
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
        plt.savefig(os.path.join(vis_dir, f"{scene}_{agent}_{w_start}.png"))
        plt.close()
        
    print("  Saved visualizations to explanation_examples/")

def main():
    device = torch.device('cpu')
    
    dataset_path = os.path.join("combined", "dataset", "dataset.pt")
    dataset = torch.load(dataset_path, weights_only=False)
    
    model = NavUncertaintyModel().to(device)
    model.load_state_dict(torch.load(os.path.join("models", "global", "best_model.pt"), weights_only=True))
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
        
    flagged_zones_path = os.path.join("outputs", "heatmaps", "flagged_zones.csv")
    preds_pos_path = os.path.join("outputs", "heatmaps", "predictions_with_positions.csv")
    profile_flagged_zones(flagged_zones_path, preds_pos_path, X_train, explainer, dataset)
    
    generate_visualizations(dataset, X_train, explainer)

if __name__ == "__main__":
    main()