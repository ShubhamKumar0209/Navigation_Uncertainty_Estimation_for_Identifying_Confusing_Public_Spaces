import os
import sys
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
    full_data = torch.load(os.path.join("combined", "dataset", "dataset.pt"), weights_only=False)
    
    # Extract unique scene_ids (videos)
    unique_scenes = sorted(list(set([s['scene_id'] for s in full_data])))
    print(f"Total unique videos found globally: {len(unique_scenes)}")
    
    # Set a fixed seed for reproducible splits
    np.random.seed(42)
    np.random.shuffle(unique_scenes)
    
    # Stratified split per location
    train_scenes = []
    val_scenes = []
    test_scenes = []
    
    # Group by location
    loc_to_scenes = {}
    for s in unique_scenes:
        loc = s.split('_')[0]
        if loc not in loc_to_scenes:
            loc_to_scenes[loc] = []
        loc_to_scenes[loc].append(s)
        
    print(f"\n{'='*50}")
    print(f"STRATIFIED SPLIT CONFIGURATION")
    for loc in sorted(loc_to_scenes.keys()):
        scenes = loc_to_scenes[loc]
        L = len(scenes)
        
        if L == 1:
            print(f"WARNING: Location {loc} has only 1 sub-video. Assigning to train. Cannot have a held-out score in this split.")
            train_scenes.extend(scenes)
            loc_train, loc_val, loc_test = scenes, [], []
        else:
            # We want to match ~79/9/12 overall (which was roughly 46/5/7 out of 58).
            n_val = max(1, int(round(L * 5 / 58)))
            n_test = max(1, int(round(L * 7 / 58)))
            n_train = L - n_val - n_test
            
            loc_train = scenes[:n_train]
            loc_val = scenes[n_train:n_train + n_val]
            loc_test = scenes[n_train + n_val:]
            
            train_scenes.extend(loc_train)
            val_scenes.extend(loc_val)
            test_scenes.extend(loc_test)
            
        print(f"Location {loc} ({L}):")
        print(f"  Train: {loc_train}")
        print(f"  Val  : {loc_val}")
        print(f"  Test : {loc_test}")
        
    print(f"{'='*50}")
    print(f"Total Train videos: {len(train_scenes)}")
    print(f"Total Val videos  : {len(val_scenes)}")
    print(f"Total Test videos : {len(test_scenes)}")
    print(f"{'='*50}\n")
    
    train_list = [s for s in full_data if s['scene_id'] in train_scenes]
    val_list = [s for s in full_data if s['scene_id'] in val_scenes]
    test_list = [s for s in full_data if s['scene_id'] in test_scenes]
    
    # For per-location evaluation, we also want to evaluate the whole dataset globally.
    # The prompt says: "Evaluate the single global model on the single global test set... Additionally... descriptive per-location evaluation statistics from the same global model".
    
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
    
    os.makedirs(os.path.join("models", "global"), exist_ok=True)
    best_model_path = os.path.join("models", "global", "best_model.pt")
    
    eval_only = '--eval-only' in sys.argv
    
    if not eval_only:
        print("Starting Global Training...")
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
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
            os.makedirs(os.path.join("outputs", "metrics"), exist_ok=True)
            pd.DataFrame(history).to_csv(os.path.join("outputs", "metrics", "global_training_curve.csv"), index=False)
    else:
        print("\nSkipping training (--eval-only flag detected).")
        best_epoch = "N/A"
        best_val_huber = 0.0
        best_train_huber_at_best_val = 0.0
    
    print("\n--- Final Global Test Evaluation ---")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    test_huber, test_mae, test_spearman, test_preds, test_labels = evaluate(model, test_loader, device)
    
    print(f"Test Huber   : {test_huber:.4f}")
    print(f"Test MAE     : {test_mae:.4f}")
    print(f"Test Spearman: {test_spearman:.4f}")
    print("\nBucketed MAE / Spearman (Global Test):")
    print_bucketed_eval(test_preds, test_labels)
    
    if not eval_only:
        train_val_gap = best_val_huber - best_train_huber_at_best_val
    else:
        train_val_gap = 0.0
    
    with open(os.path.join("outputs", "metrics", "global_evaluation.txt"), "w") as f:
        f.write("GLOBAL EVALUATION RESULTS\n")
        f.write("=========================\n")
        f.write(f"Test Huber: {test_huber:.4f}\n")
        f.write(f"Test MAE: {test_mae:.4f}\n")
        f.write(f"Test Spearman: {test_spearman:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        if not eval_only:
            f.write(f"Final Train/Val Huber Gap: {train_val_gap:.4f}\n")
    
    print("\n--- Per-Location Breakdown ---")
    location_results = []
    
    # We group the entire dataset by location (not just test, but the prompt implies evaluation breakdown. 
    # Usually, evaluating the whole dataset or just the test set? "These are analysis breakdowns of the same global model's predictions"
    # To be safe and have enough data, we will evaluate the global model on the full combined dataset per location, or just the test subsets.
    # The prompt says "Generate descriptive per-location evaluation statistics from the same global model".
    # Since some locations might not even be in the test set due to random split, we will evaluate on the FULL dataset for descriptive breakdowns.
    
    all_locations = sorted(list(set([s['scene_id'].split('_')[0] for s in full_data])))
    
    for loc in all_locations:
        loc_test_scenes = [s for s in test_scenes if s.split('_')[0] == loc]
        loc_data = [s for s in full_data if s['scene_id'] in loc_test_scenes]
        
        if len(loc_data) == 0:
            print(f"Location {loc}: N/A (no held-out test data for this location. Sub-videos: {loc_test_scenes})")
            location_results.append({
                'Location': loc,
                'Test_SubVideos': 'None',
                'Samples': 0,
                'MAE': 'N/A',
                'Huber': 'N/A',
                'Spearman': 'N/A'
            })
            continue
        
        print(f"Location {loc}: {len(loc_data)} samples. Sub-videos in test: {loc_test_scenes}")
        loc_loader = DataLoader(NavDataset(loc_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
        loc_huber, loc_mae, loc_spearman, _, _ = evaluate(model, loc_loader, device)
        
        location_results.append({
            'Location': loc,
            'Test_SubVideos': ', '.join(loc_test_scenes),
            'Samples': len(loc_data),
            'MAE': loc_mae,
            'Huber': loc_huber,
            'Spearman': loc_spearman
        })
        
    loc_df = pd.DataFrame(location_results)
    print("\n" + loc_df.to_string(index=False))
    
    loc_df.to_csv(os.path.join("outputs", "metrics", "location_breakdown.csv"), index=False)
    print("\nSaved evaluation metrics to outputs/metrics/")

if __name__ == "__main__":
    main()
