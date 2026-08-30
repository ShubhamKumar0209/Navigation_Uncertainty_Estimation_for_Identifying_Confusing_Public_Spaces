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
