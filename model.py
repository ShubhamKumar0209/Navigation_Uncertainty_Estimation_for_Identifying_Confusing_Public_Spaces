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
        
    def _encode_snapshot(self, batch_data, return_attention=False):
        """Passes a single graph batch through the GAT layers."""
        x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
        
        if x.size(0) == 0:
            if return_attention:
                return torch.empty((0, 64), device=x.device), (torch.empty((2,0), dtype=torch.long, device=x.device), torch.empty((0,1), device=x.device))
            return torch.empty((0, 64), device=x.device)
            
        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.dropout(x)
        
        if return_attention:
            x, attn_weights = self.gat2(x, edge_index, edge_attr, return_attention_weights=True)
        else:
            x = self.gat2(x, edge_index, edge_attr)
            attn_weights = None
            
        x = F.elu(x)
        
        if return_attention:
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
        attentions = []
        
        for t in range(L):
            batch_data = sequence_batches[t]
            # Encode spatial graphs for timestep t
            if return_attention:
                node_embeddings, attn = self._encode_snapshot(batch_data, return_attention=True)
                attentions.append(attn)
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
            return reg_out, aux_out, attentions
        return reg_out, aux_out
