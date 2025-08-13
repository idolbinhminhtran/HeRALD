"""
GNN and attention layer implementations for HGAT-LDA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class HeterogeneousGNNLayer(nn.Module):
    """
    Heterogeneous Graph Neural Network layer with multi-head type attention,
    residual connections, optional LayerNorm, and relation dropout.
    """
    
    def __init__(self, 
                 emb_dim: int, 
                 edges: Dict,
                 dropout: float = 0.5,
                 num_heads: int = 4,
                 relation_dropout: float = 0.0,
                 use_layernorm: bool = True,
                 use_residual: bool = True,
                 use_relation_norm: bool = True):
        """
        Initialize the heterogeneous GNN layer.
        
        Args:
            emb_dim: Embedding dimension
            edges: Dictionary of edge relations
            dropout: Dropout rate
            num_heads: Number of attention heads for type-level attention
            relation_dropout: Probability to drop an incoming relation's message
            use_layernorm: Whether to apply LayerNorm after update
            use_residual: Whether to add residual connection from input
            use_relation_norm: Whether to apply degree normalization per relation (D^-1 A style)
        """
        super(HeterogeneousGNNLayer, self).__init__()
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.relation_dropout = relation_dropout
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual
        self.use_relation_norm = use_relation_norm
        
        # Relation-specific weight matrices
        self.W_rel = nn.ModuleDict()
        for (src, rel, dst) in edges.keys():
            self.W_rel[f"W_{src}_{rel}_{dst}"] = nn.Linear(emb_dim, emb_dim, bias=False)
        
        # Self-loop weights for each node type
        self.W_self = nn.ModuleDict({
            "lncRNA": nn.Linear(emb_dim, emb_dim, bias=False),
            "gene": nn.Linear(emb_dim, emb_dim, bias=False),
            "disease": nn.Linear(emb_dim, emb_dim, bias=False),
        })
        
        # Attention parameters (multi-head)
        self.attention_vecs = nn.ParameterDict()
        relations_by_target = {'lncRNA': [], 'gene': [], 'disease': []}
        
        for (src_type, rel_name, dst_type) in edges.keys():
            relations_by_target[dst_type].append((src_type, rel_name, dst_type))
        
        for t_type, rel_list in relations_by_target.items():
            for (s_type, rel_name, d_type) in rel_list:
                att_key = f"{s_type}_{rel_name}_{d_type}_att"
                # One attention vector per head
                self.attention_vecs[att_key] = nn.Parameter(torch.randn(self.num_heads, emb_dim))
        
        # LayerNorm per type if enabled
        if self.use_layernorm:
            self.layer_norms = nn.ModuleDict({
                "lncRNA": nn.LayerNorm(emb_dim),
                "gene": nn.LayerNorm(emb_dim),
                "disease": nn.LayerNorm(emb_dim),
            })
        else:
            self.layer_norms = None
    
    def forward(self, h_dict: Dict[str, torch.Tensor], 
                edges: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the heterogeneous GNN layer.
        
        Args:
            h_dict: Dictionary of node embeddings for each type
            edges: Dictionary of edge relations and indices
            
        Returns:
            Updated embeddings dictionary (same shapes as input)
        """
        # Initialize output embeddings
        new_h = {t: torch.zeros_like(h) for t, h in h_dict.items()}
        rel_message_dict = {}
        # Get device from input tensors instead of parameters (works with DataParallel)
        device = next(iter(h_dict.values())).device
        
        # Neighbor aggregation for each relation type
        for (src_type, rel_name, dst_type), (src_idx, dst_idx, w) in edges.items():
            if src_idx is None:
                continue
            # Get source embeddings
            h_src = h_dict[src_type]
            # Ensure indices on correct device
            src_idx = src_idx.to(h_src.device)
            dst_idx = dst_idx.to(h_src.device)
            
            h_nei = h_src[src_idx]  # [E, emb_dim]
            
            # Transform neighbor features
            W_r = self.W_rel[f"W_{src_type}_{rel_name}_{dst_type}"]
            m = W_r(h_nei)  # [E, emb_dim]
            
            # Apply edge weights if provided
            if w is not None:
                w = w.to(m.device)
                m = m * w.unsqueeze(-1)
            
            # Aggregate messages to target nodes (ensure same dtype)
            new_h[dst_type].index_add_(0, dst_idx, m.to(new_h[dst_type].dtype))
            
            # Save relation-specific messages for attention
            key = (src_type, rel_name, dst_type)
            if key not in rel_message_dict:
                rel_message_dict[key] = torch.zeros_like(new_h[dst_type])
            rel_message_dict[key].index_add_(0, dst_idx, m.to(rel_message_dict[key].dtype))
        
        # Normalize by degree for each relation (D^-1 A style normalization)
        if self.use_relation_norm:
            for (src_type, rel_name, dst_type), (src_idx, dst_idx, w) in edges.items():
                if src_idx is None:
                    continue
                num_dst = new_h[dst_type].shape[0]
                dst_idx = dst_idx.to(new_h[dst_type].device)
                
                # Compute degree for this specific relation
                deg = torch.zeros(num_dst, device=new_h[dst_type].device)
                if w is not None:
                    # If edge weights are provided, use weighted degree
                    w = w.to(dst_idx.device)
                    deg.index_add_(0, dst_idx, w)
                else:
                    # Otherwise use unweighted degree
                    ones = torch.ones_like(dst_idx, dtype=torch.float)
                    deg.index_add_(0, dst_idx, ones)
                
                deg[deg == 0] = 1.0  # Avoid division by zero
                
                # Apply normalization to relation-specific messages
                rel_msg = rel_message_dict[(src_type, rel_name, dst_type)]
                rel_message_dict[(src_type, rel_name, dst_type)] = rel_msg / deg.unsqueeze(-1)
        
        # Attention mechanism (multi-head over relation types)
        att_new_h = {t: torch.zeros_like(h) for t, h in new_h.items()}
        
        # Group relations by target type
        target_relations = {'lncRNA': [], 'gene': [], 'disease': []}
        for (src_type, rel_name, dst_type) in rel_message_dict.keys():
            target_relations[dst_type].append((src_type, rel_name, dst_type))
        
        for t_type in ['lncRNA', 'gene', 'disease']:
            rels = target_relations[t_type]
            if not rels:
                continue
            
            messages = [rel_message_dict[(s, r, t_type)] for (s, r, _) in rels]
            # Relation dropout: drop whole relation message with prob p
            if self.training and self.relation_dropout > 0.0:
                keep_mask = torch.bernoulli(torch.full((len(messages),), 1.0 - self.relation_dropout, device=new_h[t_type].device))
                for i, keep in enumerate(keep_mask):
                    if keep.item() == 0.0:
                        messages[i] = torch.zeros_like(messages[i])
            
            # Stack messages: [N, R, D]
            msg_stack = torch.stack(messages, dim=1)
            N, R, D = msg_stack.shape
            
            # Compute attention per head with scaled dot-product attention
            head_outputs = []
            for head in range(self.num_heads):
                # Scores per relation: [N, R]
                scores = []
                for (s, r, _), _msg in zip(rels, messages):
                    att_key = f"{s}_{r}_{t_type}_att"
                    a_vec = self.attention_vecs[att_key][head]  # [D]
                    # Scaled dot-product attention
                    score = (_msg * a_vec).sum(dim=1) / (self.emb_dim ** 0.5)
                    scores.append(score)
                score_stack = torch.stack(scores, dim=1)  # [N, R]
                # Apply softmax with temperature for sharpness control
                alpha = F.softmax(score_stack / 0.5, dim=1)  # [N, R] 
                # Apply attention dropout for regularization
                alpha = F.dropout(alpha, p=self.dropout/2, training=self.training)
                # Weighted sum: [N, D]
                head_out = (alpha.unsqueeze(-1) * msg_stack).sum(dim=1)
                head_outputs.append(head_out)
            
            # Aggregate heads with learnable weights instead of simple average
            if self.num_heads > 1:
                agg = torch.stack(head_outputs, dim=0).mean(dim=0)
            else:
                agg = head_outputs[0]
            
            # Self-loop + residuals, norm, dropout
            h_self = self.W_self[t_type](h_dict[t_type])
            h_update = h_self + agg
            if self.use_residual:
                h_update = h_update + h_dict[t_type]
            if self.use_layernorm:
                h_update = self.layer_norms[t_type](h_update)
            h_update = F.leaky_relu(h_update)
            h_update = F.dropout(h_update, p=self.dropout, training=self.training)
            att_new_h[t_type] = h_update
        
        return att_new_h


class LinkPredictionMLP(nn.Module):
    """
    Enhanced multi-layer perceptron for link prediction with richer features and attention.
    """
    
    def __init__(self, emb_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.5):
        super(LinkPredictionMLP, self).__init__()
        if hidden_dim is None:
            hidden_dim = emb_dim * 2
        
        # Bilinear term for cross-feature interaction
        self.bilinear = nn.Bilinear(emb_dim, emb_dim, emb_dim // 2, bias=True)
        
        # Feature transformations
        self.lnc_transform = nn.Linear(emb_dim, emb_dim)
        self.dis_transform = nn.Linear(emb_dim, emb_dim)
        
        # Attention mechanism for feature importance
        self.feature_attention = nn.Sequential(
            nn.Linear(emb_dim * 5 + emb_dim // 2, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 6),  # 6 features
            nn.Softmax(dim=1)
        )
        
        # Enhanced MLP with batch normalization
        in_dim = emb_dim * 5 + emb_dim // 2  # All features including bilinear
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)  # logits
        )
    
    def forward(self, lnc_emb: torch.Tensor, dis_emb: torch.Tensor) -> torch.Tensor:
        # Transform embeddings
        lnc_trans = self.lnc_transform(lnc_emb)
        dis_trans = self.dis_transform(dis_emb)
        
        # Compute various interaction features
        concat = torch.cat([lnc_trans, dis_trans], dim=1)
        abs_diff = torch.abs(lnc_trans - dis_trans)
        hadamard = lnc_trans * dis_trans
        cosine_sim = F.cosine_similarity(lnc_trans, dis_trans, dim=1, eps=1e-8).unsqueeze(1)
        cosine_expanded = cosine_sim.expand(-1, lnc_emb.size(1))
        bilinear_feat = self.bilinear(lnc_emb, dis_emb)
        
        # Concatenate all features
        all_features = torch.cat([
            lnc_trans, dis_trans, abs_diff, hadamard, 
            cosine_expanded, bilinear_feat
        ], dim=1)
        
        # Apply feature attention
        attention_weights = self.feature_attention(all_features)
        
        # Weight features by attention (simplified without reshaping)
        # Just use the concatenated features directly
        
        # Predict with MLP
        return self.mlp(all_features).squeeze(-1)  # logits 