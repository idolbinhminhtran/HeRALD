"""
Main HGAT-LDA model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import warnings

from models.layers import HeterogeneousGNNLayer, LinkPredictionMLP


class HGAT_LDA(nn.Module):
    """
    HGAT-LDA: Heterogeneous Graph Attention Network for lncRNA-disease prediction.
    """
    
    def __init__(self, 
                 num_lncRNAs: int,
                 num_genes: int, 
                 num_diseases: int,
                 edges: Dict,
                 emb_dim: int = 128,
                 num_layers: int = 4,
                 dropout: float = 0.3,
                 num_heads: int = 8,
                 relation_dropout: float = 0.1,
                 use_layernorm: bool = True,
                 use_residual: bool = True,
                 use_relation_norm: bool = True,
                 init_from_similarity: Optional[Dict[str, torch.Tensor]] = None):
        """
        Initialize the HGAT-LDA model.
        
        Args:
            num_lncRNAs: Number of lncRNA nodes
            num_genes: Number of gene nodes
            num_diseases: Number of disease nodes
            edges: Dictionary of edge relations
            emb_dim: Dimensionality of node embeddings
            num_layers: Number of GNN layers to stack
            dropout: Dropout rate for GNN layers and prediction MLP
            num_heads: Multi-head attention heads
            relation_dropout: Relation-level dropout probability
            use_layernorm: Whether to apply LayerNorm in layers
            use_residual: Whether to use residual connections
            use_relation_norm: Whether to use relation-level normalization (D^-1 A)
            init_from_similarity: Optional dict with keys 'lncRNA','gene','disease' tensors for init
        """
        super(HGAT_LDA, self).__init__()
        
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Node embeddings with better initialization
        self.lnc_embed = nn.Embedding(num_lncRNAs, emb_dim)
        self.gene_embed = nn.Embedding(num_genes, emb_dim)
        self.disease_embed = nn.Embedding(num_diseases, emb_dim)
        
        # Add positional encodings for better node distinction
        self.lnc_pos = nn.Parameter(torch.randn(1, emb_dim) * 0.02)
        self.gene_pos = nn.Parameter(torch.randn(1, emb_dim) * 0.02)
        self.disease_pos = nn.Parameter(torch.randn(1, emb_dim) * 0.02)
        
        # Initialize embeddings
        if init_from_similarity is not None:
            # Expect tensors sized [num_nodes, feat_dim]; project to emb_dim if needed
            proj = lambda x: x if x.shape[1] == emb_dim else nn.Linear(x.shape[1], emb_dim, bias=False)(x)
            with torch.no_grad():
                self.lnc_embed.weight.copy_(proj(init_from_similarity['lncRNA']))
                self.gene_embed.weight.copy_(proj(init_from_similarity['gene']))
                self.disease_embed.weight.copy_(proj(init_from_similarity['disease']))
        else:
            # Better initialization with scaled Xavier
            nn.init.xavier_uniform_(self.lnc_embed.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.gene_embed.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.disease_embed.weight, gain=nn.init.calculate_gain('relu'))
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            HeterogeneousGNNLayer(
                emb_dim, edges, dropout,
                num_heads=num_heads,
                relation_dropout=relation_dropout,
                use_layernorm=use_layernorm,
                use_residual=use_residual,
                use_relation_norm=use_relation_norm
            )
            for _ in range(num_layers)
        ])
        
        # Link prediction MLP (returns logits)
        self.prediction_mlp = LinkPredictionMLP(emb_dim, emb_dim, dropout)
    
    def forward(self, 
                lnc_indices: torch.Tensor, 
                dis_indices: torch.Tensor, 
                edges: Dict) -> torch.Tensor:
        """
        Forward pass: returns prediction logits for BCEWithLogitsLoss.
        """
        # Initialize embeddings for each node type with positional encodings
        h_dict = {
            'lncRNA': self.lnc_embed.weight + self.lnc_pos,
            'gene': self.gene_embed.weight + self.gene_pos,
            'disease': self.disease_embed.weight + self.disease_pos
        }
        
        # Apply dropout to initial embeddings for regularization
        if self.training:
            h_dict = {k: F.dropout(v, p=self.dropout/2, training=True) for k, v in h_dict.items()}
        
        # GNN message passing
        for layer in self.gnn_layers:
            h_dict = layer(h_dict, edges)
        
        # Gather pair embeddings
        lnc_emb = h_dict['lncRNA'][lnc_indices]
        dis_emb = h_dict['disease'][dis_indices]
        
        # Return logits (no sigmoid here)
        score_logit = self.prediction_mlp(lnc_emb, dis_emb)
        return score_logit
    
    def get_embeddings(self, edges: Dict) -> Dict[str, torch.Tensor]:
        h_dict = {
            'lncRNA': self.lnc_embed.weight + self.lnc_pos,
            'gene': self.gene_embed.weight + self.gene_pos,
            'disease': self.disease_embed.weight + self.disease_pos
        }
        # Apply dropout if in training mode
        if self.training:
            h_dict = {k: F.dropout(v, p=self.dropout/2, training=True) for k, v in h_dict.items()}
        for layer in self.gnn_layers:
            h_dict = layer(h_dict, edges)
        return h_dict
    
    def predict_all_pairs(self, edges: Dict) -> torch.Tensor:
        h_dict = self.get_embeddings(edges)
        lnc_emb = h_dict['lncRNA']
        dis_emb = h_dict['disease']
        num_l = lnc_emb.size(0)
        num_d = dis_emb.size(0)
        lnc_flat = lnc_emb.unsqueeze(1).expand(-1, num_d, -1).reshape(-1, self.emb_dim)
        dis_flat = dis_emb.unsqueeze(0).expand(num_l, -1, -1).reshape(-1, self.emb_dim)
        logits = self.prediction_mlp(lnc_flat, dis_flat).reshape(num_l, num_d)
        return torch.sigmoid(logits)  # probability matrix for convenience 