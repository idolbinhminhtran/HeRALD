"""
Graph construction utilities for HGAT-LDA model.
"""

import torch
from typing import Dict, Tuple, Optional


def _row_normalize(mat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    row_sum = mat.sum(dim=1, keepdim=True)
    row_sum = torch.clamp(row_sum, min=eps)
    return mat / row_sum


def _topk_per_row(mat: torch.Tensor, k: int) -> torch.Tensor:
    if k is None or k <= 0 or k >= mat.size(1):
        return mat
    vals, idx = torch.topk(mat, k, dim=1)
    out = torch.zeros_like(mat)
    out.scatter_(1, idx, vals)
    return out


def matrix_to_edges(adj_matrix: torch.Tensor, 
                   symmetric: bool = True, 
                   threshold: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src_idx, dst_idx = (adj_matrix > threshold).nonzero(as_tuple=True)
    weights = adj_matrix[src_idx, dst_idx]
    if symmetric:
        src_idx_rev = dst_idx
        dst_idx_rev = src_idx
        weights_rev = weights
        src_idx = torch.cat([src_idx, src_idx_rev], dim=0)
        dst_idx = torch.cat([dst_idx, dst_idx_rev], dim=0)
        weights = torch.cat([weights, weights_rev], dim=0)
    return src_idx, dst_idx, weights


def construct_heterogeneous_graph(
    data_dict: Dict[str, torch.Tensor],
    sim_topk: Optional[int] = None,
    sim_row_normalize: bool = True,
    sim_threshold: float = 0.0
) -> Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """
    Construct heterogeneous graph with optional similarity pruning/normalization.
    """
    # Extract matrices
    LL_sim = data_dict['LL_sim']
    GG_sim = data_dict['GG_sim']
    DD_sim = data_dict['DD_sim']
    lnc_gene_assoc = data_dict['lnc_gene_assoc']
    gene_disease_assoc = data_dict['gene_disease_assoc']
    lnc_disease_assoc = data_dict['lnc_disease_assoc']
    
    # Optional normalization / pruning
    if sim_row_normalize:
        LL_sim = _row_normalize(LL_sim)
        GG_sim = _row_normalize(GG_sim)
        DD_sim = _row_normalize(DD_sim)
    if sim_topk and sim_topk > 0:
        LL_sim = _topk_per_row(LL_sim, sim_topk)
        GG_sim = _topk_per_row(GG_sim, sim_topk)
        DD_sim = _topk_per_row(DD_sim, sim_topk)
    
    edges = {}
    
    # Process similarity networks into edge lists
    lnc_src, lnc_dst, lnc_w = matrix_to_edges(LL_sim, symmetric=True, threshold=sim_threshold)
    gene_src, gene_dst, gene_w = matrix_to_edges(GG_sim, symmetric=True, threshold=sim_threshold)
    dis_src, dis_dst, dis_w = matrix_to_edges(DD_sim, symmetric=True, threshold=sim_threshold)
    
    edges[('lncRNA', 'lnc_sim', 'lncRNA')] = (lnc_src, lnc_dst, lnc_w)
    edges[('gene', 'gene_sim', 'gene')] = (gene_src, gene_dst, gene_w)
    edges[('disease', 'disease_sim', 'disease')] = (dis_src, dis_dst, dis_w)
    
    # lncRNA-gene associations (transpose to lnc x gene)
    lnc_gene_assoc_t = lnc_gene_assoc.t()
    lnc_gene_src, lnc_gene_dst = (lnc_gene_assoc_t > 0).nonzero(as_tuple=True)
    edges[('lncRNA', 'lnc_gene', 'gene')] = (lnc_gene_src, lnc_gene_dst, None)
    edges[('gene', 'gene_lnc', 'lncRNA')] = (lnc_gene_dst, lnc_gene_src, None)
    
    # gene-disease associations
    gene_dis_src, gene_dis_dst = (gene_disease_assoc > 0).nonzero(as_tuple=True)
    edges[('gene', 'gene_disease', 'disease')] = (gene_dis_src, gene_dis_dst, None)
    edges[('disease', 'disease_gene', 'gene')] = (gene_dis_dst, gene_dis_src, None)
    
    # lncRNA-disease associations (transpose to lnc x disease)
    lnc_disease_assoc_t = lnc_disease_assoc.t()
    lnc_dis_src, lnc_dis_dst = (lnc_disease_assoc_t > 0).nonzero(as_tuple=True)
    edges[('lncRNA', 'lnc_disease', 'disease')] = (lnc_dis_src, lnc_dis_dst, None)
    edges[('disease', 'disease_lnc', 'lncRNA')] = (lnc_dis_dst, lnc_dis_src, None)
    
    return edges


def get_positive_pairs(lnc_disease_assoc: torch.Tensor) -> torch.Tensor:
    lnc_disease_assoc_t = lnc_disease_assoc.t()
    pairs = torch.nonzero(lnc_disease_assoc_t > 0, as_tuple=False)
    return pairs


def generate_negative_pairs(pos_pairs: torch.Tensor, 
                          num_lncRNAs: int, 
                          num_diseases: int,
                          num_negatives: Optional[int] = None) -> torch.Tensor:
    if num_negatives is None:
        num_negatives = pos_pairs.size(0)
    known_set = set((int(i), int(j)) for i, j in pos_pairs)
    all_pairs = [(i, j) for i in range(num_lncRNAs) for j in range(num_diseases)]
    neg_pairs_all = [torch.tensor([i, j]) for (i, j) in all_pairs if (i, j) not in known_set]
    neg_pairs_all = torch.stack(neg_pairs_all)
    if num_negatives < len(neg_pairs_all):
        indices = torch.randperm(len(neg_pairs_all))[:num_negatives]
        return neg_pairs_all[indices]
    else:
        return neg_pairs_all 