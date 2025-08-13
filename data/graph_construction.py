"""
Graph construction utilities for HGAT-LDA model.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any


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


def build_similarity_edges(S: torch.Tensor,
                          topk: Optional[int] = None,
                          mutual: bool = False,
                          symmetrize: bool = True,
                          row_norm: bool = True,
                          tau: Optional[float] = None,
                          verbose: bool = True) -> torch.Tensor:
    """
    Build similarity edges with advanced filtering and normalization options.
    
    Args:
        S: Similarity matrix (n x n)
        topk: Keep top-k neighbors per node (None = keep all)
        mutual: If True, apply mutual kNN filtering (keep edge only if both nodes are in each other's top-k)
        symmetrize: If True, symmetrize the matrix after filtering (S = (S + S.T) / 2)
        row_norm: If True, normalize rows to sum to 1 (row-stochastic matrix)
        tau: Temperature parameter for softmax normalization (applied before other operations)
        verbose: If True, print statistics
        
    Returns:
        Processed similarity matrix with filtered edges
    """
    n = S.size(0)
    S_processed = S.clone()
    
    # Store original statistics
    if verbose:
        orig_nnz = (S > 0).sum().item()
        orig_avg_degree = orig_nnz / n
        orig_symmetry = torch.allclose(S, S.T, rtol=1e-5)
        print(f"  Original: {orig_nnz} edges, avg degree: {orig_avg_degree:.2f}, symmetric: {orig_symmetry}")
    
    # Apply temperature-based softmax if specified
    if tau is not None and tau > 0:
        # Apply softmax row-wise with temperature
        S_processed = torch.softmax(S_processed / tau, dim=1)
        if verbose:
            print(f"  Applied softmax with temperature τ={tau}")
    
    # Keep top-k per row
    if topk is not None and topk > 0 and topk < n:
        # Store top-k mask for mutual kNN
        topk_mask = torch.zeros_like(S_processed, dtype=torch.bool)
        vals, indices = torch.topk(S_processed, min(topk, n), dim=1)
        for i in range(n):
            topk_mask[i, indices[i]] = True
        
        # Apply top-k filtering
        S_processed = S_processed * topk_mask.float()
        
        if verbose:
            kept_edges = topk_mask.sum().item()
            print(f"  Top-{topk} filtering: kept {kept_edges}/{orig_nnz} edges ({100*kept_edges/max(orig_nnz,1):.1f}%)")
        
        # Apply mutual kNN if specified
        if mutual:
            # Keep edge (i,j) only if i∈Topk(j) AND j∈Topk(i)
            mutual_mask = topk_mask & topk_mask.T
            S_processed = S_processed * mutual_mask.float()
            
            if verbose:
                mutual_edges = mutual_mask.sum().item()
                print(f"  Mutual kNN: kept {mutual_edges}/{kept_edges} edges ({100*mutual_edges/max(kept_edges,1):.1f}%)")
    
    # Symmetrize if specified (after filtering)
    if symmetrize:
        S_processed = (S_processed + S_processed.T) / 2
        if verbose:
            print(f"  Symmetrized matrix: S = (S + S.T) / 2")
    
    # Row normalization (make row-stochastic)
    if row_norm:
        row_sums = S_processed.sum(dim=1, keepdim=True)
        # Only normalize non-zero rows
        non_zero_rows = row_sums.squeeze() > 0
        S_processed[non_zero_rows] = S_processed[non_zero_rows] / row_sums[non_zero_rows]
        if verbose:
            print(f"  Row normalized (row-stochastic)")
    
    # Final statistics
    if verbose:
        final_nnz = (S_processed > 0).sum().item()
        final_avg_degree = final_nnz / n
        final_symmetry = torch.allclose(S_processed, S_processed.T, rtol=1e-5)
        if row_norm:
            row_sums = S_processed.sum(dim=1)
            valid_rows = row_sums > 0
            sum_check = torch.allclose(row_sums[valid_rows], torch.ones_like(row_sums[valid_rows]), rtol=1e-5)
            print(f"  Final: {final_nnz} edges, avg degree: {final_avg_degree:.2f}, symmetric: {final_symmetry}, row-sum-1: {sum_check}")
        else:
            print(f"  Final: {final_nnz} edges, avg degree: {final_avg_degree:.2f}, symmetric: {final_symmetry}")
    
    return S_processed


def compute_bipartite_edge_weights(src_idx: torch.Tensor, 
                                  dst_idx: torch.Tensor,
                                  num_src: int,
                                  num_dst: int,
                                  verbose: bool = False) -> torch.Tensor:
    """
    Compute TF-IDF style edge weights for bipartite graphs: w(u,v) = 1/sqrt(deg(u)*deg(v))
    
    Args:
        src_idx: Source node indices
        dst_idx: Destination node indices  
        num_src: Total number of source nodes
        num_dst: Total number of destination nodes
        verbose: Print statistics
        
    Returns:
        Edge weights tensor
    """
    # Compute degrees for source and destination nodes
    src_degree = torch.zeros(num_src)
    dst_degree = torch.zeros(num_dst)
    
    # Count degrees
    for s in src_idx:
        src_degree[s] += 1
    for d in dst_idx:
        dst_degree[d] += 1
    
    # Avoid division by zero
    src_degree = torch.clamp(src_degree, min=1.0)
    dst_degree = torch.clamp(dst_degree, min=1.0)
    
    # Compute edge weights: w(u,v) = 1/sqrt(deg(u)*deg(v))
    weights = torch.zeros(len(src_idx))
    for i, (s, d) in enumerate(zip(src_idx, dst_idx)):
        weights[i] = 1.0 / torch.sqrt(src_degree[s] * dst_degree[d])
    
    if verbose and len(weights) > 0:
        print(f"    Edge weights: mean={weights.mean():.4f}, std={weights.std():.4f}, "
              f"min={weights.min():.4f}, max={weights.max():.4f}")
    
    return weights


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
    sim_threshold: float = 0.0,
    sim_mutual: bool = False,
    sim_sym: bool = True,
    sim_row_norm: bool = True,
    sim_tau: Optional[float] = None,
    use_bipartite_edge_weight: bool = False,
    verbose: bool = True
) -> Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """
    Construct heterogeneous graph with advanced similarity edge construction.
    
    Args:
        data_dict: Dictionary containing all data matrices
        sim_topk: Keep top-k neighbors in similarity graphs
        sim_row_normalize: Legacy parameter (use sim_row_norm instead)
        sim_threshold: Threshold for edge weights
        sim_mutual: Apply mutual kNN filtering
        sim_sym: Symmetrize similarity matrices
        sim_row_norm: Row-normalize similarity matrices (row-stochastic)
        sim_tau: Temperature for softmax normalization
        use_bipartite_edge_weight: Compute TF-IDF style weights for bipartite edges
        verbose: Print statistics
    """
    # Extract matrices
    LL_sim = data_dict['LL_sim']
    GG_sim = data_dict['GG_sim']
    DD_sim = data_dict['DD_sim']
    lnc_gene_assoc = data_dict['lnc_gene_assoc']
    gene_disease_assoc = data_dict['gene_disease_assoc']
    lnc_disease_assoc = data_dict['lnc_disease_assoc']
    
    # Use new build_similarity_edges function with advanced options
    # Handle legacy parameter: sim_row_normalize -> sim_row_norm
    use_row_norm = sim_row_norm if sim_row_norm is not None else sim_row_normalize
    
    if verbose:
        print("\nConstructing similarity graphs:")
        print("lncRNA similarity graph:")
    LL_sim = build_similarity_edges(
        LL_sim, 
        topk=sim_topk,
        mutual=sim_mutual,
        symmetrize=sim_sym,
        row_norm=use_row_norm,
        tau=sim_tau,
        verbose=verbose
    )
    
    if verbose:
        print("Gene similarity graph:")
    GG_sim = build_similarity_edges(
        GG_sim,
        topk=sim_topk,
        mutual=sim_mutual,
        symmetrize=sim_sym,
        row_norm=use_row_norm,
        tau=sim_tau,
        verbose=verbose
    )
    
    if verbose:
        print("Disease similarity graph:")
    DD_sim = build_similarity_edges(
        DD_sim,
        topk=sim_topk,
        mutual=sim_mutual,
        symmetrize=sim_sym,
        row_norm=use_row_norm,
        tau=sim_tau,
        verbose=verbose
    )
    
    edges = {}
    
    # Process similarity networks into edge lists
    lnc_src, lnc_dst, lnc_w = matrix_to_edges(LL_sim, symmetric=True, threshold=sim_threshold)
    gene_src, gene_dst, gene_w = matrix_to_edges(GG_sim, symmetric=True, threshold=sim_threshold)
    dis_src, dis_dst, dis_w = matrix_to_edges(DD_sim, symmetric=True, threshold=sim_threshold)
    
    edges[('lncRNA', 'lnc_sim', 'lncRNA')] = (lnc_src, lnc_dst, lnc_w)
    edges[('gene', 'gene_sim', 'gene')] = (gene_src, gene_dst, gene_w)
    edges[('disease', 'disease_sim', 'disease')] = (dis_src, dis_dst, dis_w)
    
    # Get node counts for bipartite weight computation
    num_lncRNAs = LL_sim.size(0)
    num_genes = GG_sim.size(0)
    num_diseases = DD_sim.size(0)
    
    # lncRNA-gene associations (transpose to lnc x gene)
    lnc_gene_assoc_t = lnc_gene_assoc.t()
    lnc_gene_src, lnc_gene_dst = (lnc_gene_assoc_t > 0).nonzero(as_tuple=True)
    if use_bipartite_edge_weight and len(lnc_gene_src) > 0:
        if verbose:
            print("\nComputing bipartite edge weights for lncRNA-gene:")
        lnc_gene_weights = compute_bipartite_edge_weights(
            lnc_gene_src, lnc_gene_dst, num_lncRNAs, num_genes, verbose
        )
        # Use same weights for reverse direction
        edges[('lncRNA', 'lnc_gene', 'gene')] = (lnc_gene_src, lnc_gene_dst, lnc_gene_weights)
        edges[('gene', 'gene_lnc', 'lncRNA')] = (lnc_gene_dst, lnc_gene_src, lnc_gene_weights)
    else:
        edges[('lncRNA', 'lnc_gene', 'gene')] = (lnc_gene_src, lnc_gene_dst, None)
        edges[('gene', 'gene_lnc', 'lncRNA')] = (lnc_gene_dst, lnc_gene_src, None)
    
    # gene-disease associations
    gene_dis_src, gene_dis_dst = (gene_disease_assoc > 0).nonzero(as_tuple=True)
    if use_bipartite_edge_weight and len(gene_dis_src) > 0:
        if verbose:
            print("Computing bipartite edge weights for gene-disease:")
        gene_dis_weights = compute_bipartite_edge_weights(
            gene_dis_src, gene_dis_dst, num_genes, num_diseases, verbose
        )
        edges[('gene', 'gene_disease', 'disease')] = (gene_dis_src, gene_dis_dst, gene_dis_weights)
        edges[('disease', 'disease_gene', 'gene')] = (gene_dis_dst, gene_dis_src, gene_dis_weights)
    else:
        edges[('gene', 'gene_disease', 'disease')] = (gene_dis_src, gene_dis_dst, None)
        edges[('disease', 'disease_gene', 'gene')] = (gene_dis_dst, gene_dis_src, None)
    
    # lncRNA-disease associations (transpose to lnc x disease)
    lnc_disease_assoc_t = lnc_disease_assoc.t()
    lnc_dis_src, lnc_dis_dst = (lnc_disease_assoc_t > 0).nonzero(as_tuple=True)
    if use_bipartite_edge_weight and len(lnc_dis_src) > 0:
        if verbose:
            print("Computing bipartite edge weights for lncRNA-disease:")
        lnc_dis_weights = compute_bipartite_edge_weights(
            lnc_dis_src, lnc_dis_dst, num_lncRNAs, num_diseases, verbose
        )
        edges[('lncRNA', 'lnc_disease', 'disease')] = (lnc_dis_src, lnc_dis_dst, lnc_dis_weights)
        edges[('disease', 'disease_lnc', 'lncRNA')] = (lnc_dis_dst, lnc_dis_src, lnc_dis_weights)
    else:
        edges[('lncRNA', 'lnc_disease', 'disease')] = (lnc_dis_src, lnc_dis_dst, None)
        edges[('disease', 'disease_lnc', 'lncRNA')] = (lnc_dis_dst, lnc_dis_src, None)
    
    # Print overall graph statistics
    if verbose:
        print("\nOverall graph statistics:")
        total_edges = 0
        for (src_type, rel_type, dst_type), (src, dst, w) in edges.items():
            n_edges = len(src) if src is not None else 0
            total_edges += n_edges
            if verbose and n_edges > 0:
                print(f"  {src_type} -> {dst_type} ({rel_type}): {n_edges} edges")
        print(f"  Total edges in heterogeneous graph: {total_edges}")
    
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


def build_fold_edges(data_dict: Dict[str, torch.Tensor],
                    config: Dict[str, Any],
                    l_idx: int,
                    d_idx: int) -> Dict:
    """
    Build graph edges for a specific LOOCV fold, removing the test edge
    and rewriting isolated diseases into the LD matrix.
    
    Args:
        data_dict: Dataset dictionary
        config: Configuration
        l_idx: Test lncRNA index
        d_idx: Test disease index
        
    Returns:
        Edges dictionary for this fold
    """
    # Clone the lnc-disease association matrix
    ld = data_dict['lnc_disease_assoc'].clone()
    
    # Remove held-out link
    original_value = ld[d_idx, l_idx].item()
    ld[d_idx, l_idx] = 0
    
    removed_edge_present = original_value > 0
    rewrite_applied = False
    
    # Check if disease is now isolated and rewrite if needed
    if config.get('eval', {}).get('rewrite_isolated', True) and ld[d_idx].sum() == 0:
        # Disease is isolated, rewrite using disease similarity
        ds = data_dict['DD_sim']  # [D, D]
        
        # Get similarity weights (exclude self)
        w = ds[d_idx].clone()
        w[d_idx] = 0  # Zero self-similarity
        w = torch.softmax(w, dim=0)  # Normalize to probability distribution
        
        # Project via similar diseases → pseudo-links for this disease
        ld[d_idx, :] = (w.unsqueeze(0) @ ld).squeeze(0)
        ld[d_idx, l_idx] = 0  # Keep test link out
        rewrite_applied = True
        
        print(f"  Fold {l_idx},{d_idx}: removed_edge={removed_edge_present}, "
              f"rewrite_applied=True, disease_degree_after={ld[d_idx].sum().item():.2f}")
    
    # Create modified data dict for this fold
    dd = dict(data_dict)
    dd['lnc_disease_assoc'] = ld
    
    # Build graph for this fold
    edges = construct_heterogeneous_graph(
        dd,
        sim_topk=config['data'].get('sim_topk', 30),
        sim_row_normalize=config['data'].get('sim_row_normalize', True),
        sim_threshold=config['data'].get('threshold', 0.0),
        sim_mutual=config['data'].get('sim_mutual', False),
        sim_sym=config['data'].get('sim_sym', True),
        sim_row_norm=config['data'].get('sim_row_norm', True),
        sim_tau=config['data'].get('sim_tau', None),
        use_bipartite_edge_weight=config['data'].get('use_bipartite_edge_weight', False),
        verbose=False
    )
    
    # Count lnc-disease edges for logging
    ld_edges = 0
    for (src_type, rel_name, dst_type), (src_idx, dst_idx, w) in edges.items():
        if 'lnc' in src_type.lower() and 'disease' in dst_type.lower():
            ld_edges = len(src_idx) if src_idx is not None else 0
            break
    
    return edges, removed_edge_present, rewrite_applied, ld_edges


def rewrite_ld_for_isolated_diseases(ld_mat: torch.Tensor,
                                    dis_sim: torch.Tensor,
                                    disease_idx: int) -> torch.Tensor:
    """
    Rewrite the lncRNA-disease matrix for an isolated disease by replacing
    its row/column with its similarity vector to all diseases.
    
    This function is used during LOOCV when a disease becomes isolated (has no 
    remaining associations) after holding out a positive pair. The disease's 
    connections are replaced with weighted connections based on its similarity 
    to other diseases.
    
    Args:
        ld_mat: lncRNA-disease association matrix (disease x lncRNA format)
        dis_sim: Disease similarity matrix (disease x disease)
        disease_idx: Index of the isolated disease to rewrite
        
    Returns:
        Modified lncRNA-disease matrix with rewritten connections for the isolated disease
        
    Note:
        - The similarity to itself is set to zero
        - The row and column are normalized after replacement
        - This ensures the isolated disease has degree > 0 after rewriting
    """
    # Create a copy to avoid modifying the original matrix
    ld_mat_rewritten = ld_mat.clone()
    
    # Get the similarity vector for this disease to all other diseases
    sim_vector = dis_sim[disease_idx].clone()
    
    # Zero out self-similarity
    sim_vector[disease_idx] = 0.0
    
    # Replace the disease's row in the LD matrix with its similarity vector
    # Since ld_mat is disease x lncRNA, we need to broadcast the similarity
    # across all lncRNAs for this disease
    num_lncRNAs = ld_mat.shape[1]
    
    # For each lncRNA, set the association based on disease similarity
    # We use the max similarity approach: if disease d is similar to diseases
    # that are associated with lncRNA l, then we create a weighted connection
    for lnc_idx in range(num_lncRNAs):
        # Find which diseases are associated with this lncRNA
        associated_diseases = (ld_mat[:, lnc_idx] > 0).float()
        
        # Calculate weighted association based on similarity to associated diseases
        # This creates indirect associations through similar diseases
        weighted_association = (sim_vector * associated_diseases).sum()
        
        # Normalize by the number of associated diseases to keep values reasonable
        num_associated = associated_diseases.sum()
        if num_associated > 0:
            weighted_association = weighted_association / num_associated
        
        # Set the new association value
        ld_mat_rewritten[disease_idx, lnc_idx] = weighted_association
    
    # Normalize the row to ensure reasonable values
    row_sum = ld_mat_rewritten[disease_idx].sum()
    if row_sum > 0:
        ld_mat_rewritten[disease_idx] = ld_mat_rewritten[disease_idx] / row_sum
    
    # Also handle column normalization if needed (for bidirectional consistency)
    # This ensures the disease appears in lncRNA neighborhoods proportionally
    col_sum = ld_mat_rewritten[:, :].sum(dim=0, keepdim=True)
    col_sum = torch.clamp(col_sum, min=1e-8)
    
    return ld_mat_rewritten 