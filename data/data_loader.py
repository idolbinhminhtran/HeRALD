"""
Data loading utilities for HGAT-LDA model.
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any


def load_square_sim(path: str, sep: str = r'\s+') -> torch.Tensor:
    """
    Read a square similarity-matrix text file that may contain row/column labels.
    
    Args:
        path: Path to the similarity matrix file
        sep: Separator for the CSV file
        
    Returns:
        torch.float32 tensor of the similarity matrix
    """
    df = pd.read_csv(path, sep=sep, header=None, dtype=str)
    
    # If first cell is not numeric, assume header row/col exist -> drop them
    try:
        float(df.iat[0, 0])
    except ValueError:
        df = df.iloc[1:, 1:]  # drop first row & first column

    arr = df.astype(np.float32).to_numpy(copy=False)
    return torch.from_numpy(arr)


def load_dense_mat(path: str, src_cnt: int, dst_cnt: int, sep: str = r'\s+') -> torch.Tensor:
    """
    Read a dense 0/1 adjacency matrix that may have node labels.
    
    Args:
        path: Path to the adjacency matrix file
        src_cnt: Number of source nodes
        dst_cnt: Number of destination nodes
        sep: Separator for the CSV file
        
    Returns:
        torch.float32 tensor with shape [src_cnt, dst_cnt]
    """
    path = Path(path)
    df = pd.read_csv(path, sep=sep, header=None, dtype=str)

    try:  # numeric test on (0,0)
        float(df.iat[0, 0])
    except ValueError:
        df = df.iloc[1:, 1:]  # drop header row / col

    mat = df.astype(np.float32).to_numpy(copy=False)
    assert mat.shape == (src_cnt, dst_cnt), \
        f"Shape mismatch for {path.name}: got {mat.shape}, expected {(src_cnt, dst_cnt)}"
    return torch.from_numpy(mat)


def load_dataset(data_dir: str = "Dataset") -> Dict[str, torch.Tensor]:
    """
    Load all dataset files and return as a dictionary.
    
    Args:
        data_dir: Directory containing the dataset files
        
    Returns:
        Dictionary containing all loaded matrices
    """
    # Dataset dimensions
    num_lncRNAs = 472
    num_genes = 3043
    num_diseases = 130
    
    # Load similarity matrices
    LL_sim = load_square_sim(os.path.join(data_dir, 'LncFunGauSim.txt'))
    GG_sim = load_square_sim(os.path.join(data_dir, 'GeneLlsGauSim.txt'))
    DD_sim = load_square_sim(os.path.join(data_dir, 'DisSemGauSim.txt'))
    
    # Load association matrices
    lnc_gene_assoc = load_dense_mat(os.path.join(data_dir, 'GeneLncMat.txt'), 
                                   num_genes, num_lncRNAs)
    gene_disease_assoc = load_dense_mat(os.path.join(data_dir, 'GeneDisMat.txt'), 
                                       num_genes, num_diseases)
    lnc_disease_assoc = load_dense_mat(os.path.join(data_dir, 'DisLncMat.txt'), 
                                      num_diseases, num_lncRNAs)
    
    # Shape consistency assertions
    assert LL_sim.shape == (num_lncRNAs, num_lncRNAs), \
        f"LncFunGauSim shape mismatch: got {LL_sim.shape}, expected ({num_lncRNAs}, {num_lncRNAs})"
    assert GG_sim.shape == (num_genes, num_genes), \
        f"GeneLlsGauSim shape mismatch: got {GG_sim.shape}, expected ({num_genes}, {num_genes})"
    assert DD_sim.shape == (num_diseases, num_diseases), \
        f"DisSemGauSim shape mismatch: got {DD_sim.shape}, expected ({num_diseases}, {num_diseases})"
    
    # Association matrix shapes already checked in load_dense_mat, but verify here too
    assert lnc_gene_assoc.shape == (num_genes, num_lncRNAs), \
        f"GeneLncMat shape mismatch: got {lnc_gene_assoc.shape}, expected ({num_genes}, {num_lncRNAs})"
    assert gene_disease_assoc.shape == (num_genes, num_diseases), \
        f"GeneDisMat shape mismatch: got {gene_disease_assoc.shape}, expected ({num_genes}, {num_diseases})"
    assert lnc_disease_assoc.shape == (num_diseases, num_lncRNAs), \
        f"DisLncMat shape mismatch: got {lnc_disease_assoc.shape}, expected ({num_diseases}, {num_lncRNAs})"
    
    # Verify similarity matrices are symmetric
    assert torch.allclose(LL_sim, LL_sim.t(), atol=1e-6), \
        "LncFunGauSim is not symmetric"
    assert torch.allclose(GG_sim, GG_sim.t(), atol=1e-6), \
        "GeneLlsGauSim is not symmetric"
    assert torch.allclose(DD_sim, DD_sim.t(), atol=1e-6), \
        "DisSemGauSim is not symmetric"
    
    # Verify similarity matrices have unit diagonal (or close to it)
    ll_diag = torch.diagonal(LL_sim)
    gg_diag = torch.diagonal(GG_sim)
    dd_diag = torch.diagonal(DD_sim)
    
    assert torch.allclose(ll_diag, torch.ones_like(ll_diag), atol=0.01), \
        f"LncFunGauSim diagonal not close to 1: min={ll_diag.min():.4f}, max={ll_diag.max():.4f}"
    assert torch.allclose(gg_diag, torch.ones_like(gg_diag), atol=0.01), \
        f"GeneLlsGauSim diagonal not close to 1: min={gg_diag.min():.4f}, max={gg_diag.max():.4f}"
    assert torch.allclose(dd_diag, torch.ones_like(dd_diag), atol=0.01), \
        f"DisSemGauSim diagonal not close to 1: min={dd_diag.min():.4f}, max={dd_diag.max():.4f}"
    
    return {
        'LL_sim': LL_sim,
        'GG_sim': GG_sim,
        'DD_sim': DD_sim,
        'lnc_gene_assoc': lnc_gene_assoc,
        'gene_disease_assoc': gene_disease_assoc,
        'lnc_disease_assoc': lnc_disease_assoc,
        'num_lncRNAs': num_lncRNAs,
        'num_genes': num_genes,
        'num_diseases': num_diseases
    }


def get_dataset_info() -> Dict[str, int]:
    """
    Get basic information about the dataset dimensions.
    
    Returns:
        Dictionary with dataset dimensions
    """
    return {
        'num_lncRNAs': 472,
        'num_genes': 3043,
        'num_diseases': 130
    }


def create_id_mappings(num_lncRNAs: int = 472, num_genes: int = 3043, 
                      num_diseases: int = 130) -> Dict[str, Dict]:
    """
    Create ID mappings for round-trip testing.
    
    Args:
        num_lncRNAs: Number of lncRNAs
        num_genes: Number of genes  
        num_diseases: Number of diseases
        
    Returns:
        Dictionary containing ID to index and index to ID mappings
    """
    # Create synthetic ID strings for each entity type
    lnc_id_to_idx = {f"LNC_{i:04d}": i for i in range(num_lncRNAs)}
    gene_id_to_idx = {f"GENE_{i:04d}": i for i in range(num_genes)}
    disease_id_to_idx = {f"DIS_{i:03d}": i for i in range(num_diseases)}
    
    # Create reverse mappings
    lnc_idx_to_id = {v: k for k, v in lnc_id_to_idx.items()}
    gene_idx_to_id = {v: k for k, v in gene_id_to_idx.items()}
    disease_idx_to_id = {v: k for k, v in disease_id_to_idx.items()}
    
    return {
        'lnc_id_to_idx': lnc_id_to_idx,
        'lnc_idx_to_id': lnc_idx_to_id,
        'gene_id_to_idx': gene_id_to_idx,
        'gene_idx_to_id': gene_idx_to_id,
        'disease_id_to_idx': disease_id_to_idx,
        'disease_idx_to_id': disease_idx_to_id
    }


def verify_id_mapping_roundtrip(mappings: Dict[str, Dict]) -> bool:
    """
    Verify that ID mappings work correctly in both directions.
    
    Args:
        mappings: Dictionary from create_id_mappings
        
    Returns:
        True if all round-trip tests pass
    """
    # Test lncRNA mappings
    for lnc_id, idx in mappings['lnc_id_to_idx'].items():
        assert mappings['lnc_idx_to_id'][idx] == lnc_id, \
            f"lncRNA mapping failed: {lnc_id} -> {idx} -> {mappings['lnc_idx_to_id'].get(idx)}"
    
    # Test first and last lncRNA
    first_lnc = "LNC_0000"
    last_lnc = f"LNC_{len(mappings['lnc_id_to_idx'])-1:04d}"
    assert mappings['lnc_id_to_idx'][first_lnc] == 0, f"First lncRNA mapping failed"
    assert mappings['lnc_id_to_idx'][last_lnc] == len(mappings['lnc_id_to_idx'])-1, \
        f"Last lncRNA mapping failed"
    
    # Test gene mappings  
    for gene_id, idx in mappings['gene_id_to_idx'].items():
        assert mappings['gene_idx_to_id'][idx] == gene_id, \
            f"Gene mapping failed: {gene_id} -> {idx} -> {mappings['gene_idx_to_id'].get(idx)}"
    
    # Test first and last gene
    first_gene = "GENE_0000"
    last_gene = f"GENE_{len(mappings['gene_id_to_idx'])-1:04d}"
    assert mappings['gene_id_to_idx'][first_gene] == 0, f"First gene mapping failed"
    assert mappings['gene_id_to_idx'][last_gene] == len(mappings['gene_id_to_idx'])-1, \
        f"Last gene mapping failed"
        
    # Test disease mappings
    for disease_id, idx in mappings['disease_id_to_idx'].items():
        assert mappings['disease_idx_to_id'][idx] == disease_id, \
            f"Disease mapping failed: {disease_id} -> {idx} -> {mappings['disease_idx_to_id'].get(idx)}"
    
    # Test first and last disease
    first_disease = "DIS_000"
    last_disease = f"DIS_{len(mappings['disease_id_to_idx'])-1:03d}"
    assert mappings['disease_id_to_idx'][first_disease] == 0, f"First disease mapping failed"
    assert mappings['disease_id_to_idx'][last_disease] == len(mappings['disease_id_to_idx'])-1, \
        f"Last disease mapping failed"
    
    return True 