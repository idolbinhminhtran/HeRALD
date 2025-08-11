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