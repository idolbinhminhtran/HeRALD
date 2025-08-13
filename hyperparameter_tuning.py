#!/usr/bin/env python3
"""
Refactored hyperparameter tuning script for HGAT-LDA model with true LOOCV protocol.
- Per-fold graph rebuilding with disease rewrite for isolated nodes
- Full-ranking evaluation without negative sampling
- Stratified fold selection for better coverage
- GPU-safe parallel execution with Optuna
- Improved logging and vectorized operations
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import Optuna if available
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Only grid search available.")
    print("Install with: pip install optuna")

from sklearn.metrics import roc_auc_score
from data.data_loader import load_dataset, get_dataset_info
from data.graph_construction import (
    construct_heterogeneous_graph, 
    get_positive_pairs, 
    generate_negative_pairs,
    rewrite_ld_for_isolated_diseases
)
from models.hgat_lda import HGAT_LDA
from training.trainer import HGATLDATrainer
from training.evaluator import HGATLDAEvaluator


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_for_trial(trial_number: int) -> torch.device:
    """
    Assign device round-robin across available GPUs.
    
    Args:
        trial_number: Current trial number
        
    Returns:
        Device for this trial
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = trial_number % num_gpus
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def validate_model_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fix model parameters to ensure compatibility.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Fixed configuration
    """
    # Ensure emb_dim is divisible by num_heads
    emb_dim = int(config['model']['emb_dim'])
    num_heads = int(config['model'].get('num_heads', 4))
    
    if emb_dim % num_heads != 0:
        # Round up to nearest multiple of num_heads
        new_emb_dim = ((emb_dim + num_heads - 1) // num_heads) * num_heads
        print(f"  ⚠️  Adjusted emb_dim from {emb_dim} to {new_emb_dim} (divisible by {num_heads} heads)")
        config['model']['emb_dim'] = new_emb_dim
    
    # Resolve loss_type vs use_hard_negatives conflicts
    loss_type = config['training'].get('loss_type', 'bce')
    use_hard_negatives = config['training'].get('use_hard_negatives', False)
    
    if loss_type != 'pairwise' and use_hard_negatives:
        print(f"  ⚠️  use_hard_negatives=True only works with loss_type='pairwise', setting to False")
        config['training']['use_hard_negatives'] = False
    
    # Ensure label_smoothing is reasonable
    label_smoothing = float(config['training'].get('label_smoothing', 0.0))
    if label_smoothing > 0.2:
        print(f"  ⚠️  label_smoothing={label_smoothing} too high for LOOCV, capping at 0.1")
        config['training']['label_smoothing'] = 0.1
    
    return config


def build_fold_edges(data_dict: Dict[str, torch.Tensor],
                     config: Dict[str, Any],
                     test_lnc: int,
                     test_dis: int,
                     verbose: bool = False) -> Dict:
    """
    Build graph edges for a specific LOOCV fold, removing the test edge
    and handling isolated diseases.
    
    Args:
        data_dict: Dataset dictionary
        config: Configuration
        test_lnc: Test lncRNA index
        test_dis: Test disease index
        verbose: Whether to print details
        
    Returns:
        Edges dictionary for this fold
    """
    # Create a copy of the lnc-disease association matrix
    ld_mat = data_dict['lnc_disease_assoc'].clone()
    
    # Remove the test edge
    original_value = ld_mat[test_dis, test_lnc].item()
    ld_mat[test_dis, test_lnc] = 0
    
    if verbose:
        print(f"  Removed edge: lnc={test_lnc}, dis={test_dis} (was {original_value})")
    
    # Check if disease is now isolated
    disease_degree = ld_mat[test_dis, :].sum().item()
    rewrite_applied = False
    
    if disease_degree == 0:
        # Disease is isolated, apply rewrite using disease similarity
        ld_mat = rewrite_ld_for_isolated_diseases(
            ld_mat, 
            data_dict['DD_sim'], 
            test_dis
        )
        rewrite_applied = True
        new_degree = ld_mat[test_dis, :].sum().item()
        if verbose:
            print(f"  Disease {test_dis} isolated, applied rewrite (new degree: {new_degree:.2f})")
    
    # Create modified data dict for this fold
    fold_data = data_dict.copy()
    fold_data['lnc_disease_assoc'] = ld_mat
    
    # Construct heterogeneous graph for this fold
    edges = construct_heterogeneous_graph(
        fold_data,
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
    
    return edges, rewrite_applied


def score_all_lnc(model: torch.nn.Module,
                  disease_idx: int,
                  edges: Dict,
                  num_lncRNAs: int,
                  device: torch.device,
                  batch_size: int = 100) -> torch.Tensor:
    """
    Score all lncRNAs for a given disease (vectorized for speed).
    
    Args:
        model: Trained model
        disease_idx: Disease index
        edges: Graph edges
        num_lncRNAs: Total number of lncRNAs
        device: Device to use
        batch_size: Batch size for scoring
        
    Returns:
        Scores for all lncRNAs
    """
    model.eval()
    all_scores = []
    
    with torch.no_grad():
        # Score in batches to avoid OOM
        for start in range(0, num_lncRNAs, batch_size):
            end = min(start + batch_size, num_lncRNAs)
            batch_size_actual = end - start
            
            # Create batch
            lnc_batch = torch.arange(start, end, device=device)
            dis_batch = torch.full((batch_size_actual,), disease_idx, 
                                  dtype=torch.long, device=device)
            
            # Score batch
            scores = model(lnc_batch, dis_batch, edges)
            all_scores.append(scores)
    
    return torch.cat(all_scores)


def select_loocv_folds(pos_pairs: torch.Tensor,
                      data_dict: Dict[str, torch.Tensor],
                      target_folds: int = 150,
                      seed: int = 42) -> np.ndarray:
    """
    Select LOOCV folds with stratification to ensure coverage across
    diseases and degree buckets.
    
    Args:
        pos_pairs: All positive pairs
        data_dict: Dataset dictionary
        target_folds: Target number of folds
        seed: Random seed
        
    Returns:
        Indices of selected folds
    """
    np.random.seed(seed)
    
    # Get disease degrees
    ld_mat = data_dict['lnc_disease_assoc']
    disease_degrees = ld_mat.sum(dim=1).numpy()  # Sum over lncRNAs
    
    # Create disease buckets based on degree
    num_diseases = len(disease_degrees)
    disease_buckets = {
        'low': [],     # degree <= 5
        'medium': [],  # 5 < degree <= 10
        'high': []     # degree > 10
    }
    
    for dis_idx in range(num_diseases):
        degree = disease_degrees[dis_idx]
        if degree <= 5:
            disease_buckets['low'].append(dis_idx)
        elif degree <= 10:
            disease_buckets['medium'].append(dis_idx)
        else:
            disease_buckets['high'].append(dis_idx)
    
    # Group positive pairs by disease
    pairs_by_disease = {}
    for i, (lnc, dis) in enumerate(pos_pairs):
        dis_idx = dis.item()
        if dis_idx not in pairs_by_disease:
            pairs_by_disease[dis_idx] = []
        pairs_by_disease[dis_idx].append(i)
    
    selected_folds = []
    
    # First pass: ensure coverage across buckets and diseases
    for bucket_name, disease_list in disease_buckets.items():
        if not disease_list:
            continue
            
        # Sample diseases from this bucket
        num_to_sample = min(len(disease_list), target_folds // (3 * 2))  # Divide by 3 buckets, 2 samples per disease
        sampled_diseases = np.random.choice(disease_list, num_to_sample, replace=False)
        
        for dis_idx in sampled_diseases:
            if dis_idx in pairs_by_disease:
                # Sample up to 2 pairs from this disease
                available_pairs = pairs_by_disease[dis_idx]
                num_pairs = min(2, len(available_pairs))
                sampled_pairs = np.random.choice(available_pairs, num_pairs, replace=False)
                selected_folds.extend(sampled_pairs)
    
    # Second pass: fill remaining slots randomly
    remaining_needed = target_folds - len(selected_folds)
    if remaining_needed > 0:
        all_indices = set(range(len(pos_pairs)))
        available_indices = list(all_indices - set(selected_folds))
        
        if available_indices:
            additional_folds = np.random.choice(
                available_indices, 
                min(remaining_needed, len(available_indices)), 
                replace=False
            )
            selected_folds.extend(additional_folds)
    
    # Convert to numpy array and sort
    selected_folds = np.array(selected_folds[:target_folds])
    selected_folds = np.sort(selected_folds)
    
    # Calculate coverage statistics
    unique_diseases = set()
    for fold_idx in selected_folds:
        dis_idx = pos_pairs[fold_idx, 1].item()
        unique_diseases.add(dis_idx)
    
    coverage = len(unique_diseases) / num_diseases * 100
    print(f"  Fold selection: {len(selected_folds)} folds covering {len(unique_diseases)}/{num_diseases} diseases ({coverage:.1f}%)")
    
    return selected_folds


def quick_evaluate_loocv(config: Dict[str, Any],
                        data_dict: Dict[str, torch.Tensor],
                        dataset_info: Dict,
                        pos_pairs: torch.Tensor,
                        device: torch.device,
                        num_loocv_samples: int = 150,
                        trial: Optional['optuna.Trial'] = None,
                        trial_number: int = 0) -> float:
    """
    Quick LOOCV evaluation with proper protocol:
    - Per-fold graph rebuilding
    - Full ranking without negative sampling
    - Stratified fold selection
    
    Returns:
        Mean AUC score across sampled LOOCV folds
    """
    # Set per-trial seed for reproducibility
    trial_seed = 42 + trial_number
    set_seed(trial_seed)
    
    # Validate and fix model parameters
    config = validate_model_params(config)
    
    # Select stratified folds
    selected_folds = select_loocv_folds(
        pos_pairs, 
        data_dict, 
        target_folds=num_loocv_samples,
        seed=trial_seed
    )
    
    print(f"\n🔬 Trial {trial_number}: Evaluating {len(selected_folds)} LOOCV folds")
    print(f"  Device: {device}")
    print(f"  eval.full_ranking=True, neg_sampling=False")
    
    auc_scores = []
    num_rewrites = 0
    
    # Process each fold
    for fold_count, fold_idx in enumerate(selected_folds):
        # Get test pair
        test_lnc = int(pos_pairs[fold_idx, 0].item())
        test_dis = int(pos_pairs[fold_idx, 1].item())
        
        # Build fold-specific edges (removes test edge, handles isolation)
        fold_edges, rewrite_applied = build_fold_edges(
            data_dict, config, test_lnc, test_dis, verbose=False
        )
        
        if rewrite_applied:
            num_rewrites += 1
        
        # Move edges to device
        edges_device = {}
        for key, (src, dst, w) in fold_edges.items():
            if src is not None:
                src = src.to(device)
                dst = dst.to(device)
                if w is not None:
                    w = w.to(device)
            edges_device[key] = (src, dst, w)
        
        # Create training data (exclude test pair)
        train_mask = torch.ones(len(pos_pairs), dtype=torch.bool)
        train_mask[fold_idx] = False
        train_pos_pairs = pos_pairs[train_mask]
        
        try:
            # Create fresh model for this fold
            model_fold = HGAT_LDA(
                num_lncRNAs=dataset_info['num_lncRNAs'],
                num_genes=dataset_info['num_genes'],
                num_diseases=dataset_info['num_diseases'],
                edges=edges_device,
                emb_dim=int(config['model']['emb_dim']),
                num_layers=int(config['model']['num_layers']),
                dropout=float(config['model']['dropout']),
                num_heads=int(config['model'].get('num_heads', 4)),
                relation_dropout=float(config['model'].get('relation_dropout', 0.0)),
                use_layernorm=bool(config['model'].get('use_layernorm', True)),
                use_residual=bool(config['model'].get('use_residual', True)),
                use_relation_norm=bool(config['model'].get('use_relation_norm', True))
            ).to(device)
            
            # Create trainer
            trainer_fold = HGATLDATrainer(
                model=model_fold,
                device=device,
                lr=float(config['training']['lr']),
                weight_decay=float(config['training']['weight_decay']),
                batch_size=int(config['training']['batch_size']),
                enable_progress=False,
                neg_ratio=int(config['training']['neg_ratio']),
                use_amp=config.get('system', {}).get('use_amp', torch.cuda.is_available()),
                use_focal_loss=bool(config['training'].get('use_focal_loss', False)),
                label_smoothing=float(config['training'].get('label_smoothing', 0.0)),
                cosine_tmax=config['training'].get('cosine_tmax', 50),
                use_multi_gpu=False,
                loss_type=config['training'].get('loss_type', 'bce'),
                pairwise_type=config['training'].get('pairwise_type', 'bpr'),
                use_hard_negatives=config['training'].get('use_hard_negatives', False)
            )
            
            # Generate negatives for training
            neg_pairs_all = generate_negative_pairs(
                train_pos_pairs, 
                dataset_info['num_lncRNAs'], 
                dataset_info['num_diseases']
            )
            
            # Train model
            num_epochs = min(30, int(config['training'].get('num_epochs', 30)))
            trainer_fold.train(
                pos_pairs=train_pos_pairs,
                neg_pairs_all=neg_pairs_all,
                edges=edges_device,
                num_epochs=num_epochs,
                val_split=0.0,
                early_stopping_patience=5,
                save_path=None
            )
            
            # Full-ranking evaluation (no negative sampling)
            all_scores = score_all_lnc(
                model_fold, test_dis, edges_device, 
                dataset_info['num_lncRNAs'], device
            )
            
            # Get other known positives for this disease to mask
            known_positives = set()
            for i, (lnc, dis) in enumerate(pos_pairs):
                if dis.item() == test_dis and i != fold_idx:
                    known_positives.add(lnc.item())
            
            # Create labels and scores for AUC calculation
            y_true = []
            y_score = []
            
            for lnc_idx in range(dataset_info['num_lncRNAs']):
                if lnc_idx == test_lnc:
                    # This is our test positive
                    y_true.append(1)
                    y_score.append(torch.sigmoid(all_scores[lnc_idx]).item())
                elif lnc_idx not in known_positives:
                    # This is a true negative
                    y_true.append(0)
                    y_score.append(torch.sigmoid(all_scores[lnc_idx]).item())
                # Skip other known positives (masked)
            
            # Calculate AUC
            if len(set(y_true)) == 2:  # Ensure we have both classes
                fold_auc = roc_auc_score(y_true, y_score)
                auc_scores.append(fold_auc)
            else:
                auc_scores.append(0.5)  # Default if only one class
            
            # Clean up
            del model_fold
            del trainer_fold
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  ⚠️  Fold {fold_idx} failed: {e}")
            auc_scores.append(0.5)
        
        # Report to Optuna for pruning
        if trial is not None and (fold_count + 1) % 20 == 0:
            current_mean_auc = np.mean(auc_scores)
            trial.report(current_mean_auc, fold_count)
            if trial.should_prune():
                print(f"  ✂️  Trial pruned at fold {fold_count + 1}")
                raise optuna.TrialPruned()
    
    # Calculate final metrics
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    success_rate = len([a for a in auc_scores if a > 0.5]) / len(auc_scores)
    
    print(f"  LOOCV quick-eval: mean_auc={mean_auc:.4f}, std={std_auc:.4f}, success_rate={success_rate:.2f}")
    print(f"  LOOCV folds: {len(selected_folds)}, rewrite applied: {num_rewrites}")
    
    # Return score with penalty for high variance
    return mean_auc - 0.01 * std_auc


def optuna_objective(trial: 'optuna.Trial',
                     data_dict: Dict[str, torch.Tensor],
                     dataset_info: Dict,
                     pos_pairs: torch.Tensor,
                     base_config: Dict[str, Any],
                     search_space: Dict[str, Any]) -> float:
    """
    Optuna objective function with GPU-safe parallel execution.
    """
    # Get trial number for device assignment
    trial_number = trial.number
    device = get_device_for_trial(trial_number)
    
    # Copy base config
    config = yaml.safe_load(yaml.dump(base_config))
    
    # Sample hyperparameters
    params = {}
    for param_path, param_spec in search_space.items():
        if param_spec['type'] == 'categorical':
            value = trial.suggest_categorical(param_path, param_spec['values'])
        elif param_spec['type'] == 'int':
            value = trial.suggest_int(param_path, param_spec['min'], param_spec['max'], 
                                     step=param_spec.get('step', 1))
        elif param_spec['type'] == 'float':
            if param_spec.get('log', False):
                value = trial.suggest_float(param_path, param_spec['min'], param_spec['max'], log=True)
            else:
                value = trial.suggest_float(param_path, param_spec['min'], param_spec['max'],
                                          step=param_spec.get('step'))
        else:
            continue
        
        # Update config
        keys = param_path.split('.')
        temp = config
        for key in keys[:-1]:
            if key not in temp:
                temp[key] = {}
            temp = temp[key]
        temp[keys[-1]] = value
        params[param_path] = value
    
    # Log parameters
    print(f"\n📊 Trial {trial_number} parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Evaluate
    try:
        score = quick_evaluate_loocv(
            config, data_dict, dataset_info, pos_pairs, 
            device, num_loocv_samples=150, trial=trial, trial_number=trial_number
        )
        return score
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"  ❌ Trial {trial_number} failed: {e}")
        return 0.0


def run_bayesian_optimization(base_config: Dict[str, Any],
                             data_dict: Dict[str, torch.Tensor],
                             dataset_info: Dict,
                             pos_pairs: torch.Tensor,
                             search_space: Dict[str, Any],
                             n_trials: int = 50,
                             output_dir: str = 'tuning_results') -> Tuple[Dict, float]:
    """
    Run Bayesian optimization using Optuna with parallel GPU support.
    """
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna not installed!")
        return None, 0.0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: optuna_objective(trial, data_dict, dataset_info, pos_pairs, base_config, search_space),
        n_trials=n_trials,
        n_jobs=1  # Set to 1 for GPU safety, trials will use different GPUs via round-robin
    )
    
    # Get best parameters
    best_trial = study.best_trial
    best_params = best_trial.params
    best_auc = best_trial.value
    
    # Save results
    results = {
        'best_params': best_params,
        'best_auc': best_auc,
        'n_trials': len(study.trials),
        'datetime': datetime.now().isoformat()
    }
    
    results_file = os.path.join(output_dir, 'optuna_best_params.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Best LOOCV AUC: {best_auc:.4f}")
    print(f"📊 Best parameters saved to: {results_file}")
    
    return best_params, best_auc


def main():
    parser = argparse.ArgumentParser(description='Refactored hyperparameter tuning for HGAT-LDA')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Base configuration file')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of trials for Bayesian optimization')
    parser.add_argument('--n_folds', type=int, default=150,
                       help='Number of LOOCV folds to sample')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='tuning_results',
                       help='Directory to save results')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load base configuration
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Load data
    print("Loading dataset...")
    data_dict = load_dataset(base_config['data']['data_dir'])
    dataset_info = get_dataset_info()
    
    # Get positive pairs
    pos_pairs = get_positive_pairs(data_dict['lnc_disease_assoc'])
    print(f"Total positive pairs: {len(pos_pairs)}")
    
    # Define search space
    search_space = {
        # Model parameters
        'model.emb_dim': {'type': 'categorical', 'values': [128, 192, 256]},
        'model.num_layers': {'type': 'int', 'min': 2, 'max': 3},
        'model.dropout': {'type': 'float', 'min': 0.3, 'max': 0.45, 'step': 0.05},
        'model.num_heads': {'type': 'categorical', 'values': [4, 6, 8]},
        'model.relation_dropout': {'type': 'float', 'min': 0.1, 'max': 0.25, 'step': 0.05},
        
        # Training parameters
        'training.lr': {'type': 'float', 'min': 1e-4, 'max': 5e-4, 'log': True},
        'training.weight_decay': {'type': 'float', 'min': 5e-5, 'max': 1e-4, 'log': True},
        'training.batch_size': {'type': 'categorical', 'values': [64, 128]},
        'training.neg_ratio': {'type': 'int', 'min': 2, 'max': 3},
        'training.loss_type': {'type': 'categorical', 'values': ['pairwise']},
        'training.pairwise_type': {'type': 'categorical', 'values': ['auc', 'bpr']},
        
        # Data parameters
        'data.sim_topk': {'type': 'int', 'min': 20, 'max': 40, 'step': 5},
    }
    
    # Run optimization
    print(f"\n🚀 Starting Bayesian optimization for true LOOCV protocol")
    print(f"🎯 Objective: Maximize LOOCV AUC with full ranking")
    print(f"📦 Sampling {args.n_folds} stratified folds per trial")
    print("=" * 60)
    
    best_params, best_auc = run_bayesian_optimization(
        base_config, data_dict, dataset_info, pos_pairs, 
        search_space, args.n_trials, args.output_dir
    )
    
    # Update config with best parameters
    if best_params:
        print("\n🔄 Updating configs/default.yaml with best parameters...")
        config = base_config.copy()
        
        for param_path, value in best_params.items():
            keys = param_path.split('.')
            temp = config
            for key in keys[:-1]:
                if key not in temp:
                    temp[key] = {}
                temp = temp[key]
            temp[keys[-1]] = value
        
        with open(args.config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print("✅ Configuration updated successfully!")
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ REFACTORED LOOCV OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best LOOCV AUC: {best_auc:.4f}")
    print(f"Results saved in: {args.output_dir}")
    print("\nYou can now run full LOOCV evaluation:")
    print("  python3 main.py --mode loocv --config configs/default.yaml")


if __name__ == "__main__":
    main()
