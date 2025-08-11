#!/usr/bin/env python3
"""
Hyperparameter tuning script for HGAT-LDA model.
Supports both Bayesian optimization (Optuna) and Grid Search.
Automatically updates configs/default.yaml with best parameters.
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
from typing import Dict, Any, List, Tuple
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

from data.data_loader import load_dataset, get_dataset_info
from data.graph_construction import construct_heterogeneous_graph, get_positive_pairs, generate_negative_pairs
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


def load_data_and_graph(config: Dict[str, Any]):
    """Load dataset and construct graph."""
    print("Loading dataset...")
    data_dict = load_dataset(config['data']['data_dir'])
    dataset_info = get_dataset_info()
    
    print("Constructing heterogeneous graph...")
    edges = construct_heterogeneous_graph(
        data_dict,
        sim_topk=config['data'].get('sim_topk', None),
        sim_row_normalize=config['data'].get('sim_row_normalize', True),
        sim_threshold=config['data'].get('threshold', 0.0)
    )
    
    pos_pairs = get_positive_pairs(data_dict['lnc_disease_assoc'])
    print(f"Number of positive pairs: {len(pos_pairs)}")
    
    return data_dict, dataset_info, edges, pos_pairs


def quick_evaluate(config: Dict[str, Any],
                  dataset_info: Dict,
                  edges: Dict,
                  pos_pairs: torch.Tensor,
                  device: torch.device,
                  num_val_folds: int = 3) -> float:
    """
    Quick evaluation using a subset of LOOCV folds.
    
    Returns:
        Mean AUC score across validation folds
    """
    # Use a fixed validation set for faster evaluation
    n_val = int(len(pos_pairs) * 0.2)
    val_pairs = pos_pairs[:n_val]
    train_pairs = pos_pairs[n_val:]
    
    # Create model
    model = HGAT_LDA(
        num_lncRNAs=dataset_info['num_lncRNAs'],
        num_genes=dataset_info['num_genes'],
        num_diseases=dataset_info['num_diseases'],
        edges=edges,
        emb_dim=int(config['model']['emb_dim']),
        num_layers=int(config['model']['num_layers']),
        dropout=float(config['model']['dropout']),
        num_heads=int(config['model'].get('num_heads', 4)),
        relation_dropout=float(config['model'].get('relation_dropout', 0.0)),
        use_layernorm=bool(config['model'].get('use_layernorm', True)),
        use_residual=bool(config['model'].get('use_residual', True))
    ).to(device)
    
    # Create trainer with specified parameters
    trainer = HGATLDATrainer(
        model=model,
        device=device,
        lr=float(config['training']['lr']),
        weight_decay=float(config['training']['weight_decay']),
        batch_size=int(config['training']['batch_size']),
        enable_progress=False,  # Disable progress for cleaner output
        neg_ratio=int(config['training']['neg_ratio']),
        use_amp=config.get('system', {}).get('use_amp', False),
        use_focal_loss=bool(config['training'].get('use_focal_loss', False)),
        label_smoothing=float(config['training'].get('label_smoothing', 0.0))
    )
    
    # Generate negatives
    neg_pairs_all = generate_negative_pairs(train_pairs, dataset_info['num_lncRNAs'], dataset_info['num_diseases'])
    
    # Quick training
    trainer.train(
        pos_pairs=train_pairs,
        neg_pairs_all=neg_pairs_all,
        edges=edges,
        num_epochs=20,  # Quick training for evaluation
        val_split=0.0,
        early_stopping_patience=5,
        save_path=None
    )
    
    # Evaluate
    try:
        evaluator = HGATLDAEvaluator(model, device)
        val_neg = generate_negative_pairs(val_pairs, dataset_info['num_lncRNAs'], dataset_info['num_diseases'], len(val_pairs))
        auc = evaluator.evaluate_auc(val_pairs, val_neg, edges)
        
        # Clean up
        del model
        del trainer
        del evaluator
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return auc
    except Exception as e:
        print(f"Evaluation failed: {e}")
        # Clean up on error
        del model
        del trainer
        if 'evaluator' in locals():
            del evaluator
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return 0.0


def update_default_config(best_params: Dict[str, Any]):
    """Update configs/default.yaml with best parameters."""
    config_path = 'configs/default.yaml'
    
    # Load current config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update with best parameters
    for param_path, value in best_params.items():
        # Handle nested parameters like 'model.emb_dim'
        keys = param_path.split('.')
        temp = config
        for key in keys[:-1]:
            if key not in temp:
                temp[key] = {}
            temp = temp[key]
        temp[keys[-1]] = value
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Updated {config_path} with best parameters")


# ============================================================================
# BAYESIAN OPTIMIZATION (OPTUNA)
# ============================================================================

def optuna_objective(trial: 'optuna.Trial',
                     dataset_info: Dict,
                     edges: Dict,
                     pos_pairs: torch.Tensor,
                     device: torch.device,
                     base_config: Dict[str, Any],
                     search_space: Dict[str, Any]) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    """
    # Copy base config
    config = yaml.safe_load(yaml.dump(base_config))
    
    # Sample hyperparameters based on search space
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
        
        # Update config with sampled value
        keys = param_path.split('.')
        temp = config
        for key in keys[:-1]:
            temp = temp[key]
        temp[keys[-1]] = value
        params[param_path] = value
    
    # Evaluate configuration
    try:
        auc_score = quick_evaluate(config, dataset_info, edges, pos_pairs, device)
        
        # Report intermediate value
        trial.report(auc_score, 0)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return auc_score
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        # Return a small negative value instead of 0 to help Optuna learn
        return -0.1


def run_bayesian_optimization(base_config: Dict[str, Any],
                             dataset_info: Dict,
                             edges: Dict,
                             pos_pairs: torch.Tensor,
                             device: torch.device,
                             search_space: Dict[str, Any],
                             n_trials: int = 50,
                             output_dir: str = 'tuning_results'):
    """Run Bayesian optimization using Optuna."""
    if not OPTUNA_AVAILABLE:
        print("Error: Optuna not installed. Please install with: pip install optuna")
        return None
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create study
    study = optuna.create_study(
        study_name=f"hgat_lda_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Optimize
    print(f"\n🔍 Starting Bayesian optimization with {n_trials} trials...")
    study.optimize(
        lambda trial: optuna_objective(trial, dataset_info, edges, pos_pairs, device, base_config, search_space),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get best parameters
    best_trial = study.best_trial
    print(f"\n✨ Best AUC Score: {best_trial.value:.4f}")
    print("\n📊 Best parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        'best_auc': best_trial.value,
        'best_params': best_trial.params,
        'all_trials': [
            {'number': t.number, 'value': t.value, 'params': t.params}
            for t in study.trials if t.value is not None
        ]
    }
    
    with open(output_dir / 'optuna_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_trial.params, best_trial.value


# ============================================================================
# GRID SEARCH
# ============================================================================

def run_grid_search(base_config: Dict[str, Any],
                   dataset_info: Dict,
                   edges: Dict,
                   pos_pairs: torch.Tensor,
                   device: torch.device,
                   search_space: Dict[str, Any],
                   max_combinations: int = None,
                   output_dir: str = 'tuning_results'):
    """Run grid search over parameter combinations."""
    from itertools import product
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations
    param_names = []
    param_values = []
    
    for param_path, param_spec in search_space.items():
        param_names.append(param_path)
        if param_spec['type'] == 'categorical':
            param_values.append(param_spec['values'])
        elif param_spec['type'] == 'int':
            values = list(range(param_spec['min'], param_spec['max'] + 1, param_spec.get('step', 1)))
            param_values.append(values)
        elif param_spec['type'] == 'float':
            if param_spec.get('log', False):
                # Sample log-spaced values
                values = np.logspace(np.log10(param_spec['min']), np.log10(param_spec['max']), 
                                   param_spec.get('n_samples', 3)).tolist()
            else:
                # Sample linearly-spaced values
                if 'step' in param_spec:
                    values = np.arange(param_spec['min'], param_spec['max'] + param_spec['step'], 
                                      param_spec['step']).tolist()
                else:
                    values = np.linspace(param_spec['min'], param_spec['max'], 
                                       param_spec.get('n_samples', 3)).tolist()
            param_values.append(values)
    
    # Generate combinations
    all_combinations = list(product(*param_values))
    
    # Limit combinations if specified
    if max_combinations and len(all_combinations) > max_combinations:
        np.random.shuffle(all_combinations)
        all_combinations = all_combinations[:max_combinations]
    
    print(f"\n🔍 Testing {len(all_combinations)} parameter combinations...")
    
    results = []
    best_auc = 0
    best_params = None
    
    for i, combination in enumerate(tqdm(all_combinations, desc="Grid Search")):
        # Create config for this combination
        config = yaml.safe_load(yaml.dump(base_config))
        params = dict(zip(param_names, combination))
        
        # Update config
        for param_path, value in params.items():
            keys = param_path.split('.')
            temp = config
            for key in keys[:-1]:
                temp = temp[key]
            temp[keys[-1]] = value
        
        # Evaluate
        try:
            auc = quick_evaluate(config, dataset_info, edges, pos_pairs, device)
            results.append({'params': params, 'auc': auc})
            
            if auc > best_auc:
                best_auc = auc
                best_params = params
                print(f"\n✨ New best AUC: {best_auc:.4f}")
            
        except Exception as e:
            print(f"\nError with combination {params}: {e}")
            continue
    
    # Handle case where all trials failed
    if not results:
        print("\n❌ All trials failed! No valid results.")
        return None, 0.0
    
    # Sort results
    results.sort(key=lambda x: x['auc'], reverse=True)
    
    # Save results
    with open(output_dir / 'grid_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✨ Best AUC Score: {best_auc:.4f}")
    print("\n📊 Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    return best_params, best_auc


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for HGAT-LDA')
    parser.add_argument('--method', type=str, default='bayesian', choices=['bayesian', 'grid'],
                       help='Tuning method: bayesian (Optuna) or grid search')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Base configuration file')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of trials for Bayesian optimization')
    parser.add_argument('--max_combinations', type=int, default=None,
                       help='Maximum combinations for grid search')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='tuning_results',
                       help='Directory to save results')
    parser.add_argument('--search_space', type=str, default='default',
                       choices=['default', 'quick', 'extensive'],
                       help='Search space preset')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"🖥️ Using device: {device}")
    
    # Load base configuration
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Load data
    data_dict, dataset_info, edges, pos_pairs = load_data_and_graph(base_config)
    
    # Move edges to device
    edges_device = {}
    for key, (src, dst, w) in edges.items():
        if src is not None:
            src = src.to(device)
            dst = dst.to(device)
            if w is not None:
                w = w.to(device)
        edges_device[key] = (src, dst, w)
    
    # Define search space based on preset
    if args.search_space == 'quick':
        search_space = {
            'model.emb_dim': {'type': 'categorical', 'values': [64, 128]},
            'model.num_layers': {'type': 'categorical', 'values': [1, 2]},
            'model.dropout': {'type': 'categorical', 'values': [0.2, 0.3, 0.4]},
            'training.lr': {'type': 'float', 'min': 5e-4, 'max': 5e-3, 'log': True, 'n_samples': 3},
            'training.neg_ratio': {'type': 'categorical', 'values': [3, 5]},
            'training.use_focal_loss': {'type': 'categorical', 'values': [True, False]},
        }
    elif args.search_space == 'extensive':
        search_space = {
            # Model parameters
            'model.emb_dim': {'type': 'categorical', 'values': [32, 64, 96, 128]},
            'model.num_layers': {'type': 'int', 'min': 1, 'max': 4},
            'model.dropout': {'type': 'float', 'min': 0.1, 'max': 0.5, 'step': 0.1},
            'model.num_heads': {'type': 'categorical', 'values': [1, 2, 3, 4, 6, 8]},
            'model.relation_dropout': {'type': 'float', 'min': 0.0, 'max': 0.3, 'step': 0.1},
            
            # Training parameters (only those that exist in current trainer)
            'training.lr': {'type': 'float', 'min': 1e-4, 'max': 1e-2, 'log': True},
            'training.weight_decay': {'type': 'float', 'min': 1e-6, 'max': 1e-4, 'log': True},
            'training.batch_size': {'type': 'categorical', 'values': [64, 128, 256, 512]},
            'training.neg_ratio': {'type': 'int', 'min': 1, 'max': 10},
            'training.use_focal_loss': {'type': 'categorical', 'values': [True, False]},
            'training.label_smoothing': {'type': 'float', 'min': 0.0, 'max': 0.3, 'step': 0.05},
            'training.cosine_tmax': {'type': 'categorical', 'values': [50, 100, 150, 200]},
            
            # Data parameters
            'data.sim_topk': {'type': 'int', 'min': 5, 'max': 50, 'step': 5},
        }
    else:  # default
        search_space = {
            # Model parameters
            'model.emb_dim': {'type': 'categorical', 'values': [32, 64, 96, 128]},
            'model.num_layers': {'type': 'int', 'min': 1, 'max': 3},
            'model.dropout': {'type': 'float', 'min': 0.1, 'max': 0.5, 'step': 0.1},
            'model.num_heads': {'type': 'categorical', 'values': [1, 2, 3, 4]},
            'model.relation_dropout': {'type': 'float', 'min': 0.0, 'max': 0.3, 'step': 0.1},
            
            # Training parameters (only those that exist in current trainer)
            'training.lr': {'type': 'float', 'min': 1e-4, 'max': 1e-2, 'log': True},
            'training.weight_decay': {'type': 'float', 'min': 1e-6, 'max': 1e-4, 'log': True},
            'training.batch_size': {'type': 'categorical', 'values': [128, 256, 512]},
            'training.neg_ratio': {'type': 'int', 'min': 1, 'max': 5},
            'training.use_focal_loss': {'type': 'categorical', 'values': [True, False]},
            'training.label_smoothing': {'type': 'float', 'min': 0.0, 'max': 0.2, 'step': 0.05},
            'training.cosine_tmax': {'type': 'categorical', 'values': [50, 100, 150]},
            
            # Evaluation parameters
            'evaluation.loocv_epochs': {'type': 'int', 'min': 10, 'max': 30, 'step': 5},
            'evaluation.loocv_batch_size': {'type': 'categorical', 'values': [64, 128, 256]},
            'evaluation.loocv_lr': {'type': 'float', 'min': 1e-3, 'max': 1e-2, 'log': True},
            'evaluation.loocv_neg_ratio': {'type': 'int', 'min': 1, 'max': 3},
            
            # Data parameters
            'data.sim_topk': {'type': 'int', 'min': 5, 'max': 30, 'step': 5},
        }
    
    # Run tuning
    print(f"\n🚀 Starting {args.method} optimization...")
    print(f"📦 Search space: {args.search_space}")
    print("=" * 60)
    
    if args.method == 'bayesian':
        best_params, best_auc = run_bayesian_optimization(
            base_config, dataset_info, edges_device, pos_pairs, device, 
            search_space, args.n_trials, args.output_dir
        )
    else:  # grid
        best_params, best_auc = run_grid_search(
            base_config, dataset_info, edges_device, pos_pairs, device, 
            search_space, args.max_combinations, args.output_dir
        )
    
    # Check if tuning was successful
    if best_params is None:
        print("\n❌ Tuning failed! No valid results obtained.")
        print("Check the error messages above for issues.")
        return
    
    # Always update configs/default.yaml with best parameters after successful tuning
    if best_params:
        print("\n🔄 Updating configs/default.yaml with best parameters...")
        update_default_config(best_params)
        print("✅ configs/default.yaml has been automatically updated!")
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Results saved in: {args.output_dir}")
    
    print("\n✨ configs/default.yaml has been automatically updated with best parameters")
    print("You can now run:")
    print("  python3 main.py --mode loocv --config configs/default.yaml")


if __name__ == "__main__":
    main()