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
                  num_val_folds: int = 3,
                  trial: 'optuna.Trial' = None) -> float:
    """
    Quick evaluation using k-fold cross-validation for more robust AUC estimation.
    
    Returns:
        Mean AUC score across validation folds
    """
    # Use k-fold cross-validation for more robust evaluation
    n_folds = 3  # 3-fold CV for balance between speed and reliability
    fold_size = len(pos_pairs) // n_folds
    auc_scores = []
    
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
    
    # Initialize evaluator once
    evaluator = HGATLDAEvaluator(model, device)
    
    # Implement k-fold cross-validation
    for fold in range(n_folds):
        # Split data for this fold
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else len(pos_pairs)
        val_fold = pos_pairs[val_start:val_end]
        train_fold = torch.cat([pos_pairs[:val_start], pos_pairs[val_end:]])
        
        # Generate negatives for this fold
        neg_pairs_fold = generate_negative_pairs(train_fold, dataset_info['num_lncRNAs'], dataset_info['num_diseases'])
        
        # Train with early stopping
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(15):  # Max 15 epochs per fold
            # Train one epoch
            train_loss = trainer.train_epoch(train_fold, neg_pairs_fold, edges)
            
            # Validate
            val_neg = generate_negative_pairs(val_fold, dataset_info['num_lncRNAs'], dataset_info['num_diseases'], len(val_fold))
            val_loss = trainer.validate(val_fold, val_neg, edges)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            # Optuna pruning
            if trial is not None and epoch % 5 == 0:
                intermediate_auc = evaluator.evaluate_auc(val_fold, val_neg, edges)
                trial.report(intermediate_auc, fold * 15 + epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
    
        # Evaluate this fold
        try:
            val_neg = generate_negative_pairs(val_fold, dataset_info['num_lncRNAs'], dataset_info['num_diseases'], len(val_fold))
            fold_auc = evaluator.evaluate_auc(val_fold, val_neg, edges)
            auc_scores.append(fold_auc)
            
            # Reset model for next fold
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
            
            trainer.model = model
            trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['lr']), 
                                                  weight_decay=float(config['training']['weight_decay']))
            evaluator.model = model  # Update evaluator's model reference
        except Exception as e:
            print(f"Fold {fold} evaluation failed: {e}")
            auc_scores.append(0.5)  # Default AUC for failed fold
    
    # Clean up
    del model
    del trainer
    if 'evaluator' in locals():
        del evaluator
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Return mean AUC with penalty for variance (more stable models are better)
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    return mean_auc - 0.01 * std_auc  # Small penalty for high variance


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


def analyze_study_results(study: 'optuna.Study', output_dir: Path):
    """Analyze and visualize Optuna study results for maximum insights."""
    try:
        # Get completed trials
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not trials:
            print("No completed trials to analyze.")
            return
        
        # 1. Best trials summary
        best_trials = sorted(trials, key=lambda t: t.value, reverse=True)[:5]
        with open(output_dir / 'best_trials_summary.txt', 'w') as f:
            f.write("Top 5 Trials by AUC Score\n")
            f.write("=" * 60 + "\n\n")
            for i, trial in enumerate(best_trials, 1):
                f.write(f"Rank {i}: Trial {trial.number}\n")
                f.write(f"AUC Score: {trial.value:.6f}\n")
                f.write("Parameters:\n")
                for key, value in trial.params.items():
                    f.write(f"  {key}: {value}\n")
                f.write("-" * 40 + "\n\n")
        
        # 2. Parameter importance analysis
        if len(trials) >= 10:
            try:
                importance = optuna.importance.get_param_importances(study)
                with open(output_dir / 'parameter_importance.txt', 'w') as f:
                    f.write("Parameter Importance for AUC:\n")
                    f.write("=" * 40 + "\n")
                    sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    for param, imp in sorted_params:
                        f.write(f"{param:30s}: {imp:.4f}\n")
                print(f"\n📊 Parameter importance analysis saved")
            except Exception as e:
                print(f"Could not compute parameter importance: {e}")
        
        print(f"📝 Analysis saved to {output_dir}")
        
    except Exception as e:
        print(f"\n⚠️ Error during analysis: {e}")


def run_bayesian_optimization(base_config: Dict[str, Any],
                             dataset_info: Dict,
                             edges: Dict,
                             pos_pairs: torch.Tensor,
                             device: torch.device,
                             search_space: Dict[str, Any],
                             n_trials: int = 50,
                             output_dir: str = 'tuning_results'):
    """Run Bayesian optimization using Optuna with advanced features."""
    if not OPTUNA_AVAILABLE:
        print("Error: Optuna not installed. Please install with: pip install optuna")
        return None
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup study with advanced settings
    study_name = f"hgat_lda_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Check for existing study database for warm-start
    study_file = output_dir / 'optuna_study.db'
    storage = f'sqlite:///{study_file}'
    
    # Create or load study
    if study_file.exists():
        print(f"\n🔄 Found existing study, loading for warm-start...")
        try:
            # Try to load existing study
            existing_studies = optuna.study.get_all_study_names(storage)
            if existing_studies:
                # Load the most recent study
                study = optuna.load_study(study_name=existing_studies[-1], storage=storage)
                print(f"  Loaded study: {existing_studies[-1]}")
                print(f"  Previous trials: {len(study.trials)}")
                if study.best_trial:
                    print(f"  Previous best AUC: {study.best_value:.4f}")
            else:
                raise ValueError("No studies found")
        except:
            # Create new study if loading fails
            study = optuna.create_study(
                study_name=study_name,
                direction='maximize',
                storage=storage,
                sampler=optuna.samplers.TPESampler(
                    seed=42,
                    n_startup_trials=10,
                    multivariate=True,
                    constant_liar=True
                ),
                pruner=optuna.pruners.HyperbandPruner(
                    min_resource=1,
                    max_resource=45,
                    reduction_factor=3
                )
            )
    else:
        # Create new study
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            storage=storage,
            sampler=optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=10,
                multivariate=True,
                constant_liar=True
            ),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=45,
                reduction_factor=3
            )
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
    
    # Perform comprehensive analysis
    analyze_study_results(study, output_dir)
    
    # Save detailed results
    results = {
        'best_auc': best_trial.value,
        'best_params': best_trial.params,
        'best_trial_number': best_trial.number,
        'total_trials': len(study.trials),
        'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'all_trials': [
            {
                'number': t.number, 
                'value': t.value if t.value is not None else 'pruned',
                'params': t.params,
                'state': str(t.state)
            }
            for t in study.trials
        ]
    }
    
    with open(output_dir / 'optuna_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Export trials dataframe for further analysis
    try:
        df = study.trials_dataframe()
        df.to_csv(output_dir / 'trials_history.csv', index=False)
        print(f"📊 Trials history exported to CSV")
    except:
        pass
    
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
    
    # Define optimized search spaces for maximum AUC
    if args.search_space == 'quick':
        # Quick but focused search on most impactful parameters
        search_space = {
            'model.emb_dim': {'type': 'categorical', 'values': [64, 128]},
            'model.num_layers': {'type': 'categorical', 'values': [2, 3]},  # 2-3 layers usually optimal
            'model.dropout': {'type': 'categorical', 'values': [0.2, 0.3]},  # Moderate dropout
            'model.num_heads': {'type': 'categorical', 'values': [4, 8]},  # Multi-head attention helps
            'training.lr': {'type': 'float', 'min': 3e-4, 'max': 3e-3, 'log': True},  # Focused range
            'training.neg_ratio': {'type': 'categorical', 'values': [3, 5]},
            'training.use_focal_loss': {'type': 'categorical', 'values': [True]},  # Usually better for imbalanced
            'training.label_smoothing': {'type': 'categorical', 'values': [0.1, 0.15]},  # Helps generalization
        }
    elif args.search_space == 'extensive':
        # Comprehensive search with optimized ranges for maximum AUC
        search_space = {
            # Model parameters - optimized ranges
            'model.emb_dim': {'type': 'categorical', 'values': [64, 128, 256]},  # Higher dims for complex patterns
            'model.num_layers': {'type': 'int', 'min': 2, 'max': 4},  # 2-4 layers optimal
            'model.dropout': {'type': 'float', 'min': 0.15, 'max': 0.35, 'step': 0.05},  # Moderate dropout
            'model.num_heads': {'type': 'categorical', 'values': [2, 4, 6, 8]},  # Multi-head attention
            'model.relation_dropout': {'type': 'float', 'min': 0.0, 'max': 0.2, 'step': 0.05},  # Light relation dropout
            'model.use_layernorm': {'type': 'categorical', 'values': [True]},  # Usually helps
            'model.use_residual': {'type': 'categorical', 'values': [True]},  # Helps deeper models
            
            # Training parameters - refined for better convergence
            'training.lr': {'type': 'float', 'min': 1e-4, 'max': 5e-3, 'log': True},  # Focused range
            'training.weight_decay': {'type': 'float', 'min': 1e-6, 'max': 1e-4, 'log': True},
            'training.batch_size': {'type': 'categorical', 'values': [128, 256, 512]},  # Larger batches
            'training.neg_ratio': {'type': 'int', 'min': 3, 'max': 7},  # Balanced negative sampling
            'training.use_focal_loss': {'type': 'categorical', 'values': [True, False]},
            'training.label_smoothing': {'type': 'float', 'min': 0.05, 'max': 0.2, 'step': 0.05},
            'training.cosine_tmax': {'type': 'categorical', 'values': [30, 50, 100]},  # Cosine annealing
            
            # Data parameters - similarity graph tuning
            'data.sim_topk': {'type': 'int', 'min': 10, 'max': 30, 'step': 5},  # Optimal connectivity
            'data.sim_row_normalize': {'type': 'categorical', 'values': [True]},  # Usually better
        }
    else:  # default - balanced search space
        search_space = {
            # Model parameters - balanced ranges
            'model.emb_dim': {'type': 'categorical', 'values': [64, 128]},
            'model.num_layers': {'type': 'int', 'min': 2, 'max': 3},  # Focus on 2-3 layers
            'model.dropout': {'type': 'float', 'min': 0.2, 'max': 0.4, 'step': 0.1},
            'model.num_heads': {'type': 'categorical', 'values': [2, 4, 6]},
            'model.relation_dropout': {'type': 'float', 'min': 0.0, 'max': 0.2, 'step': 0.1},
            
            # Training parameters - balanced for good convergence
            'training.lr': {'type': 'float', 'min': 3e-4, 'max': 3e-3, 'log': True},  # Narrower range
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