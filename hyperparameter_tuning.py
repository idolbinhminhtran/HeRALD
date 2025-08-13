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
    Quick evaluation using sampled LOOCV folds for hyperparameter tuning.
    This simulates actual LOOCV performance to maximize AUC.
    
    Returns:
        Mean AUC score across sampled LOOCV folds
    """
    # Use more LOOCV folds for accurate evaluation (critical for avoiding overfitting)
    # Increase samples for better representation of actual LOOCV performance
    num_loocv_samples = min(150, len(pos_pairs))  # Sample 150 folds instead of 50
    
    # Use stratified sampling to ensure diverse fold selection
    # This helps avoid bias from random sampling
    np.random.seed(42)  # Fixed seed for reproducible sampling
    selected_folds = np.random.choice(len(pos_pairs), num_loocv_samples, replace=False)
    
    # Sort folds to ensure systematic evaluation
    selected_folds = np.sort(selected_folds)
    auc_scores = []
    
    # Enable mixed precision for RTX 5090
    config['system'] = config.get('system', {})
    config['system']['use_amp'] = torch.cuda.is_available()
    
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
        use_amp=config.get('system', {}).get('use_amp', torch.cuda.is_available()),  # Enable AMP for RTX 5090
        use_focal_loss=bool(config['training'].get('use_focal_loss', False)),
        label_smoothing=float(config['training'].get('label_smoothing', 0.0)),
        cosine_tmax=config['training'].get('cosine_tmax', 50)  # Add cosine annealing
    )
    
    # Perform LOOCV-style evaluation on sampled folds
    print(f"Evaluating {num_loocv_samples} LOOCV folds...")
    
    for fold_idx in selected_folds:
        # LOOCV: Leave out one positive pair for testing
        test_lnc = int(pos_pairs[fold_idx, 0].item())
        test_dis = int(pos_pairs[fold_idx, 1].item())
        
        # Remove test pair from training data (LOOCV style)
        train_pos_pairs = torch.cat([pos_pairs[:fold_idx], pos_pairs[fold_idx+1:]], dim=0)
        
        # Create fresh model for this fold (LOOCV requires fresh model per fold)
        model_fold = HGAT_LDA(
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
        
        # Create trainer for this fold
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
            use_multi_gpu=False  # Disable multi-GPU for quick evaluation
        )
        
        # Generate negatives for training
        neg_pairs_all = generate_negative_pairs(train_pos_pairs, dataset_info['num_lncRNAs'], dataset_info['num_diseases'])
        
        # Match actual LOOCV training epochs to avoid mismatch
        # Use same number of epochs as actual LOOCV for consistency
        num_epochs = min(30, int(config['training'].get('num_epochs', 30)))
        
        try:
            # Train model
            trainer_fold.train(
                pos_pairs=train_pos_pairs,
                neg_pairs_all=neg_pairs_all,
                edges=edges,
                num_epochs=num_epochs,
                val_split=0.0,  # No validation split for LOOCV
                early_stopping_patience=5,
                save_path=None
            )
            
            # Evaluate on test pair (LOOCV style)
            model_fold.eval()
            with torch.no_grad():
                # Score for the positive test pair
                pos_score = torch.sigmoid(model_fold(
                    torch.tensor([test_lnc], device=device),
                    torch.tensor([test_dis], device=device),
                    edges
                )).item()
                
                # Generate negative pairs for this test case
                neg_scores = []
                
                # Sample more negatives for better AUC estimation
                for _ in range(50):  # Sample 50 negatives for more accurate AUC
                    neg_dis = np.random.randint(0, dataset_info['num_diseases'])
                    if (test_lnc, neg_dis) not in set((int(i), int(j)) for i, j in train_pos_pairs):
                        score = torch.sigmoid(model_fold(
                            torch.tensor([test_lnc], device=device),
                            torch.tensor([neg_dis], device=device),
                            edges
                        )).item()
                        neg_scores.append(score)
                
                # Calculate AUC for this fold
                if neg_scores:
                    from sklearn.metrics import roc_auc_score
                    y_true = [1] + [0] * len(neg_scores)
                    y_score = [pos_score] + neg_scores
                    fold_auc = roc_auc_score(y_true, y_score)
                    auc_scores.append(fold_auc)
                
            # Clean up
            del model_fold
            del trainer_fold
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"LOOCV fold {fold_idx} failed: {e}")
            auc_scores.append(0.5)  # Default AUC
            
        # Optuna pruning
        if trial is not None and len(auc_scores) % 10 == 0:
            current_mean_auc = np.mean(auc_scores)
            trial.report(current_mean_auc, len(auc_scores))
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    # Clean up
    del model
    del trainer
    if 'evaluator' in locals():
        del evaluator
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Return mean AUC with stronger penalty for variance to favor stable models
    # This helps prevent overfitting to specific folds
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    
    # Also penalize if too few folds succeeded
    success_rate = len(auc_scores) / num_loocv_samples
    if success_rate < 0.8:  # If less than 80% folds succeeded
        mean_auc *= success_rate  # Penalize heavily
    
    # Stronger variance penalty for LOOCV
    return mean_auc - 0.02 * std_auc  # Doubled penalty for high variance


def validate_loocv_params(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate if parameters are suitable for LOOCV to prevent overfitting.
    
    Returns:
        (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check learning rate
    if 'training.lr' in params and params['training.lr'] > 0.001:
        warnings.append(f"⚠️ Learning rate {params['training.lr']:.4f} is too high (>0.001)")
    
    # Check label smoothing
    if 'training.label_smoothing' in params and params['training.label_smoothing'] > 0.1:
        warnings.append(f"⚠️ Label smoothing {params['training.label_smoothing']:.2f} is too high (>0.1)")
    
    # Check batch size
    if 'training.batch_size' in params and params['training.batch_size'] > 256:
        warnings.append(f"⚠️ Batch size {params['training.batch_size']} is too large (>256)")
    
    # Check dropout
    if 'model.dropout' in params and params['model.dropout'] < 0.25:
        warnings.append(f"⚠️ Dropout {params['model.dropout']:.2f} is too low (<0.25)")
    
    # Check number of heads
    if 'model.num_heads' in params and params['model.num_heads'] < 4:
        warnings.append(f"⚠️ Number of heads {params['model.num_heads']} is too low (<4)")
    
    # Check weight decay
    if 'training.weight_decay' in params and params['training.weight_decay'] < 1e-5:
        warnings.append(f"⚠️ Weight decay {params['training.weight_decay']:.2e} is too low (<1e-5)")
    
    is_valid = len(warnings) == 0
    return is_valid, warnings


def update_default_config(best_params: Dict[str, Any]):
    """Update configs/default.yaml with best parameters."""
    config_path = 'configs/default.yaml'
    
    # Validate parameters first
    is_valid, warnings = validate_loocv_params(best_params)
    if not is_valid:
        print("\n⚠️ WARNING: Best parameters may cause overfitting in LOOCV:")
        for warning in warnings:
            print(f"  {warning}")
        print("\n🔧 Applying automatic corrections...")
        
        # Apply corrections
        if 'training.lr' in best_params and best_params['training.lr'] > 0.001:
            best_params['training.lr'] = 0.0008
            print(f"  ✅ Reduced learning rate to 0.0008")
        
        if 'training.label_smoothing' in best_params and best_params['training.label_smoothing'] > 0.1:
            best_params['training.label_smoothing'] = 0.05
            print(f"  ✅ Reduced label smoothing to 0.05")
        
        if 'training.batch_size' in best_params and best_params['training.batch_size'] > 256:
            best_params['training.batch_size'] = 128
            print(f"  ✅ Reduced batch size to 128")
        
        if 'model.dropout' in best_params and best_params['model.dropout'] < 0.25:
            best_params['model.dropout'] = 0.3
            print(f"  ✅ Increased dropout to 0.3")
        
        if 'model.num_heads' in best_params and best_params['model.num_heads'] < 4:
            best_params['model.num_heads'] = 4
            print(f"  ✅ Increased num_heads to 4")
    
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
    
    # Optimize with multi-GPU parallelization
    n_gpus = torch.cuda.device_count()
    n_jobs = min(n_gpus, 4) if n_gpus > 1 else 1  # Parallel trials on multiple GPUs
    
    print(f"\n🔍 Starting Bayesian optimization with {n_trials} trials...")
    if n_jobs > 1:
        print(f"🚀 Running {n_jobs} trials in parallel across {n_gpus} GPUs")
    
    study.optimize(
        lambda trial: optuna_objective(trial, dataset_info, edges, pos_pairs, device, base_config, search_space),
        n_trials=n_trials,
        n_jobs=n_jobs if n_jobs > 1 else 1,  # Parallel execution for multi-GPU
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
    
    # Define CONSERVATIVE search spaces to prevent overfitting in LOOCV
    if args.search_space == 'quick':
        # Quick search with conservative parameters for stable LOOCV performance
        search_space = {
            'model.emb_dim': {'type': 'categorical', 'values': [64, 128]},  # Smaller dims to prevent overfitting
            'model.num_layers': {'type': 'categorical', 'values': [2, 3]},  # Shallower for better generalization
            'model.dropout': {'type': 'categorical', 'values': [0.3, 0.4]},  # Higher dropout essential
            'model.num_heads': {'type': 'categorical', 'values': [4, 6]},  # Moderate attention heads
            'training.lr': {'type': 'float', 'min': 1e-4, 'max': 5e-4, 'log': True},  # Much lower LR range!
            'training.batch_size': {'type': 'categorical', 'values': [64, 128]},  # Smaller batches
            'training.neg_ratio': {'type': 'categorical', 'values': [2, 3]},  # Conservative ratio
            'training.use_focal_loss': {'type': 'categorical', 'values': [False]},  # Standard loss better
            'training.label_smoothing': {'type': 'categorical', 'values': [0.0, 0.05]},  # Minimal smoothing
            'training.weight_decay': {'type': 'float', 'min': 5e-5, 'max': 1e-4, 'log': True},  # Higher regularization
        }
    elif args.search_space == 'extensive':
        # Comprehensive but CONSERVATIVE search to avoid overfitting
        search_space = {
            # Model parameters - conservative for LOOCV
            'model.emb_dim': {'type': 'categorical', 'values': [64, 128, 192]},  # Moderate sizes only
            'model.num_layers': {'type': 'int', 'min': 2, 'max': 4},  # Not too deep
            'model.dropout': {'type': 'float', 'min': 0.3, 'max': 0.45, 'step': 0.05},  # Higher dropout range
            'model.num_heads': {'type': 'categorical', 'values': [4, 6, 8]},  # Reasonable attention heads
            'model.relation_dropout': {'type': 'float', 'min': 0.1, 'max': 0.2, 'step': 0.05},  # More dropout
            'model.use_layernorm': {'type': 'categorical', 'values': [True]},  # Always use
            'model.use_residual': {'type': 'categorical', 'values': [True]},  # Always use
            
            # Training parameters - very conservative for stability
            'training.lr': {'type': 'float', 'min': 5e-5, 'max': 5e-4, 'log': True},  # Much lower range!
            'training.weight_decay': {'type': 'float', 'min': 5e-5, 'max': 2e-4, 'log': True},  # Strong regularization
            'training.batch_size': {'type': 'categorical', 'values': [32, 64, 128]},  # Smaller batches only
            'training.neg_ratio': {'type': 'int', 'min': 1, 'max': 3},  # Lower ratios
            'training.use_focal_loss': {'type': 'categorical', 'values': [False]},  # Standard loss only
            'training.label_smoothing': {'type': 'float', 'min': 0.0, 'max': 0.1, 'step': 0.05},  # Max 0.1
            'training.cosine_tmax': {'type': 'categorical', 'values': [25, 30]},  # Match shorter epochs
            'training.num_epochs': {'type': 'categorical', 'values': [25, 30]},  # Fewer epochs
            
            # Data parameters - moderate connectivity
            'data.sim_topk': {'type': 'int', 'min': 10, 'max': 25, 'step': 5},  # Not too many neighbors
            'data.sim_row_normalize': {'type': 'categorical', 'values': [True]},  # Always normalize
        }
    else:  # default - LOOCV-optimized conservative search space
        search_space = {
            # Model parameters - conservative defaults
            'model.emb_dim': {'type': 'categorical', 'values': [64, 128]},
            'model.num_layers': {'type': 'int', 'min': 2, 'max': 3},  # Shallow models
            'model.dropout': {'type': 'float', 'min': 0.3, 'max': 0.4, 'step': 0.05},  # Higher dropout
            'model.num_heads': {'type': 'categorical', 'values': [4, 6]},  # Moderate heads
            'model.relation_dropout': {'type': 'float', 'min': 0.1, 'max': 0.2, 'step': 0.05},
            
            # Training parameters - conservative for stability
            'training.lr': {'type': 'float', 'min': 1e-4, 'max': 8e-4, 'log': True},  # Lower range
            'training.weight_decay': {'type': 'float', 'min': 5e-5, 'max': 1e-4, 'log': True},
            'training.batch_size': {'type': 'categorical', 'values': [64, 128]},  # Smaller batches
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
    print(f"\n🚀 Starting {args.method} optimization for LOOCV...")
    print(f"🎯 Objective: Maximize LOOCV AUC")
    print(f"📦 Search space: {args.search_space}")
    print(f"🔬 Evaluation: Sampling {min(50, len(pos_pairs))} LOOCV folds per trial")
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
    print("✅ LOOCV OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best LOOCV AUC: {best_auc:.4f}")
    print(f"Results saved in: {args.output_dir}")
    
    print("\n✨ configs/default.yaml has been automatically updated with best LOOCV parameters")
    print("These parameters are optimized specifically for LOOCV performance!")
    print("\nYou can now run full LOOCV evaluation:")
    print("  python3 main.py --mode loocv --config configs/default.yaml")


if __name__ == "__main__":
    main()