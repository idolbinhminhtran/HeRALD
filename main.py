#!/usr/bin/env python3
"""
Main entry point for HeRALD: Heterogeneous Relational Attention for lncRNA–Disease Association Prediction.
Supports training, evaluation, and LOOCV modes.
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

from data.data_loader import load_dataset, get_dataset_info
from data.graph_construction import construct_heterogeneous_graph, get_positive_pairs, generate_negative_pairs
from models.hgat_lda import HGAT_LDA
from training.trainer import HGATLDATrainer
from training.evaluator import HGATLDAEvaluator
from utils.metrics import calculate_metrics, calculate_loocv_statistics, plot_loocv_results


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
        sim_threshold=config['data'].get('threshold', 0.0),
        sim_mutual=config['data'].get('sim_mutual', False),
        sim_sym=config['data'].get('sim_sym', True),
        sim_row_norm=config['data'].get('sim_row_norm', True),
        sim_tau=config['data'].get('sim_tau', None),
        use_bipartite_edge_weight=config['data'].get('use_bipartite_edge_weight', False),
        verbose=True
    )
    
    pos_pairs = get_positive_pairs(data_dict['lnc_disease_assoc'])
    print(f"Number of positive pairs: {len(pos_pairs)}")
    
    return data_dict, dataset_info, edges, pos_pairs


def run_training(config: Dict[str, Any], dataset_info: Dict, edges: Dict, pos_pairs: torch.Tensor, device: torch.device, results_path: str):
    """Run training mode with multi-GPU support."""
    print("\n🚀 Starting training...")
    
    # Check for multi-GPU availability
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"🌐 Detected {n_gpus} GPUs - enabling DataParallel multi-GPU training")
    
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
        relation_dropout=float(config['model'].get('relation_dropout', 0.2)),
        use_layernorm=config['model'].get('use_layernorm', True),
        use_residual=config['model'].get('use_residual', True),
        use_relation_norm=config['model'].get('use_relation_norm', True)
    ).to(device)
    
    # Create trainer with multi-GPU support
    trainer = HGATLDATrainer(
        model=model,
        device=device,
        lr=float(config['training']['lr']),
        batch_size=int(config['training']['batch_size']),
        neg_ratio=int(config['training']['neg_ratio']),
        use_focal_loss=config['training'].get('use_focal_loss', False),
        label_smoothing=config['training'].get('label_smoothing', 0.0),
        use_multi_gpu=True,  # Enable multi-GPU by default
        loss_type=config['training'].get('loss_type', 'bce'),
        pairwise_type=config['training'].get('pairwise_type', 'bpr'),
        use_hard_negatives=config['training'].get('use_hard_negatives', True)
    )
    
    # Generate negative pairs
    neg_pairs_all = generate_negative_pairs(pos_pairs, dataset_info['num_lncRNAs'], dataset_info['num_diseases'])
    
    # Train model
    best_model_path = os.path.join(results_path, 'best_model.pth')
    history = trainer.train(
        pos_pairs=pos_pairs,
        neg_pairs_all=neg_pairs_all,
        edges=edges,
        num_epochs=int(config['training']['num_epochs']),
        val_split=float(config['training']['val_split']),
        early_stopping_patience=int(config['training']['early_stopping_patience']),
        save_path=best_model_path
    )
    
    # Save training history
    history_path = os.path.join(results_path, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✅ Training completed! Best model saved to: {best_model_path}")
    print(f"📊 Training history saved to: {history_path}")
    
    return model, best_model_path


def run_evaluation(config: Dict[str, Any], dataset_info: Dict, edges: Dict, pos_pairs: torch.Tensor, device: torch.device, results_path: str, model_path: str = None):
    """Run evaluation mode."""
    print("\n🔍 Starting evaluation...")
    
    # Get evaluation settings
    full_ranking = config.get('eval', {}).get('full_ranking', True)
    print(f"Evaluation settings: neg_sampling=False, full_ranking={full_ranking}")
    
    if model_path and os.path.exists(model_path):
        # Load pre-trained model
        print(f"Loading pre-trained model from: {model_path}")
        model = HGAT_LDA(
            num_lncRNAs=dataset_info['num_lncRNAs'],
            num_genes=dataset_info['num_genes'],
            num_diseases=dataset_info['num_diseases'],
            edges=edges,
            emb_dim=int(config['model']['emb_dim']),
            num_layers=int(config['model']['num_layers']),
            dropout=float(config['model']['dropout']),
            num_heads=int(config['model'].get('num_heads', 4)),
            relation_dropout=float(config['model'].get('relation_dropout', 0.2)),
            use_layernorm=config['model'].get('use_layernorm', True),
            use_residual=config['model'].get('use_residual', True)
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No pre-trained model found. Please run training first or provide a valid model path.")
        return
    
    # Create evaluator with full_ranking setting
    evaluator = HGATLDAEvaluator(model, device, full_ranking=full_ranking)
    
    # Generate negative pairs for evaluation
    neg_pairs_all = generate_negative_pairs(pos_pairs, dataset_info['num_lncRNAs'], dataset_info['num_diseases'])
    
    # Evaluate all metrics
    metrics = evaluator.evaluate_all_metrics(
        pos_pairs, neg_pairs_all, edges,
        num_lncRNAs=dataset_info['num_lncRNAs'] if full_ranking else None,
        num_diseases=dataset_info['num_diseases'] if full_ranking else None
    )
    
    # Save evaluation results
    results = {
        'metrics': metrics,
        'config': config,
        'dataset_info': dataset_info
    }
    
    results_path_json = os.path.join(results_path, 'evaluation_results.json')
    with open(results_path_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Evaluation completed! Results saved to: {results_path_json}")
    print(f"📊 Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")


def run_loocv(config: Dict[str, Any], dataset_info: Dict, edges: Dict, pos_pairs: torch.Tensor, device: torch.device, results_path: str, data_dict: Dict[str, torch.Tensor]):
    """Run LOOCV mode with disease rewrite support."""
    print("\n🔄 Starting LOOCV...")
    
    # Check evaluation protocol
    eval_protocol = config.get('eval', {}).get('protocol', 'loocv_pair')
    rewrite_isolated = config.get('eval', {}).get('rewrite_isolated', True)
    full_ranking = config.get('eval', {}).get('full_ranking', True)
    
    if eval_protocol != 'loocv_pair':
        print(f"⚠️  Note: Using traditional LOOCV. Set eval.protocol: loocv_pair in config for per-pair LOOCV with rewrite.")
        # Fall back to traditional LOOCV
        return run_loocv_traditional(config, dataset_info, edges, pos_pairs, device, results_path)
    
    print(f"📊 Configuration:")
    print(f"   - Protocol: {eval_protocol}")
    print(f"   - Rewrite isolated diseases: {rewrite_isolated}")
    print(f"   - Full ranking: {full_ranking}")
    print(f"   - Epochs per fold: {config['training']['num_epochs']}")
    print(f"   - Batch size: {config['training']['batch_size']}")
    print(f"   - Learning rate: {config['training']['lr']}")
    print(f"   - Device: {device}")
    
    # Create model (will be recreated for each fold)
    model = HGAT_LDA(
        num_lncRNAs=dataset_info['num_lncRNAs'],
        num_genes=dataset_info['num_genes'],
        num_diseases=dataset_info['num_diseases'],
        edges=edges,
        emb_dim=int(config['model']['emb_dim']),
        num_layers=int(config['model']['num_layers']),
        dropout=float(config['model']['dropout']),
        num_heads=int(config['model'].get('num_heads', 4)),
        relation_dropout=float(config['model'].get('relation_dropout', 0.2)),
        use_layernorm=config['model'].get('use_layernorm', True),
        use_residual=config['model'].get('use_residual', True),
        use_relation_norm=config['model'].get('use_relation_norm', True)
    ).to(device)
    
    # Create evaluator with full_ranking setting
    evaluator = HGATLDAEvaluator(model, device, full_ranking=full_ranking)
    
    # Run LOOCV with rewrite support
    results = evaluator.run_loocv_with_rewrite(
        pos_pairs=pos_pairs,
        edges=edges,
        data_dict=data_dict,
        num_lncRNAs=dataset_info['num_lncRNAs'],
        num_diseases=dataset_info['num_diseases'],
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'],
        lr=config['training']['lr'],
        neg_ratio=config['training']['neg_ratio'],
        weight_decay=config['training'].get('weight_decay', 1e-5),
        use_focal_loss=config['training'].get('use_focal_loss', True),
        label_smoothing=config['training'].get('label_smoothing', 0.1),
        cosine_tmax=config['training'].get('cosine_tmax', None),
        rewrite_isolated=rewrite_isolated,
        sim_topk=config['data'].get('sim_topk', None),
        sim_row_normalize=config['data'].get('sim_row_normalize', True),
        sim_threshold=config['data'].get('threshold', 0.0),
        full_ranking=full_ranking
    )
    
    # Print the required message format
    print(f"\nLOOCV folds: {results['num_folds']}, rewrite applied: {results['num_rewrites']}")
    
    # Save results
    results_path_json = os.path.join(results_path, 'loocv_results.json')
    with open(results_path_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ LOOCV completed! Results saved to: {results_path_json}")
    print(f"📊 Final Statistics:")
    for key, value in results['macro_averages'].items():
        if key.endswith('_std'):
            continue
        std_key = f"{key}_std"
        if std_key in results['macro_averages']:
            print(f"   {key.upper()}: {value:.4f} ± {results['macro_averages'][std_key]:.4f}")
        else:
            print(f"   {key.upper()}: {value:.4f}")


def run_loocv_traditional(config: Dict[str, Any], dataset_info: Dict, edges: Dict, pos_pairs: torch.Tensor, device: torch.device, results_path: str):
    """Run traditional LOOCV mode (fallback)."""
    print("\n🔄 Running traditional LOOCV...")
    
    # Detect and optimize for available GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"🚀 Detected {n_gpus} GPUs - Optimizing for parallel execution")
        print(f"   - DataParallel will automatically distribute batches across GPUs")
    
    print(f"📊 Configuration (using main training parameters):")
    print(f"   - Epochs: {config['training']['num_epochs']}")
    print(f"   - Batch size: {config['training']['batch_size']}")
    print(f"   - Learning rate: {config['training']['lr']}")
    print(f"   - Negative ratio: {config['training']['neg_ratio']}")
    print(f"   - Focal loss: {config['training'].get('use_focal_loss', True)}")
    print(f"   - Label smoothing: {config['training'].get('label_smoothing', 0.1)}")
    print(f"   - Device: {device} (GPUs available: {n_gpus})")
    
    # Create model (will be recreated for each fold)
    model = HGAT_LDA(
        num_lncRNAs=dataset_info['num_lncRNAs'],
        num_genes=dataset_info['num_genes'],
        num_diseases=dataset_info['num_diseases'],
        edges=edges,
        emb_dim=int(config['model']['emb_dim']),
        num_layers=int(config['model']['num_layers']),
        dropout=float(config['model']['dropout']),
        num_heads=int(config['model'].get('num_heads', 4)),
        relation_dropout=float(config['model'].get('relation_dropout', 0.2)),
        use_layernorm=config['model'].get('use_layernorm', True),
        use_residual=config['model'].get('use_residual', True),
        use_relation_norm=config['model'].get('use_relation_norm', True)
    ).to(device)
    
    # Create evaluator
    evaluator = HGATLDAEvaluator(model, device)
    
    # Run LOOCV using main training parameters from config
    auc_scores = evaluator.leave_one_out_cross_validation(
        pos_pairs=pos_pairs,
        edges=edges,
        num_lncRNAs=dataset_info['num_lncRNAs'],
        num_diseases=dataset_info['num_diseases'],
        num_epochs=config['training']['num_epochs'],  # Use main training epochs
        batch_size=config['training']['batch_size'],  # Use main training batch size
        lr=config['training']['lr'],  # Use main training learning rate
        neg_ratio=config['training']['neg_ratio'],  # Use main training neg_ratio
        weight_decay=config['training'].get('weight_decay', 1e-5),  # Pass weight decay
        use_focal_loss=config['training'].get('use_focal_loss', True),  # Pass focal loss
        label_smoothing=config['training'].get('label_smoothing', 0.1),  # Pass label smoothing
        cosine_tmax=config['training'].get('cosine_tmax', None)  # Pass cosine annealing
    )
    
    # Calculate statistics
    stats = calculate_loocv_statistics(auc_scores)
    
    # Save results
    results = {
        'auc_scores': auc_scores,
        'statistics': stats,
        'config': config,
        'dataset_info': dataset_info
    }
    
    results_path_json = os.path.join(results_path, 'loocv_results.json')
    with open(results_path_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create plots
    plot_path = os.path.join(results_path, 'loocv_results.png')
    plot_loocv_results(auc_scores, save_path=plot_path)
    
    print(f"✅ LOOCV completed! Results saved to: {results_path_json}")
    print(f"📊 LOOCV Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value:.4f}")
    print(f"📈 Plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='HeRALD: Heterogeneous Relational Attention for lncRNA–Disease Association Prediction')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'loocv'], required=True,
                       help='Mode to run: train, evaluate, or loocv')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Configuration file path')
    parser.add_argument('--results_path', type=str, default='results',
                       help='Results directory')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pre-trained model (for evaluate mode)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"🖥️ Using device: {device}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create results directory
    results_dir = Path(args.results_path)
    results_dir.mkdir(exist_ok=True)
    
    # Load data and construct graph
    data_dict, dataset_info, edges, pos_pairs = load_data_and_graph(config)
    
    # Move edges to device
    edges_device = {}
    for key, (src, dst, w) in edges.items():
        if src is not None:
            src = src.to(device)
            dst = dst.to(device)
            if w is not None:
                w = w.to(device)
        edges_device[key] = (src, dst, w)
    
    # Run selected mode
    if args.mode == 'train':
        model, model_path = run_training(config, dataset_info, edges_device, pos_pairs, device, args.results_path)
        print(f"\n🎉 Training completed successfully!")
        print(f"💡 You can now run evaluation with:")
        print(f"   python3 main.py --mode evaluate --config {args.config} --results_path {args.results_path} --model_path {model_path}")
        
    elif args.mode == 'evaluate':
        run_evaluation(config, dataset_info, edges_device, pos_pairs, device, args.results_path, args.model_path)
        
    elif args.mode == 'loocv':
        run_loocv(config, dataset_info, edges_device, pos_pairs, device, args.results_path, data_dict)
        
    else:
        print(f"❌ Unknown mode: {args.mode}")
        return


if __name__ == "__main__":
    main()